import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import mediapipe as mp_hands 
import torch
import torchcrepe
import sounddevice as sd
import numpy as np
import threading
import mido
from musicpy import *

try:
    outport = mido.open_output('AirPianoPort 1')
except:
    outport = mido.open_output('Microsoft GS Wavetable Synth 0')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FS = 16000
current_tonic = "C"
current_volume = 0
lock = threading.Lock()
M_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'G', 'G#', 'A', 'A#', 'B'] 
ROMANS = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']

octave_shift = 0      
scale_locked = False
left_pinch_prev = [False, False, False, False]
current_right_finger = -1 
active_right_notes = []
current_chord_label = "None"

def audio_worker():
    global current_tonic, scale_locked, current_volume
    
    def callback(indata, frames, time, status):
        global current_tonic, current_volume
        if scale_locked: return 
        
        audio_data = indata[:, 0].copy()
        rms = np.sqrt(np.mean(audio_data**2))
        current_volume = rms * 500
        
        if rms < 0.02: return 
        
        with torch.no_grad():
            try:
                audio_tensor = torch.from_numpy(audio_data).float().to(DEVICE).unsqueeze(0)
                pitch = torchcrepe.predict(audio_tensor, FS, 1024, 50, 800, 'tiny', 
                                           decoder=torchcrepe.decode.argmax, device=DEVICE)
                f0 = torch.mean(pitch).item() if pitch.numel() > 1 else pitch.item()
                
                if f0 > 60:
                    midi_num = int(round(69 + 12 * np.log2(f0 / 440.0)))
                    new_tonic = M_NOTES[midi_num % 12]
                    with lock:
                        current_tonic = new_tonic
            except:
                pass

    with sd.InputStream(channels=1, callback=callback, blocksize=2048, samplerate=FS):
        while True: sd.sleep(1000)

threading.Thread(target=audio_worker, daemon=True).start()

def get_musicpy_chord(tonic_name, y_pos, finger_idx, octave_val):
    global current_chord_label
    current_scale = scale(tonic_name, 'major')
    degree = finger_idx if y_pos < 0.5 else 4 + finger_idx
    if degree > 6: degree = 0     
    my_chord = current_scale.pick_chord_by_degree(degree, num=3)
    
    try:
        raw_name = my_chord.names()[0] if callable(my_chord.names) else my_chord.names[0]
    except:
        raw_name = str(my_chord).replace('[chord] ', '')
    current_chord_label = f"{ROMANS[degree]}: {raw_name}"

    oct_offset = (octave_val // 12)
    return [(n.degree + (oct_offset * 12)) for n in my_chord.notes]

def draw_piano_ui(img, active_notes):
    h, w, _ = img.shape
    key_h = 100
    num_white_keys = 21 
    w_w = w // num_white_keys
    start_midi = 48
    white_indices = [0, 2, 4, 5, 7, 9, 11]
    
    for i in range(num_white_keys):
        m_val = start_midi + (i // 7 * 12) + white_indices[i % 7]
        color = (0, 255, 0) if m_val in active_notes else (255, 255, 255)
        cv2.rectangle(img, (i*w_w, h-key_h), ((i+1)*w_w, h), color, -1)
        cv2.rectangle(img, (i*w_w, h-key_h), ((i+1)*w_w, h), (180, 180, 180), 1)

    black_offsets = {0:1, 1:3, 3:6, 4:8, 5:10}
    b_w, b_h = int(w_w * 0.6), int(key_h * 0.6)
    for i in range(num_white_keys):
        idx = i % 7
        if idx in black_offsets:
            m_val = start_midi + (i // 7 * 12) + black_offsets[idx]
            color = (0, 180, 0) if m_val in active_notes else (40, 40, 40)
            x_mid = (i + 1) * w_w
            cv2.rectangle(img, (x_mid - b_w//2, h-key_h), (x_mid + b_w//2, h-key_h + b_h), color, -1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hands_detector = mp_hands.solutions.hands.Hands(
    max_num_hands=2, min_detection_confidence=0.6, model_complexity=0
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    with lock: 
        tonic, cur_oct = current_tonic, octave_shift
    
    results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cv2.rectangle(frame, (0,0), (w, 80), (20,20,20), -1)
    l_color = (0, 0, 255) if scale_locked else (0, 255, 0)
    cv2.putText(frame, f"KEY: {tonic} MAJOR", (20, 35), 0, 0.7, l_color, 2)
    
    vol_w = int(np.clip(current_volume, 0, 150))
    cv2.rectangle(frame, (20, 42), (20 + vol_w, 47), (0, 255, 255), -1)
    
    cv2.putText(frame, f"OCT: {cur_oct//12}", (w-120, 35), 0, 0.7, (255, 255, 0), 2)
    
    if current_right_finger != -1:
        cv2.putText(frame, f"CHORD: {current_chord_label}", (20, 70), 0, 0.8, (255, 255, 255), 2)
    
    draw_piano_ui(frame, active_right_notes)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            lbl = results.multi_handedness[idx].classification[0].label
            thumb, tips = hand_landmarks.landmark[4], [8, 12, 16, 20]

            if lbl == 'Left':
                for i, t_id in enumerate(tips):
                    tip = hand_landmarks.landmark[t_id]
                    dist = np.sqrt((thumb.x - tip.x)**2 + (thumb.y - tip.y)**2)
                    if dist < 0.04 and not left_pinch_prev[i]:
                        if i == 0: octave_shift = 12 if octave_shift != 12 else 0
                        if i == 1: octave_shift = -12 if octave_shift != -12 else 0
                        if i == 2: scale_locked = True
                        if i == 3: scale_locked = False
                    left_pinch_prev[i] = dist < 0.04

            if lbl == 'Right':
                if current_right_finger != -1:
                    t_tip = hand_landmarks.landmark[tips[current_right_finger]]
                    if np.sqrt((thumb.x - t_tip.x)**2 + (thumb.y - t_tip.y)**2) > 0.05:
                        for n in active_right_notes:
                            if outport: outport.send(mido.Message('note_off', note=n))
                        active_right_notes, current_right_finger = [], -1
                else:
                    for i, t_id in enumerate(tips):
                        tip = hand_landmarks.landmark[t_id]
                        if np.sqrt((thumb.x - tip.x)**2 + (thumb.y - tip.y)**2) < 0.04:
                            current_right_finger = i
                            active_right_notes = get_musicpy_chord(tonic, thumb.y, i, cur_oct)
                            if outport:
                                for n in active_right_notes:
                                    outport.send(mido.Message('note_on', note=n, velocity=100))
                            break

    cv2.imshow('Musicpy Air Piano [MIDI VERSION]', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
