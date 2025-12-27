# VocalCV-Accompanist 🎹🎤
> **Real-time AI Accompaniment: Fusing Deep Learning Pitch Estimation with Computer Vision Gesture Mapping.**

VocalCV-Accompanist is a sophisticated Python-based instrument that transforms vocal input into a harmonic foundation. It utilizes a **Convolutional Neural Network (CNN)** for pitch tracking and **Mediapipe's Perception Pipeline** for 3D hand landmarking, all tied together by a custom **Diatonic Logic Engine**.

---

## 🏗️ Technical Architecture

The system operates across three parallel processing threads to ensure low-latency performance:

1.  **Vocal Processing Thread (Digital Signal Processing):** * **Capture:** 16kHz Mono Pulse-Code Modulation (PCM) stream.
    * **Inference:** Uses **TorchCrepe** (a implementation of the CREPE model) for state-of-the-art pitch estimation.
    * **Mapping:** Converts estimated Frequency ($f_0$) to MIDI numbers using the formula:  
        $$n = 69 + 12 \log_2\left(\frac{f_0}{440}\right)$$
    * **Update:** Overwrites the global `current_tonic` variable via thread-safe locking.

2.  **Computer Vision Thread (Mediapipe):**
    * **Landmarking:** Detects 21 3D hand-knuckle coordinates.
    * **Heuristics:** Calculates Euclidean distance between `THUMB_TIP` (Id 4) and `FINGER_TIPS` (Ids 8, 12, 16, 20) to detect binary "pinch" states.
    * **Normalization:** Maps the Thumb's Y-coordinate ($0.0 - 1.0$) to the musical "Zone" (Lower/Upper) to select between Tonic/Subdominant and Dominant functions.

3.  **Main/UI Thread (OpenCV & Musicpy):**
    * **Theory Engine:** On pinch detection, it initializes a `musicpy.scale` object and performs a `pick_chord_by_degree` lookup.
    * **Rendering:** UI overlay is drawn via OpenCV, displaying the active Scale, Roman Numeral Notation, and the Visual Volume Meter (RMS-based).



---

## 🎼 Music Theory Implementation

The project uses **Diatonic Set Theory**. When a key is detected (e.g., G), the engine builds a Major Scale:
`Scale('G', 'major') -> [G, A, B, C, D, E, F#]`

The hand position then selects the chord degree:
* **Index Pinch (Lower Zone):** Degree 0 → **I Major** (G - B - D)
* **Index Pinch (Upper Zone):** Degree 4 → **V Major** (D - F# - A)



---

## 📂 Repository Breakdown

| File | Technical Role | Engine / Library |
| :--- | :--- | :--- |
| `VocalCV_MIDI.py` | Asynchronous MIDI Messaging | `mido` / `python-rtmidi` |
| `VocalCV_SoundFont.py` | Real-time Wavetable Synthesis | `sf2_loader` / `pydub` |
| `Exporter_WAV.py` | Pre-rendering & Caching | `AudioSegment` / `FFmpeg` |
| `requirements.txt` | Environment Dependency Graph | `pip` |

---

## 🛠️ Advanced Configuration

### TorchCrepe Optimization
In the script, you can adjust the `model` parameter:
* `tiny`: Lowest latency, suitable for most CPUs.
* `full`: Highest precision, requires CUDA-enabled GPU for real-time performance.

### Low-Latency Audio
To minimize "crackling" or "stuttering," we utilize a small buffer size in the `sounddevice` InputStream and `pygame.mixer`:
```python
pygame.mixer.pre_init(44100, -16, 2, 512) # 512 samples (~11ms buffer)

```

---

## 🕹️ Control Interface

The system splits functionality between your hands: the **Right Hand** is your "Keyboard," and the **Left Hand** is your "Pedalboard/Control Surface."

### 🎹 Right Hand (Chord Performance)

The chords played depend on which finger you pinch and where your hand is located vertically on the screen.

| Finger Pinch | **Lower Zone** (Y < 0.5) | **Upper Zone** (Y > 0.5) |
| --- | --- | --- |
| **Index** | **I** (Tonic) | **V** (Dominant) |
| **Middle** | **ii** (Supertonic) | **vi** (Submediant) |
| **Ring** | **iii** (Mediant) | **vii°** (Leading Tone) |
| **Pinky** | **IV** (Subdominant) | **I** (Octave Up) |

### ⚙️ Left Hand (System Commands)

Use your left hand to modify the sound or lock the current musical key.

* **Index Pinch:** Transpose **+1 Octave** (12 semitones).
* **Middle Pinch:** Transpose **-1 Octave** (-12 semitones).
* **Ring Pinch:** 🔒 **Lock Scale** – Freezes the current key so you can sing lyrics without the piano shifting scales.
* **Pinky Pinch:** 🔓 **Unlock Scale** – Resumes real-time vocal pitch tracking.

---

### Visual Zone Guide

The camera view is split horizontally in the middle:

* **Top Half:** Reach up here to play "High Energy" chords like the **V** or **vi**.
* **Bottom Half:** Keep your hand low for "Home" chords like the **I** or **IV**.

## 📜 Dependencies

* `torch` & `torchcrepe`: Pitch estimation CNN.
* `mediapipe`: Hand landmark detection.
* `musicpy`: Diatonic logic and scale algorithms.
* `mido`: MIDI protocol handling.
* `sounddevice`: Low-latency audio I/O.
