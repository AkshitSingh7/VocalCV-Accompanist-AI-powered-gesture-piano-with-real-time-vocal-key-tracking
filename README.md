# VocalCV-Accompanist 🎹🎤

> **Real-time AI Accompaniment: Fusing Deep Learning Pitch Estimation with Computer Vision Gesture Mapping.**

VocalCV-Accompanist is a sophisticated Python-based instrument that transforms vocal input into a harmonic foundation. It utilizes a **Convolutional Neural Network (CNN)** for pitch tracking and **Mediapipe's Perception Pipeline** for 3D hand landmarking, all tied together by a custom **Diatonic Logic Engine**.

---

## 🚀 Quick Start Guide

### 1. Installation

Open your terminal in the project folder and run:

```bash
pip install -r requirements.txt

```

### 2. Choosing Your Engine

You have two ways to play. Choose the file that fits your setup:

| File | Best For | Requirement |
| --- | --- | --- |
| **`VocalCV_MIDI.py`** | **Professional Use.** Zero latency, high-quality sounds (Piano 10, DAWs). | Virtual MIDI Cable (loopMIDI) |
| **`VocalCV_SF2.py`** | **Casual Use.** Quick setup, plays sound directly from Python. | `ffmpeg.exe` & `.sf2` file |

---

## 🎹 How to use MIDI Mode (High Quality)

The MIDI version doesn't make sound by itself; it sends "instructions" to other music software. To use it:

1. **Install a Virtual Cable:** Download and install [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html). Create a new port named `AirPianoPort 1`.
2. **Open your Sound Software:** Open **Piano 10**, **Ableton**, **FL Studio**, or **Contact**.
3. **Set Input:** In your software's settings, set the **MIDI Input** to `AirPianoPort 1`.
4. **Run the Script:** Run `python VocalCV_MIDI.py`. Your hand gestures will now trigger the high-quality sounds in your DAW/App.

---

## 🏗️ Technical Architecture

The system operates across three parallel processing threads to ensure low-latency performance:

1. **Vocal Processing Thread (DSP):**
* **Inference:** Uses **TorchCrepe** for state-of-the-art pitch estimation.
* **Mapping:** Converts estimated Frequency () to MIDI numbers using the formula:



2. **Computer Vision Thread (MediaPipe):**
* **Heuristics:** Calculates Euclidean distance between `THUMB_TIP` and `FINGER_TIPS` to detect binary "pinch" states.
* **Normalization:** Maps the Thumb's Y-coordinate () to the musical "Zone" (Lower/Upper).


3. **Main/UI Thread (OpenCV & Musicpy):**
* **Theory Engine:** On pinch, it initializes a `musicpy.scale` object and performs a `pick_chord_by_degree` lookup.
* **Rendering:** Displays the active Scale, Roman Numerals, and a Visual Volume Meter.



---

## 🕹️ Control Interface

### 🎹 Right Hand (Performance)

The chords played depend on your finger pinch and your vertical hand position.

| Finger Pinch | **Lower Zone** (Y < 0.5) | **Upper Zone** (Y > 0.5) |
| --- | --- | --- |
| **Index** | **I** (Tonic) | **V** (Dominant) |
| **Middle** | **ii** (Supertonic) | **vi** (Submediant) |
| **Ring** | **iii** (Mediant) | **vii°** (Leading Tone) |
| **Pinky** | **IV** (Subdominant) | **I** (Octave Up) |

### ⚙️ Left Hand (System Commands)

* **Index Pinch:** Transpose **+1 Octave**.
* **Middle Pinch:** Transpose **-1 Octave**.
* **Ring Pinch:** 🔒 **Lock Scale** – Ignores your voice (perfect for singing lyrics).
* **Pinky Pinch:** 🔓 **Unlock Scale** – Resumes vocal pitch tracking.

---

## 🎼 Music Theory Implementation

The project uses **Diatonic Set Theory**. When a key is detected (e.g., G), the engine builds a Major Scale:
`Scale('G', 'major') -> [G, A, B, C, D, E, F#]`

The hand position then selects the chord degree:

* **Index Pinch (Lower Zone):** Degree 0 → **I Major** (G - B - D)
* **Index Pinch (Upper Zone):** Degree 4 → **V Major** (D - F# - A)

---

## 🛠️ Advanced Configuration

### TorchCrepe Optimization

In the script, you can adjust the `model` parameter:

* `tiny`: Lowest latency, suitable for most CPUs.
* `full`: Highest precision, requires CUDA-enabled GPU.

### Low-Latency Audio

To minimize latency in SF2 mode, we utilize a small buffer size:

```python
pygame.mixer.pre_init(44100, -16, 2, 512) # 512 samples (~11ms buffer)

```

## 📜 Dependencies

* `torch` & `torchcrepe`: Pitch estimation CNN.
* `mediapipe`: Hand landmark detection.
* `musicpy`: Diatonic logic and scale algorithms.
* `mido`: MIDI protocol handling.
* `sounddevice`: Low-latency audio I/O.
