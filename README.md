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

## 📂 Project Structure

To run the application, ensure your local directory is organized as follows. Note that large binary files are excluded from the repository and must be added manually.

```text
VocalCV-Accompanist/
├── .gitignore               # Prevents uploading large binaries
├── LICENSE                  # MIT License
├── README.md                # Technical & Setup guide
├── requirements.txt         # Dependency list
│
├── VocalCV_MIDI.py          # Pro MIDI-out version
├── VocalCV_SF2.py           # Internal SoundFont version
│
├── ffmpeg.exe               # [User-Provided] Required for SF2 audio
└── Full Grand Piano.sf2     # [User-Provided] Place your soundbank here

```

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
* **Inference:** Uses **TorchCrepe** (CNN) for state-of-the-art pitch estimation.
* **Mapping:** Converts estimated Frequency () to MIDI numbers:




2. **Computer Vision Thread (MediaPipe):**
* **Heuristics:** Euclidean distance calculation between `THUMB_TIP` and `FINGER_TIPS` for pinch detection.
* **Normalization:** Maps Y-coordinates to musical "Zones" (Lower/Upper).


3. **Main/UI Thread (OpenCV & Musicpy):**
* **Theory Engine:** Real-time diatonic chord lookup via `musicpy.scale`.
* **Rendering:** Live HUD with Scale status and RMS-based Volume Meter.



---

## 🕹️ Control Interface

### 🎹 Right Hand (Performance)

| Finger Pinch | **Lower Zone** (Y < 0.5) | **Upper Zone** (Y > 0.5) |
| --- | --- | --- |
| **Index** | **I** (Tonic) | **V** (Dominant) |
| **Middle** | **ii** (Supertonic) | **vi** (Submediant) |
| **Ring** | **iii** (Mediant) | **vii°** (Leading Tone) |
| **Pinky** | **IV** (Subdominant) | **I** (Octave Up) |

### ⚙️ Left Hand (System Commands)

* **Index Pinch:** Transpose **+1 Octave**.
* **Middle Pinch:** Transpose **-1 Octave**.
* **Ring Pinch:** 🔒 **Lock Scale** – Freezes current key (for lyrical sections).
* **Pinky Pinch:** 🔓 **Unlock Scale** – Resumes pitch tracking.

---

## 🛠️ Advanced Configuration

### TorchCrepe Optimization

Adjust the `model` parameter in the script for performance:

* `tiny`: Lowest latency (Standard).
* `full`: Highest precision (Requires GPU/CUDA).

### Low-Latency Audio

Buffer optimization for the internal synth:

```python
pygame.mixer.pre_init(44100, -16, 2, 512) # 512 samples (~11ms buffer)

```

---

## 📜 Dependencies

* `torch` & `torchcrepe`: Pitch estimation CNN.
* `mediapipe`: Hand landmark detection.
* `musicpy`: Diatonic logic and scale algorithms.
* `mido`: MIDI protocol handling.
* `sounddevice`: Low-latency audio I/O.
