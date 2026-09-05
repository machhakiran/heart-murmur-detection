# 🫀 AI-Powered Heart Murmur Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Hub-yellow.svg)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Kaggle-Heartbeat%20Sound-20BEFF.svg)](https://www.kaggle.com/datasets/abdallahaboelkhair/heartbeat-sound)

An end-to-end medical deep learning system designed for phonocardiogram (PCG) acoustic analysis and preliminary cardiac auscultation. The system processes heartbeat sound recordings, extracts 60 time-frequency acoustic features, and classifies heart sounds into **Normal**, **Heart Murmur**, and **Extrasystole (Arrhythmia)** using a Bidirectional LSTM (BiLSTM) network with temporal attention pooling.

---

## 🌟 Key Highlights

- **Acoustic Signal Processing:** Extracts multi-domain features:
  - 40 Mel-Frequency Cepstral Coefficients (MFCCs)
  - 7 Spectral Contrast bands (identifying high-frequency systolic/diastolic turbulence)
  - 12 Chroma pitch bins
  - Root Mean Square (RMS) energy envelope
- **Deep Learning BiLSTM Model:** Captures forward and backward temporal dependencies across cardiac cycles (S1 "lub" and S2 "dub"), with attention pooling to highlight diagnostic murmur intervals.
- **Hugging Face Hub Integration:** Decoupled model hosting—the web application dynamically fetches model checkpoints from Hugging Face Hub.
- **Streamlit Community Cloud:** Interactive, real-time web application with interactive audio playback, waveform visualization, Mel-scale spectrogram, and clinical confidence breakdown.
- **Google Colab Training Workflow:** Complete notebook for GPU-accelerated training directly with the Kaggle dataset (`abdallahaboelkhair/heartbeat-sound`).

---

## 📁 Project Structure

```
├── app.py                             # Streamlit interactive web application
├── requirements.txt                   # Application Python dependencies
├── packages.txt                       # Linux audio codecs for Streamlit Cloud (ffmpeg, libsndfile1)
├── DEPLOYMENT_GUIDE.md                # Step-by-step account connection & deployment guide
├── colab/
│   └── train_lstm_heart_murmur.ipynb  # End-to-end Google Colab training & HF upload notebook
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Global audio & model parameters
│   ├── audio_processor.py             # Audio normalization & 60-band feature extraction
│   ├── model.py                       # PyTorch BiLSTM with Attention Pooling
│   ├── visualizer.py                  # Waveform, Spectrogram, & Probability visualizers
│   └── inference.py                   # Prediction pipeline with Hugging Face Hub caching
├── samples/
│   ├── generate_sample_audios.py      # Synthetic cardiac audio generator for immediate testing
│   ├── normal_sample.wav              # Benchmark normal S1/S2 heart sound
│   ├── murmur_sample.wav              # Benchmark systolic murmur recording
│   └── extrasystole_sample.wav        # Benchmark premature contraction recording
└── models/
    └── heart_murmur_lstm.pt           # Local baseline model checkpoint
```

---

## ⚡ Quickstart (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/<YOUR_USERNAME>/heart-murmur-detection.git
cd heart-murmur-detection
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate benchmark samples & baseline model
```bash
python samples/generate_sample_audios.py
```

### 5. Launch the Streamlit application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Cloud Deployment & Accounts

For full step-by-step instructions on connecting your personal accounts and deploying live:
👉 **[Read the Complete Deployment Guide (DEPLOYMENT_GUIDE.md)](DEPLOYMENT_GUIDE.md)**

Summary of the 5-step cloud deployment:
1. **Kaggle**: Download `kaggle.json` API token to access `abdallahaboelkhair/heartbeat-sound`.
2. **Google Colab**: Open `colab/train_lstm_heart_murmur.ipynb` with GPU, run training, and evaluate.
3. **Hugging Face Hub**: Generate a Write Token and upload the trained model (`heart_murmur_lstm.pt`).
4. **GitHub**: Push this repository to your GitHub account.
5. **Streamlit Community Cloud**: Connect the GitHub repo and deploy with 1 click!

---

## ⚠️ Medical Disclaimer

This project is developed solely for educational, academic research, and preliminary screening demonstration purposes. It is **not** a certified medical diagnostic device. Clinical decisions must always be made by a licensed healthcare provider based on formal medical auscultation and diagnostic tests (e.g., Echocardiogram, ECG).
