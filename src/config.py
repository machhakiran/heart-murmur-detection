"""
Global Configuration for Heart Murmur Detection System.
"""
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class AudioConfig:
    sample_rate: int = 4000          # 4000 Hz captures up to 2000 Hz (Nyquist), ideal for PCG (20-1000 Hz)
    duration: float = 5.0            # Standard clip duration in seconds (5 seconds)
    n_fft: int = 512                 # FFT window size
    hop_length: int = 128            # Frame shift
    n_mels: int = 40                 # Number of Mel bands
    n_mfcc: int = 40                 # Number of MFCC coefficients

@dataclass
class ModelConfig:
    input_dim: int = 60              # Combined acoustic feature dimensions (40 MFCC + 7 Spectral + 12 Chroma + 1 RMS)
    hidden_dim: int = 128            # BiLSTM hidden dimension
    num_layers: int = 2              # Number of stacked BiLSTM layers
    dropout: float = 0.3             # Dropout probability
    bidirectional: bool = True       # Bidirectional LSTM
    num_classes: int = 3             # Normal, Murmur, Extrasystole

@dataclass
class SystemConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Class labels matching the Kaggle Heartbeat Sound dataset
    classes: List[str] = field(default_factory=lambda: [
        "Normal",
        "Murmur",
        "Extrasystole"
    ])
    
    # Human-readable risk interpretations
    class_descriptions: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "Normal": {
            "title": "Normal Heart Rhythm",
            "severity": "Low Risk",
            "badge_color": "green",
            "details": "Clear S1 (lub) and S2 (dub) sounds detected with regular intervals. No significant systolic or diastolic turbulence observed."
        },
        "Murmur": {
            "title": "Heart Murmur Detected",
            "severity": "Abnormal / Requires Review",
            "badge_color": "red",
            "details": "Turbulent blood flow vibration detected between heart sounds. May indicate valve stenosis, regurgitation, or septal defect. Clinical auscultation and echocardiogram recommended."
        },
        "Extrasystole": {
            "title": "Extrasystole (Arrhythmia)",
            "severity": "Moderate Risk",
            "badge_color": "orange",
            "details": "Premature or extra cardiac contraction detected interrupting regular sinus rhythm. Often benign, but clinical evaluation recommended if frequent."
        }
    })
    
    # Hugging Face Model Hub defaults
    default_hf_repo: str = "machhakiran/heart-murmur-bilstm"
    model_filename: str = "heart_murmur_lstm.pt"
    config_filename: str = "config.json"

CONFIG = SystemConfig()
