"""
Comprehensive Automated Pipeline Verification Tests.
"""
import os
import sys

# Ensure local writable directory for matplotlib cache
os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".matplotlib_cache"))

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from src.config import CONFIG
from src.audio_processor import AudioProcessor
from src.model import HeartMurmurLSTM, save_checkpoint, load_checkpoint
from src.inference import MurmurPredictor
from src.visualizer import AudioVisualizer


def test_audio_processor():
    print("Testing AudioProcessor...")
    processor = AudioProcessor(CONFIG.audio)
    
    sample_file = "samples/normal_sample.wav"
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"

    # Load audio
    y, sr = processor.load_audio(sample_file)
    assert sr == CONFIG.audio.sample_rate, f"Unexpected sample rate: {sr}"
    print(f"  ✓ Audio loaded: {len(y)} samples at {sr} Hz")

    # Fix duration
    y_norm = processor.normalize_and_fix_duration(y)
    expected_len = int(CONFIG.audio.sample_rate * CONFIG.audio.duration)
    assert len(y_norm) == expected_len, f"Length mismatch: {len(y_norm)} vs {expected_len}"
    print(f"  ✓ Duration standardized to exactly {CONFIG.audio.duration} seconds ({len(y_norm)} samples)")

    # Extract features
    features = processor.extract_features(y_norm)
    assert features.ndim == 2, f"Feature array must be 2D, got {features.ndim}"
    assert features.shape[1] == CONFIG.model.input_dim, f"Expected {CONFIG.model.input_dim} features, got {features.shape[1]}"
    print(f"  ✓ Extracted features sequence shape: {features.shape} (seq_len, feature_dim)")

    # Spectrogram
    mel_db, times, freqs = processor.compute_spectrogram(y_norm)
    assert mel_db.shape[0] == CONFIG.audio.n_mels
    print(f"  ✓ Mel Spectrogram shape: {mel_db.shape}")


def test_model_architecture():
    print("Testing HeartMurmurLSTM Model Architecture...")
    model = HeartMurmurLSTM(CONFIG.model)
    model.eval()

    # Dummy batch: (batch_size=2, seq_len=157, feature_dim=60)
    seq_len = 157
    dummy_input = torch.randn(2, seq_len, CONFIG.model.input_dim)
    
    with torch.no_grad():
        out = model(dummy_input)
    
    assert out.shape == (2, CONFIG.model.num_classes), f"Output shape mismatch: {out.shape}"
    print(f"  ✓ Model forward pass successful. Logits shape: {out.shape}")


def test_inference_pipeline():
    print("Testing End-to-End MurmurPredictor...")
    predictor = MurmurPredictor(local_model_path="models/heart_murmur_lstm.pt")
    
    for sample_name in ["normal_sample.wav", "murmur_sample.wav", "extrasystole_sample.wav"]:
        filepath = os.path.join("samples", sample_name)
        result = predictor.predict(filepath)
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "spectrogram" in result
        assert result["predicted_class"] in CONFIG.classes
        print(f"  ✓ Sample '{sample_name}': Predicted={result['predicted_class']} (Confidence={result['confidence']*100:.1f}%) [Source: {result['model_source']}]")


def test_visualizers():
    print("Testing AudioVisualizer...")
    processor = AudioProcessor(CONFIG.audio)
    y, sr = processor.load_audio("samples/normal_sample.wav")
    y_norm = processor.normalize_and_fix_duration(y)
    mel_db, times, freqs = processor.compute_spectrogram(y_norm)

    fig_wave = AudioVisualizer.plot_waveform(y_norm, sr)
    assert fig_wave is not None
    print("  ✓ Waveform figure generated.")

    fig_spec = AudioVisualizer.plot_spectrogram(mel_db, times, freqs)
    assert fig_spec is not None
    print("  ✓ Mel spectrogram figure generated.")

    fig_prob = AudioVisualizer.plot_probabilities({"Normal": 0.85, "Murmur": 0.10, "Extrasystole": 0.05}, "Normal")
    assert fig_prob is not None
    print("  ✓ Probability bar chart generated.")


if __name__ == "__main__":
    print("Starting automated test suite...\n")
    test_audio_processor()
    test_model_architecture()
    test_inference_pipeline()
    test_visualizers()
    print("\n All automated pipeline tests passed successfully!")
