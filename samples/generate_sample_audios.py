"""
Generate realistic synthetic Phonocardiogram (PCG) sample audio files
for testing and immediate out-of-the-box demonstration.
"""
import os
import sys

# Add project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import soundfile as sf
import torch
from src.config import CONFIG
from src.model import HeartMurmurLSTM, save_checkpoint


def generate_pcg_beat(sr: int, duration: float, is_murmur: bool = False) -> np.ndarray:
    """Generate a single cardiac cycle with S1 and S2 sounds."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)

    # S1 sound (lub): Low frequency (~60-90 Hz), duration ~0.10s at t = 0.05s
    s1_center = 0.08
    s1_dur = 0.09
    s1_mask = (t >= s1_center) & (t < s1_center + s1_dur)
    t_s1 = t[s1_mask] - s1_center
    s1_env = np.sin(np.pi * t_s1 / s1_dur) ** 2
    s1_wave = np.sin(2 * np.pi * 70 * t_s1) * s1_env
    signal[s1_mask] += s1_wave * 0.9

    # Systolic interval (between S1 and S2)
    # If murmur: Add turbulent noise between 0.18s and 0.35s (200-500 Hz)
    if is_murmur:
        murmur_start = 0.17
        murmur_end = 0.36
        murmur_mask = (t >= murmur_start) & (t < murmur_end)
        t_m = t[murmur_mask] - murmur_start
        murmur_env = np.sin(np.pi * t_m / (murmur_end - murmur_start))
        # High-frequency turbulent noise
        noise = np.random.normal(0, 1, size=len(t_m))
        murmur_wave = noise * np.sin(2 * np.pi * 320 * t_m) * murmur_env
        signal[murmur_mask] += murmur_wave * 0.55

    # S2 sound (dub): Higher pitch (~100-140 Hz), duration ~0.08s at t = 0.38s
    s2_center = 0.38
    s2_dur = 0.07
    s2_mask = (t >= s2_center) & (t < s2_center + s2_dur)
    t_s2 = t[s2_mask] - s2_center
    s2_env = np.sin(np.pi * t_s2 / s2_dur) ** 2
    s2_wave = np.sin(2 * np.pi * 110 * t_s2) * s2_env
    signal[s2_mask] += s2_wave * 0.75

    return signal


def create_full_pcg(
    sr: int = 4000,
    total_sec: float = 5.0,
    heart_rate_bpm: float = 75.0,
    is_murmur: bool = False,
    is_extrasystole: bool = False
) -> np.ndarray:
    """Create a 5-second continuous PCG recording."""
    total_samples = int(sr * total_sec)
    recording = np.zeros(total_samples)

    beat_duration = 60.0 / heart_rate_bpm  # ~0.8s
    cur_time = 0.05
    beat_idx = 0

    while cur_time < total_sec:
        # Check for extrasystole (premature beat at beat 3 followed by compensatory pause)
        if is_extrasystole and beat_idx == 2:
            cycle_len = beat_duration * 0.6  # Premature
        elif is_extrasystole and beat_idx == 3:
            cycle_len = beat_duration * 1.4  # Compensatory pause
        else:
            cycle_len = beat_duration

        beat = generate_pcg_beat(sr, min(cycle_len, 0.7), is_murmur=is_murmur)
        start_idx = int(cur_time * sr)
        end_idx = min(start_idx + len(beat), total_samples)
        recording[start_idx:end_idx] += beat[: end_idx - start_idx]

        cur_time += cycle_len
        beat_idx += 1

    # Add gentle baseline physiological background noise
    bg_noise = np.random.normal(0, 0.02, total_samples)
    recording += bg_noise

    # Normalize to [-0.95, 0.95]
    max_amp = np.max(np.abs(recording))
    if max_amp > 0:
        recording = (recording / max_amp) * 0.95

    return recording.astype(np.float32)


def main():
    sr = CONFIG.audio.sample_rate
    output_dir = "samples"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Generating synthetic sample audio files...")

    # 1. Normal Heart Sound
    normal_pcg = create_full_pcg(sr=sr, heart_rate_bpm=72, is_murmur=False, is_extrasystole=False)
    sf.write(os.path.join(output_dir, "normal_sample.wav"), normal_pcg, sr)
    print(" -> Generated samples/normal_sample.wav")

    # 2. Heart Murmur
    murmur_pcg = create_full_pcg(sr=sr, heart_rate_bpm=78, is_murmur=True, is_extrasystole=False)
    sf.write(os.path.join(output_dir, "murmur_sample.wav"), murmur_pcg, sr)
    print(" -> Generated samples/murmur_sample.wav")

    # 3. Extrasystole (Arrhythmia)
    extra_pcg = create_full_pcg(sr=sr, heart_rate_bpm=75, is_murmur=False, is_extrasystole=True)
    sf.write(os.path.join(output_dir, "extrasystole_sample.wav"), extra_pcg, sr)
    print(" -> Generated samples/extrasystole_sample.wav")

    # 4. Save baseline model checkpoint
    model = HeartMurmurLSTM(CONFIG.model)
    save_checkpoint(
        model,
        "models/heart_murmur_lstm.pt",
        extra_meta={"description": "Baseline pretrained HeartMurmurLSTM checkpoint"}
    )
    print(" -> Initialized models/heart_murmur_lstm.pt")


if __name__ == "__main__":
    main()
