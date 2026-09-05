"""
Audio Signal Processing and Feature Extraction Module for Phonocardiogram (PCG) data.
"""
import io
import os
from typing import Tuple, Union, Optional
import numpy as np
import librosa
import soundfile as sf
from src.config import CONFIG, AudioConfig


class AudioProcessor:
    """Handles audio loading, filtering, normalization, and acoustic feature extraction."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or CONFIG.audio
        self.target_length = int(self.config.sample_rate * self.config.duration)

    def load_audio(
        self, audio_source: Union[str, bytes, io.BytesIO]
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio from a file path, raw bytes, or BytesIO buffer.
        Resamples audio to target sample rate (default 4000 Hz).
        
        Returns:
            y (np.ndarray): 1D audio time series.
            sr (int): Target sampling rate.
        """
        if isinstance(audio_source, (bytes, bytearray)):
            audio_source = io.BytesIO(audio_source)

        # librosa.load can read from file path or file-like object
        y, orig_sr = librosa.load(audio_source, sr=self.config.sample_rate, mono=True)
        return y, self.config.sample_rate

    def normalize_and_fix_duration(self, y: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude to range [-1, 1] and pad or truncate to fixed duration.
        """
        # Peak normalization
        max_val = np.max(np.abs(y))
        if max_val > 1e-6:
            y = y / max_val
        else:
            y = np.zeros_like(y)

        # Pad or truncate to fixed duration
        if len(y) < self.target_length:
            # Repeat or pad with zeros
            pad_width = self.target_length - len(y)
            y = np.pad(y, (0, pad_width), mode="constant")
        else:
            y = y[: self.target_length]

        return y

    def extract_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract multi-feature acoustic sequence for LSTM input.
        
        Features:
            1. MFCC (40 coefficients)
            2. Spectral Contrast (7 bands)
            3. Chroma STFT (12 pitch classes)
            4. RMS Energy (1 feature)
        Total feature dim = 40 + 7 + 12 + 1 = 60.

        Returns:
            features (np.ndarray): Shape (seq_len, feature_dim).
        """
        sr = self.config.sample_rate
        hop_length = self.config.hop_length
        n_fft = self.config.n_fft

        # 1. MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.config.n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )  # (40, T)

        # 2. Spectral Contrast
        # For lower sampling rates like 4000 Hz, fmin must be adjusted
        try:
            spectral_contrast = librosa.feature.spectral_contrast(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin=50.0,
                n_bands=6
            )  # (7, T)
        except Exception:
            # Fallback if audio spectrum is too narrow
            t_len = mfcc.shape[1]
            spectral_contrast = np.zeros((7, t_len), dtype=np.float32)

        # 3. Chroma STFT
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )  # (12, T)

        # 4. RMS Energy
        rms = librosa.feature.rms(
            y=y,
            hop_length=hop_length
        )  # (1, T)

        # Ensure all features have identical time dimension
        min_len = min(mfcc.shape[1], spectral_contrast.shape[1], chroma.shape[1], rms.shape[1])
        mfcc = mfcc[:, :min_len]
        spectral_contrast = spectral_contrast[:, :min_len]
        chroma = chroma[:, :min_len]
        rms = rms[:, :min_len]

        # Stack features along feature dimension: (60, min_len)
        stacked = np.vstack([mfcc, spectral_contrast, chroma, rms])

        # Standardize (Z-score normalization per feature band)
        mean = np.mean(stacked, axis=1, keepdims=True)
        std = np.std(stacked, axis=1, keepdims=True) + 1e-6
        normalized_features = (stacked - mean) / std

        # Transpose to (seq_len, feature_dim) for LSTM
        return normalized_features.T.astype(np.float32)

    def compute_spectrogram(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Mel-scale spectrogram in decibels for visualization.
        
        Returns:
            mel_db (np.ndarray): Decibel scaled Mel spectrogram.
            times (np.ndarray): Time axis array in seconds.
            freqs (np.ndarray): Mel frequency bin center values in Hz.
        """
        sr = self.config.sample_rate
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmax=sr / 2
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        times = librosa.times_like(mel_db, sr=sr, hop_length=self.config.hop_length)
        freqs = librosa.mel_frequencies(n_mels=self.config.n_mels, fmax=sr / 2)
        return mel_db, times, freqs
