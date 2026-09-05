"""
Audio Signal and Acoustic Analysis Visualizer for Streamlit.
"""
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from src.config import CONFIG


class AudioVisualizer:
    """Generates clean, publication-quality figures for audio signal analysis."""

    @staticmethod
    def plot_waveform(y: np.ndarray, sr: int, title: str = "Phonocardiogram (PCG) Waveform") -> plt.Figure:
        """Plot the time-domain waveform with signal envelope."""
        time_axis = np.linspace(0, len(y) / sr, num=len(y))

        fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#161b22")

        # Plot raw waveform
        ax.plot(time_axis, y, color="#58a6ff", alpha=0.85, linewidth=0.8, label="PCG Signal")
        
        # Add zero reference line
        ax.axhline(0, color="#30363d", linestyle="--", linewidth=0.8)

        ax.set_title(title, color="#f0f6fc", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Time (seconds)", color="#8b949e", fontsize=10)
        ax.set_ylabel("Amplitude", color="#8b949e", fontsize=10)
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.grid(True, color="#21262d", linestyle=":", alpha=0.7)
        ax.set_xlim([0, time_axis[-1]])
        ax.set_ylim([-1.1, 1.1])

        # Style spines
        for spine in ax.spines.values():
            spine.set_color("#30363d")

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_spectrogram(
        mel_db: np.ndarray, times: np.ndarray, freqs: np.ndarray, title: str = "Mel-Scale Spectrogram"
    ) -> plt.Figure:
        """Plot Mel Spectrogram displaying frequency and energy distribution."""
        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#161b22")

        img = ax.pcolormesh(
            times,
            freqs,
            mel_db,
            shading="gouraud",
            cmap="inferno"
        )

        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cbar.set_label("Intensity (dB)", color="#8b949e", fontsize=9)
        cbar.ax.tick_params(colors="#8b949e", labelsize=8)
        cbar.outline.set_color("#30363d")

        ax.set_title(title, color="#f0f6fc", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Time (seconds)", color="#8b949e", fontsize=10)
        ax.set_ylabel("Frequency (Hz)", color="#8b949e", fontsize=10)
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.set_ylim([0, freqs[-1]])

        for spine in ax.spines.values():
            spine.set_color("#30363d")

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_probabilities(
        probs: Dict[str, float], predicted_class: str
    ) -> plt.Figure:
        """Plot class probabilities horizontal bar chart with color coding."""
        classes = list(probs.keys())
        values = [probs[c] * 100 for c in classes]

        fig, ax = plt.subplots(figsize=(8, 2.6), dpi=100)
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#161b22")

        # Dynamic color coding: Red for Murmur, Orange for Extrasystole, Green for Normal
        color_map = {
            "Normal": "#238636",         # Green
            "Murmur": "#da3633",         # Red
            "Extrasystole": "#d29922"     # Amber
        }
        colors = [color_map.get(c, "#58a6ff") for c in classes]

        bars = ax.barh(classes, values, color=colors, height=0.55, edgecolor="#30363d")

        # Label probabilities at bar ends
        for bar, val in zip(bars, values):
            ax.text(
                val + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                ha="left",
                color="#f0f6fc",
                fontweight="bold",
                fontsize=9
            )

        ax.set_xlim([0, 115])
        ax.set_title("Classification Confidence Distribution", color="#f0f6fc", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Confidence (%)", color="#8b949e", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.grid(axis="x", color="#21262d", linestyle=":", alpha=0.6)

        for spine in ax.spines.values():
            spine.set_color("#30363d")

        fig.tight_layout()
        return fig
