"""
Inference Pipeline with Hugging Face Model Hub Integration.
"""
import os
import io
from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from src.config import CONFIG, SystemConfig
from src.audio_processor import AudioProcessor
from src.model import HeartMurmurLSTM, load_checkpoint, save_checkpoint


class MurmurPredictor:
    """Predictor for phonocardiogram recordings with Hugging Face Hub support."""

    def __init__(
        self,
        hf_repo_id: Optional[str] = None,
        local_model_path: Optional[str] = "models/heart_murmur_lstm.pt",
        device: Optional[str] = None
    ):
        self.config: SystemConfig = CONFIG
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_repo_id = hf_repo_id or os.getenv("HF_MODEL_REPO", self.config.default_hf_repo)
        self.local_model_path = local_model_path
        self.audio_processor = AudioProcessor(self.config.audio)
        self.model: Optional[HeartMurmurLSTM] = None
        self.model_source: str = "Uninitialized"

        # Load or initialize model
        self._load_model()

    def _load_model(self):
        """Attempt to load model from Hugging Face Hub, fallback to local checkpoint or initialization."""
        # 1. Try Hugging Face Model Hub if repo configured
        if self.hf_repo_id:
            try:
                from huggingface_hub import hf_hub_download
                print(f"Checking Hugging Face Hub repository: {self.hf_repo_id}...")
                downloaded_file = hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename=self.config.model_filename,
                    repo_type="model"
                )
                self.model = load_checkpoint(downloaded_file, map_location=self.device)
                self.model.to(self.device)
                self.model_source = f"Hugging Face Hub ({self.hf_repo_id})"
                print(f"Successfully loaded model from Hugging Face: {self.hf_repo_id}")
                return
            except Exception as e:
                print(f"Hugging Face Hub download skipped or failed ({e}). Checking local files...")

        # 2. Try local model path
        if self.local_model_path and os.path.exists(self.local_model_path):
            try:
                self.model = load_checkpoint(self.local_model_path, map_location=self.device)
                self.model.to(self.device)
                self.model_source = f"Local Checkpoint ({self.local_model_path})"
                print(f"Successfully loaded local model from: {self.local_model_path}")
                return
            except Exception as e:
                print(f"Failed to load local model: {e}")

        # 3. Fallback: Initialize fresh model (ready for inference or local checkpoint saving)
        print("Initializing HeartMurmurLSTM architecture...")
        self.model = HeartMurmurLSTM(self.config.model)
        self.model.to(self.device)
        self.model.eval()
        self.model_source = "Initialized Architecture (Baseline weights)"

    def predict(
        self, audio_source: Union[str, bytes, io.BytesIO]
    ) -> Dict[str, Any]:
        """
        Run full diagnostic pipeline on audio input.
        
        Returns:
            Dict containing predicted class, confidence, all probabilities,
            processed audio, mel-spectrogram, and clinical interpretation.
        """
        # 1. Load raw audio
        y_raw, sr = self.audio_processor.load_audio(audio_source)

        # 2. Normalize and fix length to target 5.0 seconds
        y_norm = self.audio_processor.normalize_and_fix_duration(y_raw)

        # 3. Extract acoustic features: (seq_len, feature_dim)
        features = self.audio_processor.extract_features(y_norm)

        # 4. Prepare tensor for PyTorch: shape (1, seq_len, feature_dim)
        x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 5. Model forward pass
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # 6. Extract classification results
        classes = self.config.classes
        prob_dict = {cls_name: float(probs[i]) for i, cls_name in enumerate(classes)}
        predicted_idx = int(np.argmax(probs))
        predicted_class = classes[predicted_idx]
        confidence = float(probs[predicted_idx])

        # 7. Compute spectrogram for display
        mel_db, times, freqs = self.audio_processor.compute_spectrogram(y_norm)

        # 8. Clinical interpretation details
        desc = self.config.class_descriptions.get(predicted_class, {
            "title": predicted_class,
            "severity": "Unknown",
            "badge_color": "gray",
            "details": "No clinical details available."
        })

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "interpretation": desc,
            "raw_audio": y_norm,
            "sample_rate": sr,
            "spectrogram": (mel_db, times, freqs),
            "model_source": self.model_source
        }
