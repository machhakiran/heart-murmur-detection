"""
PyTorch Deep Learning Model Architecture for Heart Murmur Detection.
"""
import json
import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import CONFIG, ModelConfig


class AttentionPooling(nn.Module):
    """Temporal attention pooling to weight diagnostic heart cycles (S1, S2, murmur phases)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, hidden_dim)
        weights = self.attention(x)  # (batch_size, seq_len, 1)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * x, dim=1)  # (batch_size, hidden_dim)
        return context


class HeartMurmurLSTM(nn.Module):
    """
    Bidirectional LSTM with Attention Pooling for phonocardiogram sequence classification.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or CONFIG.model

        input_dim = self.config.input_dim
        hidden_dim = self.config.hidden_dim
        num_layers = self.config.num_layers
        dropout = self.config.dropout if num_layers > 1 else 0.0
        bidirectional = self.config.bidirectional
        num_classes = self.config.num_classes

        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # BiLSTM Layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Attention pooling
        self.pool = AttentionPooling(lstm_out_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(lstm_out_dim),
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Linear projection
        proj = self.input_proj(x)
        
        # LSTM sequence modeling
        lstm_out, _ = self.lstm(proj)
        
        # Temporal attention pooling
        context = self.pool(lstm_out)
        
        # Classification logits
        logits = self.classifier(context)
        return logits


def save_checkpoint(model: HeartMurmurLSTM, filepath: str, extra_meta: Optional[Dict[str, Any]] = None):
    """Save model weights and metadata to file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    meta = {
        "config": {
            "input_dim": model.config.input_dim,
            "hidden_dim": model.config.hidden_dim,
            "num_layers": model.config.num_layers,
            "dropout": model.config.dropout,
            "bidirectional": model.config.bidirectional,
            "num_classes": model.config.num_classes,
        },
        "extra_meta": extra_meta or {}
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": meta
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, map_location: str = "cpu") -> HeartMurmurLSTM:
    """Load model from saved checkpoint."""
    checkpoint = torch.load(filepath, map_location=map_location)
    meta_config = checkpoint.get("metadata", {}).get("config", {})
    
    if meta_config:
        model_config = ModelConfig(**meta_config)
    else:
        model_config = CONFIG.model

    model = HeartMurmurLSTM(config=model_config)
    
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model
