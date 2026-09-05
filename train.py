"""
Autonomous End-to-End Training Script.
Trains BiLSTM on the downloaded Kaggle Heartbeat Sound dataset and publishes to Hugging Face.
"""
import os
import sys
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.config import CONFIG
from src.audio_processor import AudioProcessor
from src.model import HeartMurmurLSTM, save_checkpoint


def collect_dataset(dataset_dir="data/Heartbeat_Sound"):
    """Discover audio files and map to diagnostic classes."""
    data = []
    
    # 1. Normal
    normal_files = glob.glob(os.path.join(dataset_dir, "normal", "*.wav"))
    for f in normal_files:
        data.append((f, "Normal"))
        
    # 2. Murmur
    murmur_files = glob.glob(os.path.join(dataset_dir, "murmur", "*.wav"))
    for f in murmur_files:
        data.append((f, "Murmur"))
        
    # 3. Extrasystole (extrastole + extrahls)
    extra_files = glob.glob(os.path.join(dataset_dir, "extrastole", "*.wav")) + \
                  glob.glob(os.path.join(dataset_dir, "extrahls", "*.wav"))
    for f in extra_files:
        data.append((f, "Extrasystole"))

    print(f"Collected {len(data)} total labeled audio samples:")
    print(f"  - Normal: {len(normal_files)}")
    print(f"  - Murmur: {len(murmur_files)}")
    print(f"  - Extrasystole: {len(extra_files)}")
    return data


class HeartSoundDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    print("=" * 60)
    print("AI-Powered Heart Murmur Detection: Autonomous Training Pipeline")
    print("=" * 60)

    # 1. Collect dataset
    data = collect_dataset()
    if len(data) == 0:
        raise RuntimeError("No audio samples found in data/Heartbeat_Sound.")

    # 2. Audio Feature Extraction
    processor = AudioProcessor(CONFIG.audio)
    class_map = {c: i for i, c in enumerate(CONFIG.classes)}

    print("\nExtracting 60 acoustic features (MFCC, Spectral, Chroma, RMS)...")
    X_list, y_list = [], []
    for filepath, label in tqdm(data):
        try:
            y_audio, _ = processor.load_audio(filepath)
            y_norm = processor.normalize_and_fix_duration(y_audio)
            feats = processor.extract_features(y_norm)
            X_list.append(feats)
            y_list.append(class_map[label])
        except Exception as e:
            continue

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"\nFeature matrix ready: {X.shape} (samples, seq_len, feature_dim)")

    # 3. Stratified Split (80% Train, 20% Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")

    train_loader = DataLoader(HeartSoundDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(HeartSoundDataset(X_val, y_val), batch_size=16, shuffle=False)

    # 4. Model Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on device: {device}")

    # Balance classes with class weights
    class_counts = np.bincount(y_train, minlength=len(CONFIG.classes))
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / np.sum(weights)
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    model = HeartMurmurLSTM(CONFIG.model).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    # 5. Training Loop
    epochs = 25
    best_val_loss = float("inf")
    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/heart_murmur_lstm.pt"

    print("\nStarting model training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == by).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_checkpoint(
                model,
                best_model_path,
                extra_meta={"val_loss": val_loss, "val_acc": val_acc, "epochs": epoch + 1}
            )

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1:02d}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

    print(f"\nBest Validation Accuracy: {best_val_acc*100:.1f}% (Loss: {best_val_loss:.4f})")
    print(f"Saved best model weights to: {best_model_path}")

    # 6. Detailed Evaluation
    print("\nEvaluating best checkpoint...")
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["state_dict"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())

    print("\n" + "=" * 50)
    print("Clinical Classification Report (Validation Set):")
    print("=" * 50)
    print(classification_report(all_targets, all_preds, target_names=CONFIG.classes, zero_division=0))

    # 7. Push Trained Model to Hugging Face Model Hub
    print("\nPublishing trained model to Hugging Face Model Hub...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = CONFIG.default_hf_repo
        api.upload_file(
            path_or_fileobj=best_model_path,
            path_in_repo="heart_murmur_lstm.pt",
            repo_id=repo_id
        )
        print(f"Successfully published trained model to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Hugging Face upload error: {e}")

    print("\n Autonomous training and deployment complete!")


if __name__ == "__main__":
    main()
