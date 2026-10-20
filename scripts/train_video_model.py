"""
Training, Optimization, and Evaluation Pipeline for Video Deepfake Detection.
Trains CustomVideoDeepfakeModel combining our custom CNN spatial extractor with BiLSTM
and Temporal Self-Attention on extracted sequential video frames.
Computes frame-level and video-level metrics, ROC-AUC, attention distributions, and saves model weights.
"""

import sys
import os
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

from src.models.custom_cnn import CustomDeepfakeCNN
from src.models.custom_video_model import CustomVideoDeepfakeModel

# Strict reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

METADATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "metadata")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")
CONF_DIR = os.path.join(OUTPUTS_DIR, "confusion_matrices")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)


class VideoSequenceDataset(Dataset):
    """
    Loads sequences of extracted video frames grouped by video_id.
    Each item is a tensor of shape (Num_Frames, 3, 128, 128).
    """
    def __init__(self, metadata_df, frame_df, is_train=False):
        self.metadata = metadata_df.reset_index(drop=True)
        self.frame_df = frame_df
        self.is_train = is_train

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        vid_id = row["video_id"]
        label = float(row["label"])

        # Fetch sorted frames for this video
        vid_frames = self.frame_df[self.frame_df["video_id"] == vid_id].sort_values("extracted_seq_idx")
        frame_tensors = []

        # Data augmentation flag
        do_flip = self.is_train and (random.random() > 0.5)

        for _, frow in vid_frames.iterrows():
            img_path = frow["frame_path"]
            img = Image.open(img_path).convert("RGB").resize((128, 128))
            arr = np.array(img, dtype=np.uint8)
            if do_flip:
                arr = np.fliplr(arr).copy()

            tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            tensor = (tensor - self.mean) / self.std
            frame_tensors.append(tensor)

        # Stack into (T, 3, 128, 128)
        seq_tensor = torch.stack(frame_tensors, dim=0)
        return seq_tensor, torch.tensor(label, dtype=torch.float32), vid_id


def train_video_epoch(model, loader, crit_video, crit_frame, optimizer, scaler, device):
    model.train()
    # Keep spatial backbone in eval/fixed mode to preserve robust spatial features and fit in 4GB GPU
    model.spatial_extractor.eval()

    running_loss = 0.0
    correct_video = 0
    total_videos = 0

    for videos, labels, _ in loader:
        videos = videos.to(device)  # (B, T, 3, 128, 128)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)
        b, t, c, h, w = videos.shape

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            vid_logits, frame_logits, _ = model(videos)
            loss_video = crit_video(vid_logits, labels)

            # Frame loss: expand video label to all frames
            frame_labels = labels.unsqueeze(1).expand(-1, t, -1)
            loss_frame = crit_frame(frame_logits, frame_labels)

            # Combined spatio-temporal loss
            total_loss = loss_video + 0.5 * loss_frame

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item() * b
        preds_video = (torch.sigmoid(vid_logits) >= 0.5).float()
        correct_video += (preds_video == labels).sum().item()
        total_videos += b

    return running_loss / total_videos, correct_video / total_videos


def eval_video_epoch(model, loader, crit_video, crit_frame, device):
    model.eval()
    running_loss = 0.0
    correct_video = 0
    total_videos = 0

    with torch.no_grad():
        for videos, labels, _ in loader:
            videos = videos.to(device)
            labels = labels.to(device).unsqueeze(1)
            b, t, c, h, w = videos.shape

            with torch.amp.autocast("cuda"):
                vid_logits, frame_logits, _ = model(videos)
                loss_video = crit_video(vid_logits, labels)
                frame_labels = labels.unsqueeze(1).expand(-1, t, -1)
                loss_frame = crit_frame(frame_logits, frame_labels)
                total_loss = loss_video + 0.5 * loss_frame

            running_loss += total_loss.item() * b
            preds_video = (torch.sigmoid(vid_logits) >= 0.5).float()
            correct_video += (preds_video == labels).sum().item()
            total_videos += b

    return running_loss / total_videos, correct_video / total_videos


def full_video_evaluation(model, loader, device):
    model.eval()
    vid_labels, vid_preds, vid_probs = [], [], []
    all_frame_labels, all_frame_probs, all_frame_preds = [], [], []
    attention_maps = {}

    with torch.no_grad():
        for videos, labels, vid_ids in loader:
            videos = videos.to(device)
            b, t, c, h, w = videos.shape

            with torch.amp.autocast("cuda"):
                vid_logits, frame_logits, attn_weights = model(videos)
                v_probs = torch.sigmoid(vid_logits).squeeze(1).cpu().numpy()
                f_probs = torch.sigmoid(frame_logits).squeeze(-1).cpu().numpy()
                a_weights = attn_weights.squeeze(-1).cpu().numpy()

            v_preds = (v_probs >= 0.5).astype(int)
            vid_labels.extend(labels.numpy().astype(int).tolist())
            vid_preds.extend(v_preds.tolist())
            vid_probs.extend(v_probs.tolist())

            for i in range(b):
                vid_id = vid_ids[i]
                attention_maps[vid_id] = {
                    "label": int(labels[i].item()),
                    "pred": int(v_preds[i]),
                    "prob": float(v_probs[i]),
                    "frame_probs": f_probs[i].tolist(),
                    "attn_weights": a_weights[i].tolist()
                }
                # Collect frame level
                f_lbls = [int(labels[i].item())] * t
                f_pr = (f_probs[i] >= 0.5).astype(int)
                all_frame_labels.extend(f_lbls)
                all_frame_probs.extend(f_probs[i].tolist())
                all_frame_preds.extend(f_pr.tolist())

    vid_labels = np.array(vid_labels)
    vid_preds = np.array(vid_preds)
    vid_probs = np.array(vid_probs)

    all_frame_labels = np.array(all_frame_labels)
    all_frame_preds = np.array(all_frame_preds)
    all_frame_probs = np.array(all_frame_probs)

    # Video-Level Metrics
    v_acc = accuracy_score(vid_labels, vid_preds)
    v_prec = precision_score(vid_labels, vid_preds, zero_division=0)
    v_rec = recall_score(vid_labels, vid_preds, zero_division=0)
    v_f1 = f1_score(vid_labels, vid_preds, zero_division=0)
    v_auc = roc_auc_score(vid_labels, vid_probs) if len(np.unique(vid_labels)) > 1 else 1.0

    # Frame-Level Metrics
    f_acc = accuracy_score(all_frame_labels, all_frame_preds)
    f_prec = precision_score(all_frame_labels, all_frame_preds, zero_division=0)
    f_rec = recall_score(all_frame_labels, all_frame_preds, zero_division=0)
    f_f1 = f1_score(all_frame_labels, all_frame_preds, zero_division=0)
    f_auc = roc_auc_score(all_frame_labels, all_frame_probs)

    cm_vid = confusion_matrix(vid_labels, vid_preds)
    cm_frame = confusion_matrix(all_frame_labels, all_frame_preds)

    results = {
        "video_level": {
            "accuracy": float(v_acc),
            "precision": float(v_prec),
            "recall": float(v_rec),
            "f1_score": float(v_f1),
            "roc_auc": float(v_auc),
            "confusion_matrix": cm_vid.tolist()
        },
        "frame_level": {
            "accuracy": float(f_acc),
            "precision": float(f_prec),
            "recall": float(f_rec),
            "f1_score": float(f_f1),
            "roc_auc": float(f_auc),
            "confusion_matrix": cm_frame.tolist()
        },
        "attention_analysis": attention_maps
    }

    return results, (vid_labels, vid_preds, vid_probs), (all_frame_labels, all_frame_preds, all_frame_probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Video Model on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load video metadata and frame dataset
    meta_path = os.path.join(METADATA_DIR, "video_metadata.csv")
    frame_path = os.path.join(METADATA_DIR, "video_frame_dataset.csv")

    meta_df = pd.read_csv(meta_path)
    frame_df = pd.read_csv(frame_path)

    train_meta = meta_df[meta_df["split"] == "train"]
    val_meta = meta_df[meta_df["split"] == "val"]
    test_meta = meta_df[meta_df["split"] == "test"]

    print(f"Video Datasets Split: Train={len(train_meta)}, Val={len(val_meta)}, Test={len(test_meta)} videos")

    train_ds = VideoSequenceDataset(train_meta, frame_df, is_train=True)
    val_ds = VideoSequenceDataset(val_meta, frame_df, is_train=False)
    test_ds = VideoSequenceDataset(test_meta, frame_df, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    # 1. Initialize Custom CNN Spatial Backbone and Load Trained Scratch Weights
    spatial_backbone = CustomDeepfakeCNN()
    cnn_weights_path = os.path.join(MODELS_DIR, "image_deepfake_cnn.pth")
    if os.path.exists(cnn_weights_path):
        spatial_backbone.load_state_dict(torch.load(cnn_weights_path, weights_only=True))
        print("Loaded trained CustomDeepfakeCNN weights into spatial backbone.")
    else:
        print("Warning: image_deepfake_cnn.pth not found; initializing backbone with random weights.")

    # 2. Build Video Model with Spatial Backbone + BiLSTM + Temporal Attention
    video_model = CustomVideoDeepfakeModel(
        spatial_backbone=spatial_backbone,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ).to(device)

    crit_video = nn.BCEWithLogitsLoss()
    crit_frame = nn.BCEWithLogitsLoss()

    # Train temporal heads and fine-tune spatial projection
    optimizer = torch.optim.AdamW(
        [
            {"params": video_model.lstm.parameters(), "lr": 1e-3},
            {"params": video_model.temporal_attention.parameters(), "lr": 1e-3},
            {"params": video_model.frame_classifier.parameters(), "lr": 1e-3},
            {"params": video_model.video_classifier.parameters(), "lr": 1e-3},
            {"params": video_model.spatial_extractor.block4.parameters(), "lr": 1e-4}
        ],
        weight_decay=1e-3
    )

    EPOCHS = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print("\n--- Commencing Video Model Training (BiLSTM + Temporal Attention) ---")
    start_time = time.time()
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_video_epoch(video_model, train_loader, crit_video, crit_frame, optimizer, scaler, device)
        val_loss, val_acc = eval_video_epoch(video_model, val_loader, crit_video, crit_frame, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(video_model.state_dict(), os.path.join(MODELS_DIR, "video_deepfake_model.pth"))
            print(f"  --> Saved new best checkpoint: Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.1f}%")

    total_time = time.time() - start_time
    print(f"\nVideo Model Training Complete in {total_time:.1f}s.")
    history["training_time_seconds"] = total_time

    # Load best checkpoint
    video_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "video_deepfake_model.pth"), weights_only=True))

    # Evaluate on Test Split
    print("\n--- Evaluating Best Video Model on Test Set ---")
    test_results, (v_lbl, v_prd, v_prb), (f_lbl, f_prd, f_prb) = full_video_evaluation(video_model, test_loader, device)

    print("Test Set Video-Level Performance:")
    for k, v in test_results["video_level"].items():
        if isinstance(v, float):
            print(f"  {k:18s}: {v:.4f}")
        else:
            print(f"  {k:18s}: {v}")

    print("\nTest Set Frame-Level Performance:")
    for k, v in test_results["frame_level"].items():
        if isinstance(v, float):
            print(f"  {k:18s}: {v:.4f}")
        else:
            print(f"  {k:18s}: {v}")

    # Save metrics JSON
    with open(os.path.join(OUTPUTS_DIR, "video_model_evaluation_metrics.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    # 3. Generate Visualizations for Video Deepfake Detection
    print("\n--- Generating Video Model Evaluation Visualizations ---")

    # A. Video Loss & Accuracy Curves
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Training Loss", lw=2)
    axes[0].plot(epochs, history["val_loss"], "r--s", label="Validation Loss", lw=2)
    axes[0].set_title("Video Model Loss vs Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epochs", fontsize=11)
    axes[0].set_ylabel("Loss (Video + Frame)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "g-o", label="Training Acc", lw=2)
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], "m--s", label="Validation Acc", lw=2)
    axes[1].set_title("Video Model Accuracy vs Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epochs", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "video_training_curves.png"), dpi=300)
    plt.close()

    # B. Frame-level vs Video-level ROC Curves
    plt.figure(figsize=(7, 6))
    f_fpr, f_tpr, _ = roc_curve(f_lbl, f_prb)
    plt.plot(f_fpr, f_tpr, color="blue", lw=2, label=f"Frame-Level ROC (AUC = {test_results['frame_level']['roc_auc']:.4f})")

    if len(np.unique(v_lbl)) > 1:
        v_fpr, v_tpr, _ = roc_curve(v_lbl, v_prb)
        plt.plot(v_fpr, v_tpr, color="crimson", lw=2.5, linestyle="--", label=f"Video-Level ROC (AUC = {test_results['video_level']['roc_auc']:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.title("ROC Comparison: Frame-Level vs Video-Level", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "video_frame_vs_sequence_roc.png"), dpi=300)
    plt.close()

    # C. Temporal Attention Distribution across Video Frames
    attn_data = test_results["attention_analysis"]
    plt.figure(figsize=(10, 5))
    for vid_id, data in list(attn_data.items())[:4]:
        lbl_str = "REAL" if data["label"] == 0 else "FAKE"
        plt.plot(range(1, len(data["attn_weights"]) + 1), data["attn_weights"], "-o", lw=2, label=f"{vid_id} ({lbl_str})")

    plt.title("Learned Temporal Attention Weights Across Consecutive Frames", fontsize=13, fontweight="bold")
    plt.xlabel("Frame Index in Video Sequence", fontsize=11)
    plt.ylabel("Temporal Attention Weight", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "video_temporal_attention_weights.png"), dpi=300)
    plt.close()

    # D. Video Confusion Matrix Heatmap
    cm = np.array(test_results["video_level"]["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=["Pred REAL", "Pred FAKE"],
        yticklabels=["Actual REAL", "Actual FAKE"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Video Classification Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CONF_DIR, "video_confusion_matrix_heatmap.png"), dpi=300)
    plt.close()

    print("All Video Model Visualizations & Metrics Saved Successfully!")


if __name__ == "__main__":
    main()
