"""
Fast, GPU-Accelerated Training, Optimization, Evaluation, and Explainability Pipeline.
Trains CustomDeepfakeCNN from scratch using random initialization (Kaiming Normal).
Utilizes pre-cached memory arrays, automatic mixed precision (AMP), and Cosine Annealing.
"""

import sys
import os
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)

from src.models.custom_cnn import CustomDeepfakeCNN
from src.explainability.gradcam import GradCAM

# Strict reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

METADATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "metadata")
PROCESSED_IMG_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed", "images")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")
HEATMAPS_DIR = os.path.join(OUTPUTS_DIR, "heatmaps")
CONF_DIR = os.path.join(OUTPUTS_DIR, "confusion_matrices")
PRED_DIR = os.path.join(OUTPUTS_DIR, "sample_predictions")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(HEATMAPS_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


class FastMemoryDataset(Dataset):
    """Zero-disk-overhead in-memory dataset for lightning-fast training."""
    def __init__(self, images_npy, labels_npy, is_train=False):
        self.images = np.load(images_npy)  # (N, 128, 128, 3) uint8
        self.labels = np.load(labels_npy)  # (N,) float32
        self.is_train = is_train

        # Standard ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (128, 128, 3) uint8

        # Fast data augmentation on train
        if self.is_train and random.random() > 0.5:
            img = np.fliplr(img).copy()

        # Convert to Tensor (3, 128, 128) in [0, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return tensor, label, idx


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def full_evaluation(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_indices = []

    with torch.no_grad():
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(images)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().astype(int).tolist())
            all_indices.extend(indices.numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr)
    }

    return metrics, all_labels, all_preds, all_probs, all_indices


def plot_and_save_visualizations(history, test_metrics, labels, preds, probs, raw_test_images, raw_paths, model, device):
    print("\n--- Generating Evaluation and Explainability Visualizations ---")
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Training & Validation Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    plt.title("Training and Validation Loss vs Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "training_validation_loss.png"), dpi=300)
    plt.close()

    # 2. Training & Validation Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "g-o", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "m--s", label="Validation Accuracy", linewidth=2)
    plt.title("Training and Validation Accuracy vs Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "training_validation_accuracy.png"), dpi=300)
    plt.close()

    # 3. Learning Rate Progression
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["learning_rate"], "c-d", label="Learning Rate", linewidth=2)
    plt.title("Learning Rate Schedule vs Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "learning_rate_schedule.png"), dpi=300)
    plt.close()

    # 4. Confusion Matrix Heatmap
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted REAL", "Predicted FAKE"],
        yticklabels=["Actual REAL", "Actual FAKE"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Test Confusion Matrix (Acc: {test_metrics['accuracy']*100:.2f}%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CONF_DIR, "confusion_matrix_heatmap.png"), dpi=300)
    plt.close()

    # 5. ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC Curve (AUC = {test_metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "roc_curve.png"), dpi=300)
    plt.close()

    # 6. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(recall_vals, precision_vals, color="purple", lw=2.5, label=f"PR Curve (F1 = {test_metrics['f1_score']:.4f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall (PR) Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "precision_recall_curve.png"), dpi=300)
    plt.close()

    # 7. Grad-CAM Explainability Visualizations
    gradcam = GradCAM(model, target_layer=model.block4.conv2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    # Pick samples: Real TP, Fake TP, FP, FN
    tp_idx = np.where((labels == 1) & (preds == 1))[0]
    tn_idx = np.where((labels == 0) & (preds == 0))[0]
    fp_idx = np.where((labels == 0) & (preds == 1))[0]
    fn_idx = np.where((labels == 1) & (preds == 0))[0]

    selected = []
    for arr in [tn_idx, tp_idx, fp_idx, fn_idx]:
        if len(arr) > 0:
            selected.append(arr[0])

    for i, idx in enumerate(selected):
        raw_img = raw_test_images[idx]  # (128, 128, 3) uint8
        tensor_img = torch.from_numpy(raw_img).permute(2, 0, 1).float().to(device) / 255.0
        tensor_img = ((tensor_img - mean) / std).unsqueeze(0)

        actual = "REAL" if labels[idx] == 0 else "FAKE"
        predicted = "REAL" if preds[idx] == 0 else "FAKE"
        conf = probs[idx] if preds[idx] == 1 else 1 - probs[idx]

        cam = gradcam.generate_heatmap(tensor_img, target_class=preds[idx])
        hm_color, overlay = gradcam.overlay_heatmap(raw_img, cam, alpha=0.5)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(raw_img)
        axes[0].set_title(f"Original Image\nActual: {actual}", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM Activation Map\n(Model Attention Gradients)", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay Visual Explanation\nPred: {predicted} ({conf*100:.1f}%)", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        save_name = f"gradcam_sample_{i+1}_{actual}_pred_{predicted}.png"
        plt.savefig(os.path.join(HEATMAPS_DIR, save_name), dpi=300)
        plt.close()

    print("Saved all graphs, curves, and Grad-CAM heatmaps successfully!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_img_p = os.path.join(PROCESSED_IMG_DIR, "train_images.npy")
    train_lbl_p = os.path.join(PROCESSED_IMG_DIR, "train_labels.npy")
    val_img_p = os.path.join(PROCESSED_IMG_DIR, "val_images.npy")
    val_lbl_p = os.path.join(PROCESSED_IMG_DIR, "val_labels.npy")
    test_img_p = os.path.join(PROCESSED_IMG_DIR, "test_images.npy")
    test_lbl_p = os.path.join(PROCESSED_IMG_DIR, "test_labels.npy")
    cross_img_p = os.path.join(PROCESSED_IMG_DIR, "cross_images.npy")
    cross_lbl_p = os.path.join(PROCESSED_IMG_DIR, "cross_labels.npy")

    train_ds = FastMemoryDataset(train_img_p, train_lbl_p, is_train=True)
    val_ds = FastMemoryDataset(val_img_p, val_lbl_p, is_train=False)
    test_ds = FastMemoryDataset(test_img_p, test_lbl_p, is_train=False)
    cross_ds = FastMemoryDataset(cross_img_p, cross_lbl_p, is_train=False)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    cross_loader = DataLoader(cross_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f"Dataset Counts: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, Cross={len(cross_ds)}")

    # Initialize Custom Model from Scratch (No Pretrained Weights!)
    model = CustomDeepfakeCNN(num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    EPOCHS = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": []
    }

    print("\n--- Commencing Model Training (from Scratch) ---")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | LR: {current_lr:.6f}")

        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "image_deepfake_cnn.pth"))
            print(f"  --> Saved new best checkpoint: Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

    total_training_time = time.time() - start_time
    print(f"\nTraining Complete in {total_training_time:.1f} seconds ({total_training_time/60.0:.2f} mins).")
    history["total_training_time_seconds"] = total_training_time
    with open(os.path.join(OUTPUTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "image_deepfake_cnn.pth"), weights_only=True))

    # 1. In-Distribution Test Set Evaluation
    print("\n--- Evaluating Best Model on In-Distribution Test Set ---")
    test_metrics, labels, preds, probs, indices = full_evaluation(model, test_loader, device)
    print("In-Distribution Test Results:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:22s}: {v:.4f}")
        else:
            print(f"  {k:22s}: {v}")

    # 2. Cross-Dataset Evaluation (CIPLAB)
    print("\n--- Evaluating Model on Cross-Dataset Benchmark (CIPLAB) ---")
    cross_metrics, _, _, _, _ = full_evaluation(model, cross_loader, device)
    print("Cross-Dataset Generalization Results (CIPLAB):")
    for k, v in cross_metrics.items():
        if isinstance(v, float):
            print(f"  {k:22s}: {v:.4f}")
        else:
            print(f"  {k:22s}: {v}")

    evaluation_report = {
        "in_distribution_test": test_metrics,
        "cross_dataset_test": cross_metrics,
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss)
    }

    with open(os.path.join(OUTPUTS_DIR, "model_evaluation_metrics.json"), "w") as f:
        json.dump(evaluation_report, f, indent=4)

    # 3. Generate Visualizations, Curves, and Heatmaps
    raw_test_imgs = np.load(test_img_p)
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test_split.csv"))
    raw_paths = df_test["path"].values

    plot_and_save_visualizations(
        history, test_metrics, labels, preds, probs,
        raw_test_imgs, raw_paths, model, device
    )

    print("\nImage CNN Training & Evaluation Pipeline Finished Successfully!")


if __name__ == "__main__":
    main()
