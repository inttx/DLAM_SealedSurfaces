import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import Module

CLASS_NAMES = ['Impervious surfaces',
               'Building',
               'Low vegetation',
               'Tree',
               'Car',
               'Clutter/background']


def fast_confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix fast using numpy bincount.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    mask = (y_true >= 0) & (y_true < num_classes)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    cm = np.bincount(
        y_true * num_classes + y_pred,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return cm


def evaluate_segmentation(model, data_loader, device, model_type, num_classes,
                          patch_size, eval_path):
    model.eval()
    total_correct = 0
    total_pixels = 0

    # Running totals
    iou_sum = np.zeros(num_classes, dtype=np.float64)
    iou_count = np.zeros(num_classes, dtype=np.float64)
    all_preds_flat = []
    all_labels_flat = []

    with torch.no_grad():
        for X, y in tqdm(data_loader, desc="Evaluating"):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward pass
            if model_type == 'SegFormer':
                logits = model(pixel_values=X).logits
                logits = F.interpolate(logits, size=(y.shape[1], y.shape[2]), mode='bilinear', align_corners=False)
            else:
                logits = model(X)
                if model_type == 'DeepLabV3':
                    logits = logits['out']
                if model_type == 'ResNet18':
                    logits = logits.view(X.size(0), num_classes, patch_size, patch_size)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_pixels += y.numel()

            # Flatten and store for final confusion matrix
            preds_np = preds.cpu().numpy().reshape(-1)
            y_np = y.cpu().numpy().reshape(-1)
            all_preds_flat.append(preds_np)
            all_labels_flat.append(y_np)

    # Concatenate all predictions and labels
    all_preds_flat = np.concatenate(all_preds_flat)
    all_labels_flat = np.concatenate(all_labels_flat)

    # Compute full confusion matrix fast
    cm_total = fast_confusion_matrix(all_labels_flat, all_preds_flat, num_classes)

    # Compute per-class IoU
    intersection = np.diag(cm_total)
    union = cm_total.sum(axis=1) + cm_total.sum(axis=0) - intersection
    per_class_iou = np.where(union > 0, intersection / union, 0.0)
    mean_iou = per_class_iou.mean()

    pixel_accuracy = total_correct / total_pixels

    print("Evaluation completed.")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print("Per-class IoU:")
    for c, iou in enumerate(per_class_iou):
        print(f"  Class {c}: {iou:.4f}")

    # ---- Efficient plotting ----
    MAX_CLASSES_TO_PLOT = 30
    plot_classes = min(num_classes, MAX_CLASSES_TO_PLOT)
    cm_normalized = cm_total.astype(float) / np.maximum(cm_total.sum(axis=1, keepdims=True), 1)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized[:plot_classes, :plot_classes],
        display_labels=CLASS_NAMES[:plot_classes]
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='viridis', colorbar=True)
    plt.title("Segmentation Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig(eval_path, dpi=150)
    plt.close(fig)
