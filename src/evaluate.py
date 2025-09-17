import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, balanced_accuracy_score, classification_report
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import Module


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


def plot_metric_bar(metric_values, class_names, metric_name, eval_path):
    """
    Plot results of metrics in a bar-plot
    """
    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(num_classes), metric_values, tick_label=class_names)

    ax.set_ylim(0, 1)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Per-class {metric_name}")
    plt.xticks(rotation=45, ha="right")

    # Werte über die Balken schreiben
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = eval_path.replace(".svg", f"_{metric_name.lower()}_bar.svg")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)



def evaluate_segmentation(model, data_loader, device, model_type, class_names, patch_size, eval_path):
    """
    Evaluate a segmentation model on a dataset and save evaluation results.

    Metrics:
        - Pixel Accuracy
        - Balanced Accuracy
        - Mean IoU
        - Precision (macro + per-class)
        - Recall (macro + per-class)
        - F1 Score (macro)
        - Confusion Matrix (normalized)

    Also generates and saves:
        - Normalized confusion matrix (per model)
        - Per-class bar plots for Precision, Recall, and IoU

    Args:
        model: Trained segmentation model.
        data_loader: DataLoader for evaluation.
        device: Computation device (CPU/GPU).
        model_type (str): Model name (e.g., 'ResNet18', 'SegFormer').
        class_names (list): List of class labels.
        patch_size (int): Patch size (for ResNet18 reshaping).
        eval_path (str): Base path for saving plots (.svg).
    Returns:
        dict: Aggregated evaluation results.
    """

    model.eval()
    total_correct = 0
    total_pixels = 0
    num_classes = len(class_names)

    # Running totals
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


    bal_acc = balanced_accuracy_score(all_labels_flat, all_preds_flat)
    precision = precision_score(all_labels_flat, all_preds_flat, average="macro", zero_division=0)
    precision_per_class = precision_score(all_labels_flat, all_preds_flat, average=None, labels=range(num_classes), zero_division=0)
    recall = recall_score(all_labels_flat, all_preds_flat, average="macro", zero_division=0)
    recall_per_class = recall_score(all_labels_flat, all_preds_flat, average=None, labels=range(num_classes), zero_division=0)
    f1 = f1_score(all_labels_flat, all_preds_flat, average="macro", zero_division=0)
    f1_per_class = f1_score(all_labels_flat, all_preds_flat, average=None, labels=range(num_classes), zero_division=0)

    print("Evaluation completed.")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print("Per-class IoU:")
    for c, iou in enumerate(per_class_iou):
        print(f"  {class_names[c]}: {iou:.4f}")

    print("\nClassification report:\n")
    print(classification_report(all_labels_flat, all_preds_flat, target_names=class_names, zero_division=0))


    # ---- Efficient plotting ----

    # Plotting Confusion Matrix
    MAX_CLASSES_TO_PLOT = 30
    plot_classes = min(num_classes, MAX_CLASSES_TO_PLOT)
    cm_normalized = cm_total.astype(float) / np.maximum(cm_total.sum(axis=1, keepdims=True), 1)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized[:plot_classes, :plot_classes],
        display_labels=class_names[:plot_classes]
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='viridis', colorbar=False)
    plt.title(f"{model_type} Confusion Matrix")
    plt.tight_layout()

    base_path = eval_path.replace(".svg", "")
    save_path = f"{base_path}_confusion_matrix.svg"

    plt.savefig(save_path, dpi=150)
    plt.close(fig)


    # Plotting Barplots

    # Precision
    precision_per_class = precision_score(all_labels_flat, all_preds_flat,
                                      average=None, labels=range(num_classes), zero_division=0)
    plot_metric_bar(precision_per_class, class_names, "Precision", eval_path)

    # Recall
    recall_per_class = recall_score(all_labels_flat, all_preds_flat,
                                average=None, labels=range(num_classes), zero_division=0)
    plot_metric_bar(recall_per_class, class_names, "Recall", eval_path)

    # IoU
    plot_metric_bar(per_class_iou, class_names, "IoU", eval_path)


    # returns values for later comparison
    results = {
        "model": model_type,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "iou_per_class": per_class_iou,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "mean_iou": mean_iou
    }
    return results
