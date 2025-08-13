import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

CLASS_NAMES = ['Impervious surfaces',
               'Building',
               'Low vegetation',
               'Tree',
               'Car',
               'Clutter/background']


def evaluate_segmentation(model, data_loader, device, model_type, num_classes, eval_path):
    model.eval()
    total_correct = 0
    total_pixels = 0
    iou_per_class = np.zeros(num_classes, dtype=np.float64)
    total_iou_counts = np.zeros(num_classes, dtype=np.float64)

    all_preds_flat = []
    all_labels_flat = []

    with torch.no_grad():
        for X, y in tqdm(data_loader, desc="Evaluating"):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # [B, H, W] ground truth class IDs

            # Forward pass
            if model_type == 'SegFormer':
                logits = model(pixel_values=X).logits
            else:
                logits = model(X)
                if model_type == 'DeepLabV3':
                    logits = logits['out']  # take main output

            # Upsample logits to match label size
            logits = F.interpolate(logits, size=(y.shape[1], y.shape[2]), mode='bilinear', align_corners=False)

            # Predicted class per pixel
            preds = torch.argmax(logits, dim=1)  # [B, H, W]

            # Pixel accuracy
            correct = (preds == y).sum().item()
            total_correct += correct
            total_pixels += torch.numel(y)

            # IoU computation
            preds_np = preds.cpu().numpy().reshape(-1)
            y_np = y.cpu().numpy().reshape(-1)
            all_preds_flat.extend(preds_np)
            all_labels_flat.extend(y_np)

            cm = confusion_matrix(y_np, preds_np, labels=list(range(num_classes)))
            intersection = np.diag(cm)
            union = (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
            for c in range(num_classes):
                if union[c] > 0:
                    iou_per_class[c] += intersection[c] / union[c]
                    total_iou_counts[c] += 1

    pixel_accuracy = total_correct / total_pixels
    mean_iou = (iou_per_class / np.maximum(total_iou_counts, 1)).mean()
    per_class_iou = (iou_per_class / np.maximum(total_iou_counts, 1))

    # Confusion matrix plot (normalized)
    cm_total = confusion_matrix(all_labels_flat, all_preds_flat, labels=list(range(num_classes)))
    cm_normalized = cm_total.astype('float') / cm_total.sum(axis=1)[:, np.newaxis]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized,
                                  display_labels=CLASS_NAMES)
    disp.plot()
    plt.title("Segmentation Confusion Matrix (Normalized)")
    plt.savefig(eval_path)

    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print("Per-class IoU:")
    for c, iou in enumerate(per_class_iou):
        print(f"  Class {c}: {iou:.4f}")