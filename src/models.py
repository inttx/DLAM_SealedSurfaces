from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, TrainingArguments, Trainer


def custom_resnet18(patch_size, device):
    # Load pre-trained ResNet18 model
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Configure custom layer
    model.fc = nn.Linear(512 * 1, patch_size * patch_size * 3)

    # Freeze model parameters except for custom layer
    for params in model.parameters():
        params.requires_grad = False
    for params in model.fc.parameters():
        params.requires_grad = True

    return model.to(device)


def baseline_deeplabv3plus_resnet101(num_classes: int, device):
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model.to(device=device)


def seg_former(num_classes: int, device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    return model.to(device)
