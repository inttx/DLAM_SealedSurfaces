from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, TrainingArguments, Trainer


def custom_resnet18(patch_size: int, num_classes: int, device: str):
    """
    Create a custom ResNet18 model for semantic segmentation.

    :param patch_size: patch size for the output logits
    :param num_classes: number of classes for segmentation
    :param device: device to load the model onto (e.g., 'cuda' or 'cpu')
    :return: ResNet18 model with modified output layer for semantic segmentation
    """
    # Load pre-trained ResNet18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Change output layer to predict per-pixel class logits
    model.fc = nn.Linear(512, patch_size * patch_size * num_classes)

    # Freeze all layers except final FC
    for params in model.parameters():
        params.requires_grad = False
    for params in model.fc.parameters():
        params.requires_grad = True

    return model.to(device)


def get_trained_custom_resnet18(model_path: str, patch_size: int, num_classes: int, device) -> nn.Module:
    """
    Load a trained custom ResNet18 model for semantic segmentation.

    :param model_path: path to the trained model weights
    :param patch_size: patch size for the output logits
    :param num_classes: number of classes for segmentation
    :param device: device to load the model onto (e.g., 'cuda' or 'cpu')
    :return: model with trained weights
    """
    # Load pre-trained ResNet18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Change output layer to predict per-pixel class logits
    model.fc = nn.Linear(512, patch_size * patch_size * num_classes)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model.to(device)


def baseline_deeplabv3_resnet101(num_classes: int, device: str) -> nn.Module:
    """
    Create a baseline DeepLabV3 model with ResNet101 backbone for semantic segmentation.

    :param num_classes: number of classes for segmentation
    :param device: device to load the model onto (e.g., 'cuda' or 'cpu')
    :return: DeepLabV3 model with modified output layer for semantic segmentation
    """
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model.to(device=device)


def get_trained_deeplabv3_resnet101(model_path: str, num_classes: int, device: str) -> nn.Module:
    """
    Load a trained DeepLabV3 model with ResNet101 backbone for semantic segmentation.

    :param model_path: path to the trained model weights
    :param num_classes: number of classes for segmentation
    :param device: device to load the model onto (e.g., 'cuda' or 'cpu')
    :return: DeepLabV3 model with trained weights
    """
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model.to(device)


def seg_former(num_classes: int, device) -> nn.Module:
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    return model.to(device)


def get_trained_segformer_model(model_path: str, num_classes: int, device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model.to(device)
