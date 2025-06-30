from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


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
