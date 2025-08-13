from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW

from settings import *
from dataset import PotsdamDataset
from models import custom_resnet18
from train import train_loop

# Hyperparameters
patch_size = 250
stride = 250
batch_size = 512
num_epochs = 2
lr = 0.001

# Dataset
IMAGE_PATH = '../data/2_Ortho_RGB'
LABEL_PATH = '../data/5_Labels_all'
dataset = PotsdamDataset(IMAGE_PATH, LABEL_PATH, patch_size=patch_size, stride=stride, device=DEVICE)

# Model
model = custom_resnet18(patch_size=patch_size, device=DEVICE)

# Train loop
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
loss_fn = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=lr)

train_loop(train_loader=dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, num_epochs=num_epochs, device=DEVICE)

