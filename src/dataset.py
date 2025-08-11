import os
import os.path as path

import torch
from torch.utils.data import Dataset

import rasterio
from rasterio.windows import Window
import numpy as np
from torch.utils.data import random_split, DataLoader



# Number of Patches = (img_dim - patch_size) // Stride + 1
class PotsdamDataset(Dataset):
    COLOR_MAP = {
        (255, 255, 255): 0,  # Impervious surfaces
        (0, 0, 255): 1,  # Building
        (0, 255, 255): 2,  # Low vegetation
        (0, 255, 0): 3,  # Tree
        (255, 0, 0): 4,  # Car
        (255, 0, 255): 5,  # Clutter/background
    }

    def __init__(self, image_dir_path, label_dir_path, patch_size, stride, device, no_color_labels=False, transform=None):
        self.image_dir = image_dir_path
        self.label_dir = label_dir_path
        self.image_files = [path.join(image_dir_path, f) for f in sorted(os.listdir(image_dir_path)) if
                            path.isfile(path.join(image_dir_path, f)) and path.join(image_dir_path, f).endswith('.tif')]
        self.image_labels = [path.join(label_dir_path, f) for f in sorted(os.listdir(label_dir_path)) if
                             path.isfile(path.join(label_dir_path, f)) and path.join(label_dir_path, f).endswith(
                                 '.tif')]

        self.device = device
        self.no_color_labels = no_color_labels
        self.transform = transform

        self.patch_size = patch_size
        self.stride = stride
        self.index_map = []
        self._build_index()
        self.index_map = self.index_map

    def _build_index(self):
        with rasterio.open(self.image_files[0]) as img:
            height, width = img.height, img.width
            for row in range(0, height - self.patch_size + 1, self.stride):
                for col in range(0, width - self.patch_size + 1, self.stride):
                    self.index_map.append((row, col))

    def __len__(self):
        return len(self.index_map) * len(self.image_files)

    def __getitem__(self, idx):
        row, col = self.index_map[idx % len(self.index_map)]
        file_idx = idx // len(self.index_map)
        with rasterio.open(self.image_files[file_idx]) as img:
            image_patch = img.read(window=Window(col, row, self.patch_size, self.patch_size))
            image_patch = torch.from_numpy(image_patch).float().to(self.device)

        with rasterio.open(self.image_labels[file_idx]) as img:
            label_patch = img.read(window=Window(col, row, self.patch_size, self.patch_size))

            if self.no_color_labels:
                label_patch = np.transpose(label_patch, (1, 2, 0))
                class_mask = np.zeros((self.patch_size, self.patch_size), dtype=np.int64)

                for color, idx in self.COLOR_MAP.items():
                    match = np.all(label_patch == np.array(color), axis=-1)
                    class_mask[match] = idx

                # Return tensor
                return image_patch, torch.from_numpy(class_mask).long().to(self.device)
            label_patch = torch.from_numpy(label_patch).float().to(self.device)
        # TODO: Apply Transform

        return image_patch, label_patch


def get_data_loaders(dataset, dist, batch_size, seed=42):
    train_size = int(dist[0] * len(dataset))
    val_size = int(dist[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader
