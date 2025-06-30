import os
import os.path as path

import torch
from torch.utils.data import Dataset

import rasterio
from rasterio.windows import Window


# Number of Patches = (img_dim - patch_size) // Stride + 1
class PotsdamDataset(Dataset):
    def __init__(self, image_dir_path, label_dir_path, patch_size, stride, device, transform=None):
        self.image_dir = image_dir_path
        self.label_dir = label_dir_path
        self.image_files = [path.join(image_dir_path, f) for f in sorted(os.listdir(image_dir_path)) if
                            path.isfile(path.join(image_dir_path, f)) and path.join(image_dir_path, f).endswith('.tif')]
        self.image_labels = [path.join(label_dir_path, f) for f in sorted(os.listdir(label_dir_path)) if
                             path.isfile(path.join(label_dir_path, f)) and path.join(label_dir_path, f).endswith(
                                 '.tif')]

        self.device = device

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
            label_patch = torch.from_numpy(label_patch).float().to(self.device)

        # TODO: Apply Transform

        return image_patch, label_patch
