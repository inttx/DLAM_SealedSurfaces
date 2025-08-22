import os
import os.path as path

import torch
from torch.utils.data import Dataset

import rasterio
from rasterio.windows import Window
import numpy as np
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler
from tqdm import tqdm
from typing import List, Tuple


# Number of Patches = (img_dim - patch_size) // Stride + 1
class PotsdamDataset(Dataset):
    COLOR_MAP = {
        # Impervious surfaces: 0
        (255, 255, 255): 0,  # Impervious surfaces
        (0, 0, 255): 0,  # Building
        (255, 0, 0): 0,  # Car
        # Pervious surfaces: 1
        (0, 255, 255): 1,  # Low vegetation
        (0, 255, 0): 1,  # Tree
        # Others: 2
        (255, 0, 255): 2,  # Clutter/background
    }

    CLASS_NAMES = [
        'Impervious surfaces',
        'Pervious surfaces',
        'Others'
    ]

    @staticmethod
    def get_num_classes() -> int:
        """
        Return the number of distinct segmentation classes in the labels
        :return: number of segmentation classes
        """
        assert len(PotsdamDataset.CLASS_NAMES) == len(set(PotsdamDataset.COLOR_MAP.values())), f"COLOR_MAP and CLASS_NAMES differ in the number of classes"
        return len(PotsdamDataset.CLASS_NAMES)

    def __init__(self, image_dir_path, label_dir_path, patch_size: int, stride: int, device: str, transform=None):
        """
        Dataset for ISPRS Potsdam semantic segmentation.

        :param image_dir_path: Path to the directory containing image files.
        :param label_dir_path: Path to the directory containing label files.
        :param patch_size: Size of the patches to extract from the images and labels.
        :param stride: Stride for extracting patches from the images and labels.
        :param device: Device to load the data onto (e.g., 'cuda' or 'cpu').
        :param transform: Optional transform to apply to the image and label patches.
        """
        self.image_dir = image_dir_path
        self.label_dir = label_dir_path
        self.image_files = sorted(
            [path.join(image_dir_path, f) for f in os.listdir(image_dir_path) if f.endswith(".tif")]
        )
        self.label_files = sorted(
            [path.join(label_dir_path, f) for f in os.listdir(label_dir_path) if f.endswith(".tif")]
        )

        self.device = device
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride

        # Precompute all patch positions
        self.index_map = []
        self._build_index()

        # Lazy load raster files
        self._images = None
        self._labels = None

        # Create colormap array for fast RGB→class mapping
        self._colormap_arr = np.array(list(self.COLOR_MAP.keys()))
        self._class_indices = np.array(list(self.COLOR_MAP.values()))

    def _build_index(self):
        with rasterio.open(self.image_files[0]) as img:
            height, width = img.height, img.width
            for row in tqdm(range(0, height - self.patch_size + 1, self.stride), desc="Building index"):
                for col in range(0, width - self.patch_size + 1, self.stride):
                    self.index_map.append((row, col))

    def _init_files(self):
        """Open raster files only once."""
        self._images = [rasterio.open(f) for f in self.image_files]
        self._labels = [rasterio.open(f) for f in self.label_files]

    def __len__(self):
        return len(self.index_map) * len(self.image_files)

    def _rgb_to_class_mask(self, rgb_img):
        """
        Convert RGB mask to class index mask using COLOR_MAP.
        rgb_img: (H, W, 3) ndarray
        returns: (H, W) ndarray of class indices
        """
        mask = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.int64)
        for color, cls_idx in self.COLOR_MAP.items():
            mask[np.all(rgb_img == np.array(color), axis=-1)] = cls_idx
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a patch of image and label at the specified index.

        :param idx: index of the patch to retrieve
        :return:
        image_patch: Float tensor of shape (3, patch_size, patch_size), normalized to ImageNet stats
        label_patch: Long tensor of shape (patch_size, patch_size) with class indices (0..5)
        """
        if self._images is None:
            self._init_files()

        row, col = self.index_map[idx % len(self.index_map)]
        file_idx = idx // len(self.index_map)

        # Read and normalize image
        image_patch = self._images[file_idx].read(window=Window(col, row, self.patch_size, self.patch_size))
        image_patch = torch.from_numpy(image_patch).float() / 255.0  # (C, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_patch = (image_patch - mean) / std

        # Read and convert label to class indices
        label_rgb = self._labels[file_idx].read(window=Window(col, row, self.patch_size, self.patch_size))
        label_rgb = np.transpose(label_rgb, (1, 2, 0))  # (H, W, 3)
        label_patch = torch.from_numpy(self._rgb_to_class_mask(label_rgb)).long()

        if self.transform:
            image_patch, label_patch = self.transform(image_patch, label_patch)

        return image_patch, label_patch


def get_data_loaders(dataset: PotsdamDataset, dist: List[float], batch_size: int, pin_memory: bool = False,
                     num_workers: int = 0, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test data loaders for the Potsdam dataset.

    :param dataset: dataset to split into train, validation, and test sets
    :param dist: distribution of samples across train, validation, and test sets
    :param batch_size: batch size for the data loaders
    :param pin_memory: if True, data loader will use pinned memory for faster data transfer to GPU
    :param num_workers: number of subprocesses to use for data loading
    :param seed: random seed for reproducibility
    :return: train_loader, val_loader, test_loader
    """
    train_size = int(dist[0] * len(dataset))
    val_size = int(dist[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(seed))

    # TODO: put shuffle=True after testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
