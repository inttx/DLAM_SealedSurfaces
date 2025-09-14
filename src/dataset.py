import os
import os.path as path

import torch
from torch.utils.data import Dataset, Subset

import rasterio
from rasterio.windows import Window
import numpy as np
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler
from tqdm import tqdm
from typing import List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(patch_size: int):
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Cutout(num_holes=8, max_h_size=patch_size//8, max_w_size=patch_size//8, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# Number of Patches = (img_dim - patch_size) // Stride + 1
class PotsdamDataset(Dataset):
    # Single-class version (all classes separate)
    COLOR_MAP_SINGLE = {
        (255, 255, 255): 0,  # Impervious surfaces
        (0, 0, 255): 1,  # Building
        (255, 255, 0): 2,  # Car
        (0, 255, 255): 3,  # Low vegetation
        (0, 255, 0): 4,  # Tree
        (255, 0, 0): 5,  # Clutter/background
    }

    COLOR_MAP_COMBINED = {
        # Impervious surfaces: 0
        (255, 255, 255): 0,  # Impervious surfaces
        (0, 0, 255): 0,  # Building
        (255, 255, 0): 0,  # Car
        # Previous surfaces: 1
        (0, 255, 255): 1,  # Low vegetation
        (0, 255, 0): 1,  # Tree
        # Others: 2
        (255, 0, 0): 2,  # Clutter/background
    }

    CLASS_NAMES_SINGLE = [
        'Impervious surfaces', 'Building', 'Car', 'Low vegetation', 'Tree', 'Clutter/background'
    ]

    CLASS_NAMES_COMBINED = [
        'Impervious surfaces', 'Previous surfaces', 'Others'
    ]

    @classmethod
    def get_num_classes(cls, mode='combined') -> int:
        if mode == 'single':
            return len(cls.CLASS_NAMES_SINGLE)
        else:
            return len(cls.CLASS_NAMES_COMBINED)

    def __init__(self, image_dir_path, label_dir_path, patch_size: int, stride: int, device: str,
                 mode='combined'):
        """
        Dataset for ISPRS Potsdam semantic segmentation.

        :param image_dir_path: Path to the directory containing image files.
        :param label_dir_path: Path to the directory containing label files.
        :param patch_size: Size of the patches to extract from the images and labels.
        :param stride: Stride for extracting patches from the images and labels.
        :param device: Device to load the data onto (e.g., 'cuda' or 'cpu').
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
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.is_training = True
        self.transform = get_train_transform(self.patch_size)

        # Precompute all patch positions
        self.index_map = []
        self._build_index()

        # Lazy load raster files
        self._images = None
        self._labels = None

        if self.mode == 'single':
            self._colormap = self.COLOR_MAP_SINGLE
        else:
            self._colormap = self.COLOR_MAP_COMBINED

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

    def set_is_training(self, is_training):
        self.is_training = is_training
        if self.is_training:
            self.transform = get_train_transform(self.patch_size)
        else:
            self.transform = get_val_transform()

    def __len__(self):
        return len(self.index_map) * len(self.image_files)

    def _rgb_to_class_mask(self, rgb_img):
        """
        Convert RGB mask to class index mask using COLOR_MAP.
        rgb_img: (H, W, 3) ndarray
        returns: (H, W) ndarray of class indices
        """
        mask = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.int64)
        for color, cls_idx in self._colormap.items():
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

        # Read image and label
        image_patch = self._images[file_idx].read(window=Window(col, row, self.patch_size, self.patch_size))
        image_patch = np.transpose(image_patch, (1, 2, 0)).astype(np.uint8)  # (H, W, C)

        label_rgb = self._labels[file_idx].read(window=Window(col, row, self.patch_size, self.patch_size))
        label_rgb = np.transpose(label_rgb, (1, 2, 0))  # (H, W, C)
        label_patch = self._rgb_to_class_mask(label_rgb)

        # Apply Albumentations transform (including normalization + ToTensorV2)
        if self.transform:
            augmented = self.transform(image=image_patch, mask=label_patch)
            image_patch, label_patch = augmented['image'], augmented['mask']
        else:
            # Fallback to tensor without augmentation
            image_patch = torch.from_numpy(image_patch).permute(2, 0, 1).float() / 255.0
            label_patch = torch.from_numpy(label_patch).long()

        return image_patch, label_patch


def get_data_loaders(image_dir: str,
                     label_dir: str,
                     patch_size: int,
                     stride: int,
                     dist: List[float],
                     batch_size: int,
                     device: str,
                     mode: str = "combined",
                     pin_memory: bool = False,
                     num_workers: int = 0,
                     seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test data loaders for the Potsdam dataset.

    :param image_dir: path to images
    :param label_dir: path to labels
    :param patch_size: patch size
    :param stride: stride for patch extraction
    :param dist: distribution of samples across train, validation, and test sets (e.g. [0.8, 0.1, 0.1])
    :param batch_size: batch size
    :param device: device string (e.g. 'cuda')
    :param mode: 'single' for all classes separate, 'combined' for grouped classes
    :param pin_memory: dataloader pinned memory
    :param num_workers: dataloader workers
    :param seed: random seed
    :return: train_loader, val_loader, test_loader
    """

    # Build index with one temporary dataset
    full_dataset = PotsdamDataset(image_dir, label_dir, patch_size, stride, device, mode=mode)
    num_samples = len(full_dataset)

    # Compute split sizes
    train_size = int(dist[0] * num_samples)
    val_size = int(dist[1] * num_samples)
    test_size = num_samples - train_size - val_size

    # Generate shuffled indices
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]

    # Create dataset instances for each split with correct transforms
    train_dataset_full = PotsdamDataset(image_dir, label_dir, patch_size, stride, device, mode=mode)
    train_dataset_full.set_is_training(True)
    train_dataset = Subset(train_dataset_full, train_idx)

    val_dataset_full = PotsdamDataset(image_dir, label_dir, patch_size, stride, device, mode=mode)
    val_dataset_full.set_is_training(False)
    val_dataset = Subset(val_dataset_full, val_idx)

    test_dataset_full = PotsdamDataset(image_dir, label_dir, patch_size, stride, device, mode=mode)
    test_dataset_full.set_is_training(False)
    test_dataset = Subset(test_dataset_full, test_idx)

    # Build loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
