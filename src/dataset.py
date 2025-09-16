import abc
from abc import ABCMeta

import os
import os.path as path

from types import MappingProxyType

import torch
from torch.utils.data import Dataset, Subset

import rasterio
from rasterio.windows import Window
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations import ToTensorV2


class CustomDataset(Dataset, metaclass=ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def color_map_single() -> MappingProxyType[tuple[int, int, int], int]:
        """
        Color map for single-class segmentation (all classes separate).

        Returns a mapping from RGB tuples to class indices.
        To be implemented in subclasses for specific datasets.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def color_map_combined() -> MappingProxyType[tuple[int, int, int], int]:
        """
        Color map for combined-class segmentation (impervious/pervious/others).

        Returns a mapping from RGB tuples to class indices (0/1/2).
        To be implemented in subclasses for specific datasets.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def class_names_single() -> tuple[str, ...]:
        """
        Class names for single-class segmentation (all classes separate).

        To be implemented in subclasses for specific datasets.
        """
        ...

    @staticmethod
    def class_names_combined() -> tuple[str, ...]:
        """
        Class names for combined-class segmentation.

        All datasets use the same combined classes.
        The original single classes are combined to these three classes.
        """
        return 'Impervious surfaces', 'Pervious surfaces', 'Others'

    @classmethod
    def get_num_classes(cls, mode='combined') -> int:
        """
        Get number of classes based on the mode.

        :param mode: 'single' for all classes separate, 'combined' for grouped classes
        :return: number of classes
        """
        if mode == 'single':
            return len(cls.class_names_single())
        if mode == 'combined':
            return len(cls.class_names_combined())

        raise ValueError(f"Invalid mode: {mode}. Choose 'single' or 'combined'.")

    def __init__(
            self,
            image_dir_path: str, label_dir_path: str,
            patch_size: int, stride: int,
            device: str,
            mode: str = "combined"
    ):
        """
        Setup custom dataset for image semantic segmentation.

        :param image_dir_path: Path to the directory containing image files.
        :param label_dir_path: Path to the directory containing label files.
        :param patch_size: Size of the patches to extract from the images and labels.
        :param stride: Stride for extracting patches from the images and labels.
        :param device: Device to load the data onto (e.g., 'cuda' or 'cpu').
        :param mode: 'single' for all classes separate, 'combined' for dividing classes into impervious/pervious/others
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
        self.transform = self._get_train_transform()

        # Precompute all patch positions
        self.index_map = []
        self._build_index()

        # Lazy load raster files
        self._images = None
        self._labels = None

        if self.mode == 'single':
            self._colormap = self.color_map_single()
        else:
            self._colormap = self.color_map_combined()

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

    def set_is_training(self, is_training: bool):
        self.is_training = is_training
        if self.is_training:
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_val_transform()

    def __len__(self):
        return len(self.index_map) * len(self.image_files)

    def _rgb_to_class_mask(self, rgb_img):
        """
        Convert RGB mask to class index mask using the color map.
        rgb_img: (H, W, 3) ndarray
        returns: (H, W) ndarray of class indices
        """
        mask = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.int64)
        for color, cls_idx in self._colormap.items():
            mask[np.all(rgb_img == np.array(color), axis=-1)] = cls_idx
        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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
            image_patch, label_patch = augmented['image'], augmented['mask'].long()
        else:
            # Fallback to tensor without augmentation
            image_patch = torch.from_numpy(image_patch).permute(2, 0, 1).float() / 255.0
            label_patch = torch.from_numpy(label_patch).long()

        return image_patch, label_patch

    @staticmethod
    def _get_normalize_transform() -> A.Normalize:
        """ Normalize to ImageNet statistics. """
        return A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    @classmethod
    def _get_train_transform(cls) -> A.Compose:
        return A.Compose([
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=45, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(p=0.3),
            A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
            cls._get_normalize_transform(),
            ToTensorV2(),
        ])

    @classmethod
    def _get_val_transform(cls):
        return A.Compose([
            cls._get_normalize_transform(),
            ToTensorV2(),
        ])

    @classmethod
    def get_data_loaders(
            cls,
            image_dir: str,
            label_dir: str,
            patch_size: int,
            stride: int,
            dist: tuple[float, float, float],
            batch_size: int,
            device: str,
            mode: str = "combined",
            pin_memory: bool = False,
            num_workers: int = 0,
            seed: int = 42
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, validation, and test data loaders for the dataset.

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
        full_dataset = cls(image_dir, label_dir, patch_size, stride, device, mode=mode)
        num_samples = len(full_dataset)

        # Compute split sizes
        train_size = int(dist[0] * num_samples)
        val_size = int(dist[1] * num_samples)

        # Generate shuffled indices
        indices = np.arange(num_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        # Create dataset instances for each split with correct transforms
        train_dataset_full = cls(image_dir, label_dir, patch_size, stride, device, mode=mode)
        train_dataset_full.set_is_training(True)
        train_dataset = Subset(train_dataset_full, train_idx)

        val_dataset_full = cls(image_dir, label_dir, patch_size, stride, device, mode=mode)
        val_dataset_full.set_is_training(False)
        val_dataset = Subset(val_dataset_full, val_idx)

        test_dataset_full = cls(image_dir, label_dir, patch_size, stride, device, mode=mode)
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


# Number of Patches = (img_dim - patch_size) // Stride + 1
class PotsdamDataset(CustomDataset):
    @staticmethod
    def color_map_single() -> MappingProxyType[tuple[int, int, int], int]:
        return MappingProxyType({
            (255, 255, 255): 0,  # Impervious surfaces
            (0, 0, 255): 1,  # Building
            (255, 255, 0): 2,  # Car
            (0, 255, 255): 3,  # Low vegetation
            (0, 255, 0): 4,  # Tree
            (255, 0, 0): 5,  # Clutter/background
        })

    @staticmethod
    def color_map_combined() -> MappingProxyType[tuple[int, int, int], int]:
        return MappingProxyType({
            # Impervious surfaces: 0
            (255, 255, 255): 0,  # Impervious surfaces
            (0, 0, 255): 0,  # Building
            (255, 255, 0): 0,  # Car
            # Pervious surfaces: 1
            (0, 255, 255): 1,  # Low vegetation
            (0, 255, 0): 1,  # Tree
            # Others: 2
            (255, 0, 0): 2,  # Clutter/background
        })

    @staticmethod
    def class_names_single() -> tuple[str, ...]:
        return (
            'Impervious surfaces',
            'Building',
            'Car',
            'Low vegetation',
            'Tree',
            'Clutter/background'
        )


class HessigheimDataset(CustomDataset):
    @staticmethod
    def color_map_single() -> MappingProxyType[tuple[int, int, int], int]:
        return MappingProxyType({
            (178, 203, 47): 0,  # Low vegetation
            (183, 178, 170): 1,  # Impervious surface
            (32, 151, 163): 2,  # Vehicle
            (168, 33, 107): 3,  # Urban furniture
            (255, 122, 89): 4,  # Roof
            (255, 215, 136): 5,  # Facade
            (89, 125, 53): 6,  # Shrub
            (0, 128, 65): 7,  # Tree
            (170, 85, 0): 8,  # Soil/Gravel
            (252, 225, 5): 9,  # Vertical surface
            (128, 0, 0): 10,  # Chimney
        })

    @staticmethod
    def color_map_combined() -> MappingProxyType[tuple[int, int, int], int]:
        return MappingProxyType({  # Impervious surfaces: 0
            (183, 178, 170): 0,  # Impervious surface
            (32, 151, 163): 0,  # Vehicle
            (168, 33, 107): 0,  # Urban furniture
            (255, 122, 89): 0,  # Roof
            (255, 215, 136): 0,  # Facade
            (252, 225, 5): 0,  # Vertical surface
            (128, 0, 0): 0,  # Chimney
            # Pervious surfaces: 1
            (178, 203, 47): 0,  # Low vegetation
            (89, 125, 53): 1,  # Shrub
            (0, 128, 65): 1,  # Tree
            (170, 85, 0): 1,  # Soil/Gravel
            # Others: 2
        })

    @staticmethod
    def class_names_single() -> tuple[str, ...]:
        return (
            'Low vegetation',
            'Impervious surface',
            'Vehicle',
            'Urban furniture',
            'Roof',
            'Facade',
            'Shrub',
            'Tree',
            'Soil/Gravel',
            'Vertical surface',
            'Chimney'
        )