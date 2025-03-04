from __future__ import annotations

import random
from collections.abc import Iterable
from enum import Enum

import albumentations as A
import numpy as np
from torch.utils.data import Dataset


class AugmentationStrategy(Enum):
    """Augmentation Strategy for TwoAugmentationDataset."""

    SAME_IMAGE = 1
    SAME_CLASS = 2


class TwoAugmentationDataset(Dataset):
    """Two Image Augmentation Dataset for using in self-supervised learning.

    Args:
        dataset: A torch Dataset object
        transform: albumentation transformations for each image.
            If you use single transformation, it will be applied to both images.
            If you use tuple, it will be applied to first image and second image separately.
        strategy: Defaults to AugmentationStrategy.SAME_IMAGE.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: A.Compose | tuple[A.Compose, A.Compose],
        strategy: AugmentationStrategy = AugmentationStrategy.SAME_IMAGE,
    ):
        self.dataset = dataset
        self.transform = transform
        self.stategy = strategy
        if isinstance(transform, Iterable) and not isinstance(transform, str) and len(set(transform)) != 2:
            raise ValueError("transform must be an Iterable of length 2")

    def __getitem__(self, index):
        image1, target = self.dataset[index]

        if self.stategy == AugmentationStrategy.SAME_IMAGE:
            image2 = image1
        elif self.stategy == AugmentationStrategy.SAME_CLASS:
            positive_pair_idx = random.choice(np.where(self.dataset.y == target)[0])
            image2, _ = self.dataset[positive_pair_idx]
        else:
            raise ValueError("Unknown strategy")

        if isinstance(self.transform, Iterable):
            image1 = self.transform[0](image=image1)["image"]
            image2 = self.transform[1](image=image2)["image"]
        else:
            image1 = self.transform(image=image1)["image"]
            image2 = self.transform(image=image2)["image"]

        return [image1, image2], target

    def __len__(self):
        return len(self.dataset)


class TwoSetAugmentationDataset(Dataset):
    """Two Set Augmentation Dataset for using in self-supervised learning (DINO).

    Args:
        dataset: Base dataset
        global_transforms: Global transformations for each image.
        local_transform: Local transformations for each image.
        num_local_transforms: Number of local transformations to apply. In total you will have
            two + num_local_transforms transformations for each image. First element of the array will always
            return the original image.

    Example:
        >>> `images[0] = global_transform[0](original_image)`
        >>> `images[1] = global_transform[1](original_image)`
        >>> `images[2:] = local_transform(s)(original_image)`
    """

    def __init__(
        self,
        dataset: Dataset,
        global_transforms: tuple[A.Compose, A.Compose],
        local_transform: A.Compose,
        num_local_transforms: int,
    ):
        self.dataset = dataset
        self.global_transforms = global_transforms
        self.local_transform = local_transform
        self.num_local_transforms = num_local_transforms

        if num_local_transforms < 1:
            raise ValueError("num_local_transforms must be greater than 0")

    def __getitem__(self, index):
        original_image, target = self.dataset[index]
        global_outputs = []
        local_outputs = []
        for global_transform in self.global_transforms:
            global_outputs.append(global_transform(image=original_image)["image"])
        for _ in range(self.num_local_transforms):
            local_outputs.append(self.local_transform(image=original_image)["image"])
        all_outputs = global_outputs + local_outputs
        return all_outputs, target

    def __len__(self):
        return len(self.dataset)
