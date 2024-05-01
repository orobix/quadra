from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import albumentations
import cv2
import numpy as np
import torch

from quadra.utils.deprecation import deprecated
from quadra.utils.imaging import keep_aspect_ratio_resize
from quadra.utils.segmentation import smooth_mask


# DEPRECATED -> we can use SegmentationDatasetMulticlass also for one class segmentation
@deprecated("Use SegmentationDatasetMulticlass instead")
class SegmentationDataset(torch.utils.data.Dataset):
    """Custom SegmentationDataset class for loading images and masks.

    Args:
        image_paths: List of paths to images.
        mask_paths: List of paths to masks.
        batch_size: Batch size.
        object_masks: List of paths to object masks.
        resize: Resize image to this size.
        mask_preprocess: Preprocess mask.
        labels: List of labels.
        transform: Transformations to apply to images and masks.
        mask_smoothing: Smooth mask.
        defect_transform: Transformations to apply to images and masks for defects.
    """

    def __init__(
        self,
        image_paths: list[str],
        mask_paths: list[str],
        batch_size: int | None = None,
        object_masks: list[np.ndarray | Any] | None = None,
        resize: int = 224,
        mask_preprocess: Callable | None = None,
        labels: list[str] | None = None,
        transform: albumentations.Compose | None = None,
        mask_smoothing: bool = False,
        defect_transform: albumentations.Compose | None = None,
    ):
        self.transform = transform
        self.defect_transform = defect_transform
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.mask_preprocess = mask_preprocess
        self.resize = resize
        self.object_masks = object_masks
        self.data_len = len(self.image_paths)
        self.batch_size = None if batch_size is None else max(batch_size, self.data_len)
        self.smooth_mask = mask_smoothing

    def __getitem__(self, index):
        # This is required to avoid infinite loop when running the dataset outside of a dataloader
        if self.batch_size is not None and self.batch_size == index:
            raise StopIteration

        if self.batch_size is None and self.data_len == index:
            raise StopIteration

        index = index % self.data_len
        image_path = self.image_paths[index]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        object_mask_path = self.object_masks[index] if self.object_masks is not None else None
        if object_mask_path is not None:
            object_mask = cv2.imread(str(object_mask_path), 0) if os.path.isfile(object_mask_path) else None
        else:
            object_mask = None
        label = self.labels[index] if self.labels is not None else None
        if (
            self.mask_paths[index] is np.nan
            or self.mask_paths[index] is None
            or not os.path.isfile(self.mask_paths[index])
        ):
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask_path = self.mask_paths[index]
            mask = cv2.imread(str(mask_path), 0)
        if self.defect_transform is not None and label == 1 and np.sum(mask) == 0:
            if object_mask is not None:
                object_mask *= 255
            aug = self.defect_transform(image=image, mask=mask, object_mask=object_mask, label=label)
            image = aug["image"]
            mask = aug["mask"]
            label = aug["label"]
        if self.mask_preprocess:
            mask = self.mask_preprocess(mask)
            if object_mask is not None:
                object_mask = self.mask_preprocess(object_mask)
        if self.resize:
            image = keep_aspect_ratio_resize(image, self.resize)
            mask = keep_aspect_ratio_resize(mask, self.resize)
            if object_mask is not None:
                object_mask = keep_aspect_ratio_resize(object_mask, self.resize)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]
        if isinstance(mask, np.ndarray):
            mask_sum = np.sum(mask)
        elif isinstance(mask, torch.Tensor):
            mask_sum = torch.sum(mask)
        else:
            raise ValueError("Unsupported type for mask")
        if mask_sum > 0 and (label is None or label == 0):
            label = 1
        if mask_sum == 0:
            label = 0

        if isinstance(image, np.ndarray):
            mask = (mask > 0).astype(np.uint8)

            if self.smooth_mask:
                mask = smooth_mask(mask)
            mask = np.expand_dims(mask, axis=0)
        else:
            mask = (mask > 0).int()
            if self.smooth_mask:
                mask = torch.from_numpy(smooth_mask(mask.numpy()))
            mask = mask.unsqueeze(0)

        return image, mask, label

    def __len__(self):
        if self.batch_size is None:
            return self.data_len

        return max(self.data_len, self.batch_size)


class SegmentationDatasetMulticlass(torch.utils.data.Dataset):
    """Custom SegmentationDataset class for loading images and multilabel masks.

    Args:
        image_paths: List of paths to images.
        mask_paths: List of paths to masks.
        idx_to_class: dict with corrispondence btw mask index and classes: {1: class_1, 2: class_2, ..., N: class_N}
        batch_size: Batch size.
        transform: Transformations to apply to images and masks.
        one_hot: if True return a binary mask (n_classxHxW), otherwise the labelled mask HxW. SMP loss requires the
            second format.
    """

    def __init__(
        self,
        image_paths: list[str],
        mask_paths: list[str],
        idx_to_class: dict,
        batch_size: int | None = None,
        transform: albumentations.Compose | None = None,
        one_hot: bool = False,
    ):
        self.transform = transform
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.idx_to_class = idx_to_class
        self.data_len = len(self.image_paths)
        self.batch_size = None if batch_size is None else max(batch_size, self.data_len)
        self.one_hot = one_hot

    def _preprocess_mask(self, mask: np.ndarray):
        """Function to preprocess the mask -> needed for albumentations
        Args:
            mask: a numpy array of dimension HxW with values in [0] + self.idx_to_class.

        Output:
            a binary numpy array with dims len(self.idx_to_class) + 1 x H x W
        """
        multilayer_mask = np.zeros((len(self.idx_to_class) + 1, *mask.shape[:2]))
        # provide background information for completeness
        # single channel mask does not use it anyway.
        multilayer_mask[0] = (mask == 0).astype(np.uint8)
        for idx in self.idx_to_class:
            multilayer_mask[int(idx)] = (mask == int(idx)).astype(np.uint8)

        return multilayer_mask

    def __getitem__(self, index):
        """Get image and mask."""
        # This is required to avoid infinite loop when running the dataset outside of a dataloader
        if self.batch_size is not None and self.batch_size == index:
            raise StopIteration
        if self.batch_size is None and self.data_len == index:
            raise StopIteration

        index = index % self.data_len
        image_path = self.image_paths[index]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if (
            self.mask_paths[index] is np.nan
            or self.mask_paths[index] is None
            or not os.path.isfile(self.mask_paths[index])
        ):
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask_path = self.mask_paths[index]
            mask = cv2.imread(str(mask_path), 0)

        # we go back to binary masks avoid transformation errors
        mask = self._preprocess_mask(mask)

        if self.transform is not None:
            masks = list(mask)
            aug = self.transform(image=image, masks=masks)
            image = aug["image"]
            mask = np.stack(aug["masks"])  # C x H x W

        # we compute single channel mask again
        # zero is the background
        if not self.one_hot:  # one hot is done by smp dice loss
            mask_out = np.zeros(mask.shape[1:])
            for i in range(1, mask.shape[0]):
                mask_out[mask[i] == 1] = i
            # mask_out shape -> HxW
        else:
            mask_out = mask
            # mask_out shape -> CxHxW where C is number of classes (included the background)

        return image, mask_out.astype(int), 0

    def __len__(self):
        """Returns the dataset lenght."""
        if self.batch_size is None:
            return self.data_len

        return max(self.data_len, self.batch_size)
