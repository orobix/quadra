# pylint: disable=redefined-outer-name
from __future__ import annotations

import glob
import os

import albumentations as alb
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from quadra.datasets import SegmentationDataset, SegmentationDatasetMulticlass
from quadra.utils.tests.fixtures.dataset.segmentation import (
    base_binary_segmentation_dataset,
    base_multiclass_segmentation_dataset,
)


@pytest.mark.parametrize("use_albumentations", [True, False])
@pytest.mark.parametrize("batch_size", [None, 32, 256])
def test_binary_segmentation_dataset(
    base_binary_segmentation_dataset: base_binary_segmentation_dataset,
    batch_size: int | None,
    use_albumentations: bool,
):
    data_path, arguments, _ = base_binary_segmentation_dataset

    samples = glob.glob(os.path.join(data_path, "images", "*"))
    masks = [os.path.join(data_path, "masks", os.path.basename(sample)) for sample in samples]

    if use_albumentations:
        transform = alb.Compose(
            [
                alb.Resize(224, 224),
                ToTensorV2(),
            ]
        )
    else:
        transform = None

    dataset = SegmentationDataset(
        image_paths=samples,
        mask_paths=masks,
        transform=transform,
        resize=224,
        batch_size=batch_size,
    )

    count_good = 0
    count_bad = 0
    for image, mask, target in dataset:
        if use_albumentations:
            assert isinstance(image, torch.Tensor)
            assert isinstance(mask, torch.Tensor)
            assert image.shape == (3, 224, 224)
        else:
            assert isinstance(image, np.ndarray)
            assert isinstance(mask, np.ndarray)
            assert image.shape == (224, 224, 3)

        assert mask.shape == (1, 224, 224)

        if mask.sum() == 0:
            assert target == 0
            count_good += 1
        else:
            assert target == 1
            count_bad += 1

    if batch_size is not None:
        assert (count_good + count_bad) == max(batch_size, len(samples))
    else:
        assert count_good == (arguments.train_samples[0] + arguments.val_samples[0] + arguments.test_samples[0])
        assert count_bad == (arguments.train_samples[1] + arguments.val_samples[1] + arguments.test_samples[1])

    dataloader = DataLoader(dataset, batch_size=1)
    for image, mask, target in dataloader:
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        if use_albumentations:
            assert image.shape == (1, 3, 224, 224)
        else:
            assert image.shape == (1, 224, 224, 3)
        assert mask.shape == (1, 1, 224, 224)


@pytest.mark.parametrize("use_albumentations", [True, False])
@pytest.mark.parametrize("batch_size", [None, 32, 256])
@pytest.mark.parametrize("one_hot", [True, False])
def test_multiclass_segmentation_dataset(
    base_multiclass_segmentation_dataset: base_multiclass_segmentation_dataset,
    batch_size: int | None,
    use_albumentations: bool,
    one_hot: bool,
):
    data_path, _, class_to_idx = base_multiclass_segmentation_dataset
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    samples = glob.glob(os.path.join(data_path, "images", "*"))
    masks = [os.path.join(data_path, "masks", os.path.basename(sample)) for sample in samples]

    if use_albumentations:
        transform = alb.Compose(
            [
                alb.Resize(224, 224),
                ToTensorV2(),
            ]
        )
    else:
        transform = None

    dataset = SegmentationDatasetMulticlass(
        image_paths=samples,
        mask_paths=masks,
        transform=transform,
        idx_to_class=idx_to_class,
        batch_size=batch_size,
        one_hot=one_hot,
    )

    for image, mask, _ in dataset:
        if use_albumentations:
            assert isinstance(image, torch.Tensor)
            assert isinstance(mask, np.ndarray)
            assert image.shape == (3, 224, 224)
            if one_hot:
                assert mask.shape == (len(class_to_idx) + 1, 224, 224)
            else:
                assert mask.shape == (224, 224)
        else:
            assert isinstance(image, np.ndarray)
            assert isinstance(mask, np.ndarray)
            if one_hot:
                assert mask.shape[0] == len(class_to_idx) + 1
            else:
                assert len(mask.shape) == 2

    dataloader = DataLoader(dataset, batch_size=1)
    for image, mask, _ in dataloader:
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        if use_albumentations:
            assert image.shape == (1, 3, 224, 224)

        if one_hot:
            assert mask.shape == (1, len(class_to_idx) + 1, 224, 224)
        else:
            assert mask.shape == (1, 224, 224)
