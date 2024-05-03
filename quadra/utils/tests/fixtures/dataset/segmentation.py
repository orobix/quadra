from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from quadra.utils.tests.helpers import _random_image


@dataclass
class SegmentationDatasetArguments:
    """Segmentation dataset arguments.

    Args:
        train_samples: List of samples per class in train set, element at index 0 are good samples
        val_samples: List of samples per class in validation set, same as above.
        test_samples: List of samples per class in test set, same as above.
        classes: Optional list of class names, must be equal to len(train_samples) - 1
    """

    train_samples: list[int]
    val_samples: list[int] | None = None
    test_samples: list[int] | None = None
    classes: list[str] | None = None


def _build_segmentation_dataset(
    tmp_path: Path, dataset_arguments: SegmentationDatasetArguments
) -> tuple[str, SegmentationDatasetArguments, dict[str, int]]:
    """Generate segmentation dataset.

    Args:
        tmp_path: path to temporary directory
        dataset_arguments: dataset arguments

    Returns:
        Tuple containing path to dataset, dataset arguments and class to index mapping
    """
    train_samples = dataset_arguments.train_samples
    val_samples = dataset_arguments.val_samples
    test_samples = dataset_arguments.test_samples
    classes = (
        dataset_arguments.classes if dataset_arguments.classes else list(range(1, len(dataset_arguments.train_samples)))
    )

    segmentation_dataset_path = tmp_path / "segmentation_dataset"
    segmentation_dataset_path.mkdir()
    images_path = segmentation_dataset_path / "images"
    masks_path = segmentation_dataset_path / "masks"
    images_path.mkdir(parents=True)
    masks_path.mkdir(parents=True)
    class_to_idx = {class_name: i + 1 for i, class_name in enumerate(classes)}
    classes = [0] + classes

    counter = 0
    for split_name, split_samples in zip(["train", "val", "test"], [train_samples, val_samples, test_samples]):
        if split_samples is None:
            continue

        with open(segmentation_dataset_path / f"{split_name}.txt", "w") as split_file:
            for class_name, samples in zip(classes, split_samples):
                for _ in range(samples):
                    image = _random_image(size=(224, 224))
                    mask = np.zeros((224, 224), dtype=np.uint8)
                    if class_name != 0:
                        mask[100:150, 100:150] = class_to_idx[class_name]
                    image_path = images_path / f"{class_name}_{counter}.png"
                    mask_path = masks_path / f"{class_name}_{counter}.png"
                    cv2.imwrite(str(image_path), image)
                    cv2.imwrite(str(mask_path), mask)
                    split_file.write(f"images/{image_path.name}\n")
                    counter += 1

    return str(segmentation_dataset_path), dataset_arguments, class_to_idx


@pytest.fixture
def segmentation_dataset(
    tmp_path: Path, dataset_arguments: SegmentationDatasetArguments
) -> tuple[str, SegmentationDatasetArguments, dict[str, int]]:
    """Fixture to dinamically generate a segmentation dataset. By default generated images are 224x224 pixels
        and associated masks contains a 50x50 pixels square with the corresponding image class, so at the current stage
        is not possible to have images with multiple annotations. Split files are saved as train.txt,
        val.txt and test.txt.

    Args:
        tmp_path: path to temporary directory
        dataset_arguments: dataset arguments

    Yields:
        Tuple containing path to dataset, dataset arguments and class to index mapping
    """
    yield _build_segmentation_dataset(tmp_path, dataset_arguments)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


@pytest.fixture(
    params=[
        SegmentationDatasetArguments(
            **{"train_samples": [3, 2], "val_samples": [2, 2], "test_samples": [1, 1], "classes": ["bad"]}
        )
    ]
)
def base_binary_segmentation_dataset(
    tmp_path: Path, request: Any
) -> tuple[str, SegmentationDatasetArguments, dict[str, int]]:
    """Generate a base binary segmentation dataset with the following structure:
        - 3 good and 2 bad samples in train set
        - 2 good and 2 bad samples in validation set
        - 11 good and 1 bad sample in test set
        - 2 classes: good and bad.

    Args:
        tmp_path: path to temporary directory
        request: pytest request

    Yields:
        Tuple containing path to dataset, dataset arguments and class to index mapping
    """
    yield _build_segmentation_dataset(tmp_path, request.param)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


@pytest.fixture(
    params=[
        SegmentationDatasetArguments(
            **{
                "train_samples": [2, 2, 2],
                "val_samples": [2, 2, 2],
                "test_samples": [1, 1, 1],
                "classes": ["defect_1", "defect_2"],
            }
        )
    ]
)
def base_multiclass_segmentation_dataset(
    tmp_path: Path, request: Any
) -> tuple[str, SegmentationDatasetArguments, dict[str, int]]:
    """Generate a base binary segmentation dataset with the following structure:
        - 2 good, 2 defect_1 and 2 defect_2 samples in train set
        - 2 good, 2 defect_1 and 2 defect_2 samples in validation set
        - 1 good, 1 defect_1 and 1 defect_2 sample in test set
        - 3 classes: good, defect_1 and defect_2.

    Args:
        tmp_path: path to temporary directory
        request: pytest request

    Yields:
        Tuple containing path to dataset, dataset arguments and class to index mapping
    """
    yield _build_segmentation_dataset(tmp_path, request.param)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
