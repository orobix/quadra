from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pytest

from quadra.utils.tests.helpers import _random_image


@dataclass
class AnomalyDatasetArguments:
    """Anomaly dataset arguments.

    Args:
        train_samples: number of train samples
        val_samples: number of validation samples (good, bad)
        test_samples: number of test samples (good, bad)
    """

    train_samples: int
    val_samples: tuple[int, int]
    test_samples: tuple[int, int]


def _build_anomaly_dataset(
    tmp_path: Path, dataset_arguments: AnomalyDatasetArguments
) -> tuple[str, AnomalyDatasetArguments]:
    """Generate anomaly dataset in the standard mvtec format.

    Args:
        tmp_path: path to temporary directory
        dataset_arguments: dataset arguments

    Returns:
        path to anomaly dataset
    """
    train_samples = dataset_arguments.train_samples
    val_samples = dataset_arguments.val_samples
    test_samples = dataset_arguments.test_samples

    anomaly_dataset_path = tmp_path / "anomaly_dataset"
    anomaly_dataset_path.mkdir()
    train_good_path = anomaly_dataset_path / "train" / "good"
    val_good_path = anomaly_dataset_path / "val" / "good"
    val_bad_path = anomaly_dataset_path / "val" / "bad"
    test_good_path = anomaly_dataset_path / "test" / "good"
    test_bad_path = anomaly_dataset_path / "test" / "bad"

    train_good_path.mkdir(parents=True)
    val_good_path.mkdir(parents=True)
    val_bad_path.mkdir(parents=True)
    test_good_path.mkdir(parents=True)
    test_bad_path.mkdir(parents=True)

    # Generate train good images
    for i in range(train_samples):
        image = _random_image()
        image_path = train_good_path / f"train_{i}.png"
        cv2.imwrite(str(image_path), image)

    # Generate val good images
    for i in range(val_samples[0]):
        image = _random_image()
        image_path = val_good_path / f"val_{i}.png"
        cv2.imwrite(str(image_path), image)
    # Generate val bad images
    for i in range(val_samples[1]):
        image = _random_image()
        image_path = val_bad_path / f"val_{i}.png"
        cv2.imwrite(str(image_path), image)

    # Generate test good images
    for i in range(test_samples[0]):
        image = _random_image()
        image_path = test_good_path / f"test_{i}.png"
        cv2.imwrite(str(image_path), image)
    # Generate test bad images
    for i in range(test_samples[1]):
        image = _random_image()
        image_path = test_bad_path / f"test_{i}.png"
        cv2.imwrite(str(image_path), image)

    return str(anomaly_dataset_path), dataset_arguments


@pytest.fixture
def anomaly_dataset(tmp_path: Path, dataset_arguments: AnomalyDatasetArguments) -> tuple[str, AnomalyDatasetArguments]:
    """Fixture used to dinamically generate anomaly dataset. By default images are random grayscales with size 10x10.

    Args:
        tmp_path: path to temporary directory
        dataset_arguments: dataset arguments

    Returns:
        path to anomaly dataset
    """
    yield _build_anomaly_dataset(tmp_path, dataset_arguments)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


@pytest.fixture(
    params=[AnomalyDatasetArguments(**{"train_samples": 10, "val_samples": (1, 1), "test_samples": (1, 1)})]
)
def base_anomaly_dataset(tmp_path: Path, request: Any) -> tuple[str, AnomalyDatasetArguments]:
    """Generate base anomaly dataset with the following parameters:
        - train_samples: 10
        - val_samples: (10, 10)
        - test_samples: (10, 10).

    Args:
        tmp_path: Path to temporary directory
        request: Pytest SubRequest object

    Yields:
        Path to anomaly dataset and dataset arguments
    """
    yield _build_anomaly_dataset(tmp_path, request.param)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
