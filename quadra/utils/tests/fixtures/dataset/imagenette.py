import shutil
from pathlib import Path

import cv2
import pytest

from quadra.utils.tests.helpers import _random_image


def _build_imagenette_dataset(tmp_path: Path, classes: int, class_samples: int) -> str:
    """Generate imagenette dataset in the format required by efficient_ad model.

    Args:
        tmp_path: Path to temporary directory
        classes: Number of mock imagenette classes
        class_samples: Number of samples for each mock imagenette class

    Returns:
        Path to imagenette dataset
    """
    parent_path = tmp_path / "imagenette2"
    parent_path.mkdir()
    train_path = parent_path / "train"
    train_path.mkdir()
    val_path = parent_path / "val"
    val_path.mkdir()

    for split in [train_path, val_path]:
        for i in range(classes):
            cl_path = split / f"class_{i}"
            cl_path.mkdir()
            for j in range(class_samples):
                image = _random_image()
                image_path = cl_path / f"fake_{j}.png"
                cv2.imwrite(str(image_path), image)

    return parent_path


@pytest.fixture
def imagenette_dataset(tmp_path: Path) -> str:
    """Generate a mock imagenette dataset to test efficient_ad model.

    Args:
        tmp_path: Path to temporary directory

    Yields:
        Path to imagenette dataset folder
    """
    yield _build_imagenette_dataset(tmp_path, classes=3, class_samples=3)

    if tmp_path.exists():
        shutil.rmtree(tmp_path)
