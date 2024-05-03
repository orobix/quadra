from __future__ import annotations

import glob
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from quadra.utils.patch import generate_patch_dataset, get_image_mask_association
from quadra.utils.tests.helpers import _random_image


@dataclass
class ClassificationDatasetArguments:
    """Classification dataset arguments.

    Args:
        samples: number of samples per class
        classes: class names, if set it must be the same length as samples
        val_size: validation set size
        test_size: test set size
    """

    samples: list[int]
    classes: list[str] | None = None
    val_size: float | None = None
    test_size: float | None = None


@dataclass
class ClassificationMultilabelDatasetArguments:
    """Classification dataset arguments.

    Args:
        samples: number of samples per class
        classes: class names, if set it must be the same length as samples
        val_size: validation set size
        test_size: test set size
        percentage_other_classes: probability of adding other classes to the labels of each sample
    """

    samples: list[int]
    classes: list[str] | None = None
    val_size: float | None = None
    test_size: float | None = None
    percentage_other_classes: float | None = 0.0


@dataclass
class ClassificationPatchDatasetArguments:
    """Classification patch dataset arguments.

    Args:
        samples: number of samples per class
        overlap: overlap between patches
        patch_size: patch size
        patch_number: number of patches
        classes: class names, if set it must be the same length as samples
        val_size: validation set size
        test_size: test set size
        annotated_good: list of class names that are considered as good annotations (E.g. ["good"])
    """

    samples: list[int]
    overlap: float
    patch_size: tuple[int, int] | None = None
    patch_number: tuple[int, int] | None = None
    classes: list[str] | None = None
    val_size: float | None = 0.0
    test_size: float | None = 0.0
    annotated_good: list[str] | None = None


def _build_classification_dataset(
    tmp_path: Path, dataset_arguments: ClassificationDatasetArguments
) -> tuple[str, ClassificationDatasetArguments]:
    """Generate classification dataset. If val_size or test_size are set, it will generate a train.txt, val.txt and
        test.txt file in the dataset directory. By default generated images are 10x10 pixels.

    Args:
        tmp_path: path to temporary directory
        dataset_arguments: dataset arguments

    Returns:
        Tuple containing path to created dataset and dataset arguments
    """
    classification_dataset_path = tmp_path / "classification_dataset"
    classification_dataset_path.mkdir()

    classes = dataset_arguments.classes if dataset_arguments.classes else range(len(dataset_arguments.samples))

    for class_name, samples in zip(classes, dataset_arguments.samples):
        class_path = classification_dataset_path / str(class_name)
        class_path.mkdir()
        for i in range(samples):
            image = _random_image()
            image_path = class_path / f"{class_name}_{i}.png"
            cv2.imwrite(str(image_path), image)

    if dataset_arguments.val_size is not None or dataset_arguments.test_size is not None:
        all_images = glob.glob(os.path.join(str(classification_dataset_path), "**", "*.png"))
        all_images = [f"{os.path.basename(os.path.dirname(image))}/{os.path.basename(image)}" for image in all_images]
        val_size = dataset_arguments.val_size if dataset_arguments.val_size is not None else 0
        test_size = dataset_arguments.test_size if dataset_arguments.test_size is not None else 0
        train_size = 1 - val_size - test_size

        # pylint: disable=unbalanced-tuple-unpacking
        train_images, val_images, test_images = np.split(
            np.random.permutation(all_images),
            [int(train_size * len(all_images)), int((train_size + val_size) * len(all_images))],
        )

        with open(classification_dataset_path / "train.txt", "w") as f:
            f.write("\n".join(train_images))

        with open(classification_dataset_path / "val.txt", "w") as f:
            f.write("\n".join(val_images))

        with open(classification_dataset_path / "test.txt", "w") as f:
            f.write("\n".join(test_images))

    return str(classification_dataset_path), dataset_arguments


@pytest.fixture
def classification_dataset(
    tmp_path: Path, dataset_arguments: ClassificationDatasetArguments
) -> tuple[str, ClassificationDatasetArguments]:
    """Generate classification dataset. If val_size or test_size are set, it will generate a train.txt, val.txt and
        test.txt file in the dataset directory. By default generated images are 10x10 pixels.

    Args:
        tmp_path: path to temporary directory
        dataset_arguments: dataset arguments

    Yields:
        Tuple containing path to created dataset and dataset arguments
    """
    yield _build_classification_dataset(tmp_path, dataset_arguments)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


@pytest.fixture(
    params=[
        ClassificationDatasetArguments(
            **{"samples": [10, 10], "classes": ["class_1", "class_2"], "val_size": 0.1, "test_size": 0.1}
        )
    ]
)
def base_classification_dataset(tmp_path: Path, request: Any) -> tuple[str, ClassificationDatasetArguments]:
    """Generate base classification dataset with the following parameters:
        - 10 samples per class
        - 2 classes (class_1 and class_2)
        By default generated images are grayscale and 10x10 pixels.

    Args:
        tmp_path: path to temporary directory
        request: pytest request

    Yields:
        Tuple containing path to created dataset and dataset arguments
    """
    yield _build_classification_dataset(tmp_path, request.param)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


def _build_multilabel_classification_dataset(
    tmp_path: Path, dataset_arguments: ClassificationMultilabelDatasetArguments
) -> tuple[str, ClassificationMultilabelDatasetArguments]:
    """Generate a multilabel classification dataset.
        Generates a samples.txt file in the dataset directory containing the path to the image and the corresponding
        classes. If val_size or test_size are set, it will generate a train.txt, val.txt and test.txt file in the
        dataset directory. By default generated images are 10x10 pixels.

    Args:
        tmp_path: path to temporary directory
        dataset_arguments: dataset arguments

    Returns:
        Tuple containing path to created dataset and dataset arguments
    """
    classification_dataset_path = tmp_path / "multilabel_classification_dataset"
    images_path = classification_dataset_path / "images"
    classification_dataset_path.mkdir()
    images_path.mkdir()

    classes = dataset_arguments.classes if dataset_arguments.classes else range(len(dataset_arguments.samples))
    percentage_other_classes = dataset_arguments.percentage_other_classes

    generated_samples = []
    counter = 0
    for class_name, samples in zip(classes, dataset_arguments.samples):
        for _ in range(samples):
            image = _random_image()
            image_path = images_path / f"{counter}.png"
            counter += 1
            cv2.imwrite(str(image_path), image)
            targets = [class_name]
            targets = targets + [
                cl_name for cl_name in classes if cl_name != class_name and random.random() < percentage_other_classes
            ]
            generated_samples.append(f"images/{image_path.name},{','.join(targets)}")

    with open(classification_dataset_path / "samples.txt", "w") as f:
        f.write("\n".join(generated_samples))

    if dataset_arguments.val_size is not None or dataset_arguments.test_size is not None:
        val_size = dataset_arguments.val_size if dataset_arguments.val_size is not None else 0
        test_size = dataset_arguments.test_size if dataset_arguments.test_size is not None else 0
        train_size = 1 - val_size - test_size

        # pylint: disable=unbalanced-tuple-unpacking
        train_images, val_images, test_images = np.split(
            np.random.permutation(generated_samples),
            [int(train_size * len(generated_samples)), int((train_size + val_size) * len(generated_samples))],
        )

        with open(classification_dataset_path / "train.txt", "w") as f:
            f.write("\n".join(train_images))

        with open(classification_dataset_path / "val.txt", "w") as f:
            f.write("\n".join(val_images))

        with open(classification_dataset_path / "test.txt", "w") as f:
            f.write("\n".join(test_images))

    return str(classification_dataset_path), dataset_arguments


@pytest.fixture
def multilabel_classification_dataset(
    tmp_path: Path, dataset_arguments: ClassificationMultilabelDatasetArguments
) -> tuple[str, ClassificationMultilabelDatasetArguments]:
    """Fixture to dinamically generate a multilabel classification dataset.
        Generates a samples.txt file in the dataset directory containing the path to the image and the corresponding
        classes. If val_size or test_size are set, it will generate a train.txt, val.txt and test.txt file in the
        dataset directory. By default generated images are 10x10 pixels.

    Args:
        tmp_path: path to temporary directory
        dataset_arguments: dataset arguments

    Returns:
        Tuple containing path to created dataset and dataset arguments
    """
    yield _build_multilabel_classification_dataset(tmp_path, dataset_arguments)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


@pytest.fixture(
    params=[
        ClassificationMultilabelDatasetArguments(
            **{
                "samples": [10, 10, 10],
                "classes": ["class_1", "class_2", "class_3"],
                "val_size": 0.1,
                "test_size": 0.1,
                "percentage_other_classes": 0.3,
            }
        )
    ]
)
def base_multilabel_classification_dataset(
    tmp_path: Path, request: Any
) -> tuple[str, ClassificationMultilabelDatasetArguments]:
    """Fixture to generate base multilabel classification dataset with the following parameters:
        - 10 samples per class
        - 3 classes (class_1, class_2 and class_3)
        - 10% of samples in validation set
        - 10% of samples in test set
        - 30% of possibility to add each other class to the sample
        By default generated images are grayscale and 10x10 pixels.

    Args:
        tmp_path: path to temporary directory
        request: pytest request

    Yields:
        Tuple containing path to created dataset and dataset arguments
    """
    yield _build_multilabel_classification_dataset(tmp_path, request.param)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


def _build_classification_patch_dataset(
    tmp_path: Path, dataset_arguments: ClassificationDatasetArguments
) -> tuple[str, ClassificationDatasetArguments, dict[str, int]]:
    """Generate a classification patch dataset. By default generated images are 224x224 pixels
        and associated masks contains a 50x50 pixels square with the corresponding image class, so at the current stage
        is not possible to have images with multiple annotations. The patch dataset will be generated using the standard
        parameters of generate_patch_dataset function.

    Args:
        tmp_path: path to temporary directory
        dataset_arguments: dataset arguments

    Returns:
        Tuple containing path to created dataset, dataset arguments and class to index mapping
    """
    initial_dataset_path = tmp_path / "initial_dataset"
    initial_dataset_path.mkdir()

    images_path = initial_dataset_path / "images"
    masks_path = initial_dataset_path / "masks"
    images_path.mkdir()
    masks_path.mkdir()

    classes = dataset_arguments.classes if dataset_arguments.classes else range(len(dataset_arguments.samples))

    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    for class_name, samples in zip(classes, dataset_arguments.samples):
        for i in range(samples):
            image = _random_image(size=(224, 224))
            mask = np.zeros((224, 224), dtype=np.uint8)
            mask[100:150, 100:150] = class_to_idx[class_name]
            image_path = images_path / f"{class_name}_{i}.png"
            mask_path = masks_path / f"{class_name}_{i}.png"
            cv2.imwrite(str(image_path), image)
            cv2.imwrite(str(mask_path), mask)

    patch_dataset_path = tmp_path / "patch_dataset"
    patch_dataset_path.mkdir()

    data_dictionary = get_image_mask_association(data_folder=str(images_path), mask_folder=str(masks_path))

    _ = generate_patch_dataset(
        data_dictionary=data_dictionary,
        class_to_idx=class_to_idx,
        val_size=dataset_arguments.val_size,
        test_size=dataset_arguments.test_size,
        patch_number=dataset_arguments.patch_number,
        patch_size=dataset_arguments.patch_size,
        overlap=dataset_arguments.overlap,
        output_folder=str(patch_dataset_path),
        annotated_good=dataset_arguments.annotated_good,
    )

    return str(patch_dataset_path), dataset_arguments, class_to_idx


@pytest.fixture
def classification_patch_dataset(
    tmp_path: Path, dataset_arguments: ClassificationDatasetArguments
) -> tuple[str, ClassificationDatasetArguments, dict[str, int]]:
    """Fixture to dinamically generate a classification patch dataset.

        By default generated images are 224x224 pixels
        and associated masks contains a 50x50 pixels square with the corresponding image class, so at the current stage
        is not possible to have images with multiple annotations. The patch dataset will be generated using the standard
        parameters of generate_patch_dataset function.

    Args:
        tmp_path: path to temporary directory
        dataset_arguments: dataset arguments

    Yields:
        Tuple containing path to created dataset, dataset arguments and class to index mapping
    """
    yield _build_classification_patch_dataset(tmp_path, dataset_arguments)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


@pytest.fixture(
    params=[
        ClassificationPatchDatasetArguments(
            **{
                "samples": [5, 5, 5],
                "classes": ["bg", "a", "b"],
                "patch_number": [2, 2],
                "overlap": 0,
                "val_size": 0.1,
                "test_size": 0.1,
            }
        )
    ]
)
def base_patch_classification_dataset(
    tmp_path: Path, request: Any
) -> tuple[str, ClassificationDatasetArguments, dict[str, int]]:
    """Generate a classification patch dataset with the following parameters:
        - 3 classes named bg, a and b
        - 5, 5 and 5 samples for each class
        - 2 horizontal patches and 2 vertical patches
        - 0% overlap
        - 10% validation set
        - 10% test set.

    Args:
        tmp_path: path to temporary directory
        request: pytest SubRequest object
    """
    yield _build_classification_patch_dataset(tmp_path, request.param)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
