from __future__ import annotations

import warnings
from collections.abc import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from quadra.utils.imaging import crop_image, keep_aspect_ratio_resize


class ImageClassificationListDataset(Dataset):
    """Standard classification dataset.

    Args:
        samples: List of paths to images to be read
        targets: List of labels, one for every image
            in samples
        class_to_idx: mapping from classes
            to unique indexes.
            Defaults to None.
        resize: Integer specifying the size of
            a first optional resize keeping the aspect ratio: the smaller side
            of the image will be resized to `resize`, while the longer will be
            resized keeping the aspect ratio.
            Defaults to None.
        roi: Optional ROI, with
            (x_upper_left, y_upper_left, x_bottom_right, y_bottom_right).
            Defaults to None.
        transform: Optional Albumentations
            transform.
            Defaults to None.
        rgb: if False, image will be converted in grayscale
        channel: 1 or 3. If rgb is True, then channel will be set at 3.
        allow_missing_label: If set to false warn the user if the dataset contains missing labels
    """

    def __init__(
        self,
        samples: list[str],
        targets: list[str | int],
        class_to_idx: dict | None = None,
        resize: int | None = None,
        roi: tuple[int, int, int, int] | None = None,
        transform: Callable | None = None,
        rgb: bool = True,
        channel: int = 3,
        allow_missing_label: bool | None = False,
    ):
        super().__init__()
        assert len(samples) == len(targets), (
            f"Samples ({len(samples)}) and targets ({len(targets)}) must have the same length"
        )
        # Setting the ROI
        self.roi = roi

        # Keep-Aspect-Ratio resize
        self.resize = resize

        if not allow_missing_label and None in targets:
            warnings.warn(
                (
                    "Dataset contains empty targets but allow_missing_label is set to False, "
                    "be careful because None labels will not work inside Dataloaders"
                ),
                UserWarning,
                stacklevel=2,
            )

        targets = [-1 if target is None else target for target in targets]
        # Data
        self.x = np.array(samples)
        self.y = np.array(targets)

        if class_to_idx is None:
            unique_targets = np.unique(targets)
            class_to_idx = {c: i for i, c in enumerate(unique_targets)}

        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.samples = [
            (path, self.class_to_idx[self.y[i]] if (self.y[i] != -1 and self.y[i] != "-1") else -1)
            for i, path in enumerate(self.x)
        ]

        self.rgb = rgb
        self.channel = 3 if rgb else channel

        self.transform = transform

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        path, y = self.samples[idx]

        # Load image
        x = cv2.imread(str(path))
        if self.rgb:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        else:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        if self.channel == 1:
            x = x[:, :, 0]

        # Crop with ROI
        if self.roi:
            x = crop_image(x, self.roi)

        # Resize keeping aspect ratio
        if self.resize:
            x = keep_aspect_ratio_resize(x, self.resize)

        if self.transform:
            aug = self.transform(image=x)
            x = aug["image"]

        return x, y

    def __len__(self):
        return len(self.samples)


class ClassificationDataset(ImageClassificationListDataset):
    """Custom Classification Dataset.

    Args:
        samples: List of paths to images
        targets: List of targets
        class_to_idx: Defaults to None.
        resize: Resize image to this size. Defaults to None.
        roi: Region of interest. Defaults to None.
        transform: transform function. Defaults to None.
        rgb: Use RGB space
        channel: Number of channels. Defaults to 3.
        random_padding: Random padding. Defaults to False.
        circular_crop: Circular crop. Defaults to False.
    """

    def __init__(
        self,
        samples: list[str],
        targets: list[str | int],
        class_to_idx: dict | None = None,
        resize: int | None = None,
        roi: tuple[int, int, int, int] | None = None,
        transform: Callable | None = None,
        rgb: bool = True,
        channel: int = 3,
        random_padding: bool = False,
        circular_crop: bool = False,
    ):
        super().__init__(samples, targets, class_to_idx, resize, roi, transform, rgb, channel)
        if transform is None:
            self.transform = None

        self.random_padding = random_padding
        self.circular_crop = circular_crop

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        path = str(path)

        # Load image
        x = cv2.imread(path)
        if self.rgb:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        else:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            aug = self.transform(image=x)
            x = aug["image"]

        if self.channel == 1:
            x = x[:1]

        return x, y


class MultilabelClassificationDataset(torch.utils.data.Dataset):
    """Custom MultilabelClassification Dataset.

    Args:
        samples: list of paths to images.
        targets: array of multiple targets per sample. The array must be a one-hot enoding.
            It must have a shape of (n_samples, n_targets).
        class_to_idx: Defaults to None.
        transform: transform function. Defaults to None.
        rgb: Use RGB space
    """

    def __init__(
        self,
        samples: list[str],
        targets: np.ndarray,
        class_to_idx: dict | None = None,
        transform: Callable | None = None,
        rgb: bool = True,
    ):
        super().__init__()
        assert len(samples) == len(targets), (
            f"Samples ({len(samples)}) and targets ({len(targets)}) must have the same length"
        )

        # Data
        self.x = samples
        self.y = targets

        # Class to idx and the other way around
        if class_to_idx is None:
            unique_targets = targets.shape[1]
            class_to_idx = {c: i for i, c in enumerate(range(unique_targets))}
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.samples = list(zip(self.x, self.y, strict=False))
        self.rgb = rgb
        self.transform = transform

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        path = str(path)

        # Load image
        x = cv2.imread(path)
        if self.rgb:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        else:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            aug = self.transform(image=x)
            x = aug["image"]

        return x, torch.from_numpy(y).float()

    def __len__(self):
        return len(self.samples)
