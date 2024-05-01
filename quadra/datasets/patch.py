from __future__ import annotations

import os
import random
from collections.abc import Callable

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset

from quadra.utils import utils
from quadra.utils.imaging import keep_aspect_ratio_resize
from quadra.utils.patch.dataset import compute_safe_patch_range, trisample

log = utils.get_logger(__name__)


class PatchSklearnClassificationTrainDataset(Dataset):
    """Dataset used for patch sampling, it expects samples to be paths to h5 files containing all the required
    information for patch sampling from images.

    Args:
        data_path: base path to the dataset
        samples: Paths to h5 files
        targets: Labels associated with each sample
        class_to_idx: Mapping between class and corresponding index
        resize: Whether to perform an aspect ratio resize of the patch before the transformations
        transform: Optional function applied to the image
        rgb: if False, image will be converted in grayscale
        channel: 1 or 3. If rgb is True, then channel will be set at 3.
        balance_classes: if True, the dataset will be balanced by duplicating samples of the minority class
    """

    def __init__(
        self,
        data_path: str,
        samples: list[str],
        targets: list[str | int],
        class_to_idx: dict | None = None,
        resize: int | None = None,
        transform: Callable | None = None,
        rgb: bool = True,
        channel: int = 3,
        balance_classes: bool = False,
    ):
        super().__init__()

        # Keep-Aspect-Ratio resize
        self.resize = resize
        self.data_path = data_path

        if balance_classes:
            samples_array = np.array(samples)
            targets_array = np.array(targets)
            samples_to_use: list[str] = []
            targets_to_use: list[str | int] = []

            cls, counts = np.unique(targets_array, return_counts=True)
            max_count = np.max(counts)
            for cl, count in zip(cls, counts, strict=False):
                idx_to_pick = list(np.where(targets_array == cl)[0])

                if count < max_count:
                    idx_to_pick += random.choices(idx_to_pick, k=max_count - count)

                samples_to_use.extend(samples_array[idx_to_pick])
                targets_to_use.extend(targets_array[idx_to_pick])
        else:
            samples_to_use = samples
            targets_to_use = targets

        # Data
        self.x = np.array(samples_to_use)
        self.y = np.array(targets_to_use)

        if class_to_idx is None:
            unique_targets = np.unique(targets_to_use)
            class_to_idx = {c: i for i, c in enumerate(unique_targets)}

        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        self.samples = [
            (path, self.class_to_idx[self.y[i]] if self.y[i] is not None else None) for i, path in enumerate(self.x)
        ]

        self.rgb = rgb
        self.channel = 3 if rgb else channel

        self.transform = transform

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        path, y = self.samples[idx]

        h5_file = h5py.File(path)
        x = cv2.imread(os.path.join(self.data_path, h5_file["img_path"][()].decode("utf-8")))

        weights = h5_file["triangles_weights"][()]
        patch_size = h5_file["patch_size"][()]

        if weights.shape[0] == 0:  # pylint: disable=no-member
            # If the image is completely good sample a point anywhere
            patch_y = np.random.randint(0, x.shape[0] + 1)
            patch_x = np.random.randint(0, x.shape[1] + 1)
        else:
            random_triangle = np.random.choice(weights.shape[0], p=weights)
            [patch_y, patch_x] = trisample(h5_file["triangles"][random_triangle])

        h5_file.close()

        # If the patch is outside the image reduce the exceeding area by taking more patch from the inner area
        [y_left, y_right] = compute_safe_patch_range(patch_y, patch_size[0], x.shape[0])
        [x_left, x_right] = compute_safe_patch_range(patch_x, patch_size[1], x.shape[1])

        x = x[patch_y - y_left : patch_y + y_right, patch_x - x_left : patch_x + x_right]

        if self.rgb:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        else:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        if self.channel == 1:
            x = x[:, :, 0]

        # Resize keeping aspect ratio
        if self.resize:
            x = keep_aspect_ratio_resize(x, self.resize)

        if self.transform:
            aug = self.transform(image=x)
            x = aug["image"]

        return x, y

    def __len__(self):
        return len(self.samples)
