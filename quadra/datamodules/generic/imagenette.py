from __future__ import annotations

import os
import shutil
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_and_extract_archive

from quadra.datamodules import ClassificationDataModule, SSLDataModule
from quadra.utils.utils import get_logger

IMAGENETTE_LABEL_MAPPER = {
    "n01440764": "tench",
    "n02102040": "english_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}

DEFAULT_CLASS_TO_IDX = {cl: idx for idx, cl in enumerate(sorted(IMAGENETTE_LABEL_MAPPER.values()))}

log = get_logger(__name__)


class ImagenetteClassificationDataModule(ClassificationDataModule):
    """Initializes the classification data module for Imagenette dataset.

    Args:
        data_path: Path to the dataset.
        name: Name of the dataset.
        imagenette_version: Version of the Imagenette dataset. Can be 320 or 160 or full.
        force_download: If True, the dataset will be downloaded even if the data_path already exists. The data_path
            will be deleted and recreated.
        class_to_idx: Dictionary mapping class names to class indices.
        **kwargs: Keyword arguments for the ClassificationDataModule.
    """

    def __init__(
        self,
        data_path: str,
        name: str = "imagenette_classification_datamodule",
        imagenette_version: str = "320",
        force_download: bool = False,
        class_to_idx: dict[str, int] | None = None,
        **kwargs: Any,
    ):
        if imagenette_version not in ["320", "160", "full"]:
            raise ValueError(f"imagenette_version must be one of 320, 160 or full. Got {imagenette_version} instead.")

        if imagenette_version == "full":
            imagenette_version = ""
        else:
            imagenette_version = f"-{imagenette_version}"

        self.download_url = f"https://s3.amazonaws.com/fast-ai-imageclas/imagenette2{imagenette_version}.tgz"
        self.force_download = force_download
        self.imagenette_version = imagenette_version

        if class_to_idx is None:
            class_to_idx = DEFAULT_CLASS_TO_IDX

        super().__init__(
            data_path=data_path,
            name=name,
            test_split_file=None,
            train_split_file=None,
            val_size=None,
            class_to_idx=class_to_idx,
            **kwargs,
        )

    def download_data(self, download_url: str, force_download: bool = False) -> None:
        """Download the Imagenette dataset.

        Args:
            download_url: Dataset download url.
            force_download: If True, the dataset will be downloaded even if the data_path already exists. The data_path
                will be removed.
        """
        if os.path.exists(self.data_path):
            if force_download:
                log.info("The path %s already exists. Removing it and downloading the dataset again.", self.data_path)
                shutil.rmtree(self.data_path)
            else:
                log.info("The path %s already exists. Skipping download.", self.data_path)
                return

        log.info("Downloading and extracting Imagenette dataset to %s", self.data_path)
        download_and_extract_archive(download_url, self.data_path, remove_finished=True)

    def _prepare_data(self) -> None:
        """Prepares the data for the data module."""
        self.download_data(download_url=self.download_url, force_download=self.force_download)
        self.data_path = os.path.join(self.data_path, f"imagenette2{self.imagenette_version}")

        train_images_and_targets, class_to_idx = self._find_images_and_targets(os.path.join(self.data_path, "train"))
        self.class_to_idx = {IMAGENETTE_LABEL_MAPPER[k]: v for k, v in class_to_idx.items()}

        samples_train, targets_train = [], []
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        for image, target in train_images_and_targets:
            samples_train.append(image)
            targets_train.append(idx_to_class[target])

        samples_train, samples_val, targets_train, targets_val = train_test_split(
            samples_train,
            targets_train,
            test_size=self.val_size,
            random_state=self.seed,
            stratify=targets_train,
        )

        test_images_and_targets, _ = self._find_images_and_targets(os.path.join(self.data_path, "val"))
        samples_test, targets_test = [], []
        for image, target in test_images_and_targets:
            samples_test.append(image)
            targets_test.append(idx_to_class[target])

        train_df = pd.DataFrame({"samples": samples_train, "targets": targets_train})
        train_df["split"] = "train"
        val_df = pd.DataFrame({"samples": samples_val, "targets": targets_val})
        val_df["split"] = "val"
        test_df = pd.DataFrame({"samples": samples_test, "targets": targets_test})
        test_df["split"] = "test"
        self.data = pd.concat([train_df, val_df, test_df], axis=0)


class ImagenetteSSLDataModule(ImagenetteClassificationDataModule, SSLDataModule):
    """Initializes the SSL data module for Imagenette dataset."""

    def __init__(
        self,
        *args: Any,
        name="imagenette_ssl",
        **kwargs: Any,
    ):
        super().__init__(*args, name=name, **kwargs)  # type: ignore[misc]
