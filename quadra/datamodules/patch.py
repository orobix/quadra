from __future__ import annotations

import json
import os

import albumentations
import pandas as pd
from torch.utils.data import DataLoader

from quadra.datamodules.base import BaseDataModule
from quadra.datasets import ImageClassificationListDataset, PatchSklearnClassificationTrainDataset
from quadra.utils.classification import find_test_image
from quadra.utils.patch.dataset import PatchDatasetInfo, load_train_file


class PatchSklearnClassificationDataModule(BaseDataModule):
    """DataModule for patch classification.

    Args:
        data_path: Location of the dataset
        name: Name of the datamodule
        train_filename: Name of the file containing the list of training samples
        exclude_filter: Filter to exclude samples from the dataset
        include_filter: Filter to include samples from the dataset
        class_to_idx: Dictionary mapping class names to indices
        seed: Random seed
        batch_size: Batch size
        num_workers: Number of workers
        train_transform: Transform to apply to the training samples
        val_transform: Transform to apply to the validation samples
        test_transform: Transform to apply to the test samples
        balance_classes: If True repeat low represented classes
        class_to_skip_training: List of classes skipped during training.
    """

    def __init__(
        self,
        data_path: str,
        class_to_idx: dict,
        name: str = "patch_classification_datamodule",
        train_filename: str = "dataset.txt",
        exclude_filter: list[str] | None = None,
        include_filter: list[str] | None = None,
        seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 6,
        train_transform: albumentations.Compose | None = None,
        val_transform: albumentations.Compose | None = None,
        test_transform: albumentations.Compose | None = None,
        balance_classes: bool = False,
        class_to_skip_training: list | None = None,
        **kwargs,
    ):
        super().__init__(
            data_path=data_path,
            name=name,
            seed=seed,
            num_workers=num_workers,
            batch_size=batch_size,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            **kwargs,
        )
        self.class_to_idx = class_to_idx
        self.balance_classes = balance_classes
        self.train_filename = train_filename
        self.include_filter = include_filter
        self.exclude_filter = exclude_filter
        self.class_to_skip_training = class_to_skip_training

        self.train_folder = os.path.join(self.data_path, "train")
        self.val_folder = os.path.join(self.data_path, "val")
        self.test_folder = os.path.join(self.data_path, "test")
        self.info: PatchDatasetInfo
        self.train_dataset: PatchSklearnClassificationTrainDataset
        self.val_dataset: ImageClassificationListDataset
        self.test_dataset: ImageClassificationListDataset

    def _prepare_data(self):
        """Prepare data function."""
        if os.path.isfile(os.path.join(self.data_path, "info.json")):
            with open(os.path.join(self.data_path, "info.json")) as f:
                self.info = PatchDatasetInfo(**json.load(f))
        else:
            raise FileNotFoundError("No `info.json` file found in the dataset folder")

        split_df_list: list[pd.DataFrame] = []
        if os.path.isfile(os.path.join(self.train_folder, self.train_filename)):
            train_samples, train_labels = load_train_file(
                train_file_path=os.path.join(self.train_folder, self.train_filename),
                include_filter=self.include_filter,
                exclude_filter=self.exclude_filter,
                class_to_skip=self.class_to_skip_training,
            )
            train_df = pd.DataFrame({"samples": train_samples, "targets": train_labels})
            train_df["split"] = "train"
            split_df_list.append(train_df)
        if os.path.isdir(self.val_folder):
            val_samples, val_labels = find_test_image(
                folder=self.val_folder,
                exclude_filter=self.exclude_filter,
                include_filter=self.include_filter,
                include_none_class=False,
            )
            val_df = pd.DataFrame({"samples": val_samples, "targets": val_labels})
            val_df["split"] = "val"
            split_df_list.append(val_df)
        if os.path.isdir(self.test_folder):
            test_samples, test_labels = find_test_image(
                folder=self.test_folder,
                exclude_filter=self.exclude_filter,
                include_filter=self.include_filter,
                include_none_class=True,
            )
            test_df = pd.DataFrame({"samples": test_samples, "targets": test_labels})
            test_df["split"] = "test"
            split_df_list.append(test_df)
        if len(split_df_list) == 0:
            raise ValueError("No data found in all split folders")
        self.data = pd.concat(split_df_list, axis=0)

    def setup(self, stage: str | None = None) -> None:
        """Setup function."""
        if stage == "fit":
            self.train_dataset = PatchSklearnClassificationTrainDataset(
                data_path=self.data_path,
                class_to_idx=self.class_to_idx,
                samples=self.data[self.data["split"] == "train"]["samples"].tolist(),
                targets=self.data[self.data["split"] == "train"]["targets"].tolist(),
                transform=self.train_transform,
                balance_classes=self.balance_classes,
            )

            self.val_dataset = ImageClassificationListDataset(
                class_to_idx=self.class_to_idx,
                samples=self.data[self.data["split"] == "val"]["samples"].tolist(),
                targets=self.data[self.data["split"] == "val"]["targets"].tolist(),
                transform=self.val_transform,
                allow_missing_label=False,
            )

        elif stage in ["test", "predict"]:
            self.test_dataset = ImageClassificationListDataset(
                class_to_idx=self.class_to_idx,
                samples=self.data[self.data["split"] == "test"]["samples"].tolist(),
                targets=self.data[self.data["split"] == "test"]["targets"].tolist(),
                transform=self.test_transform,
                allow_missing_label=True,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        if not self.train_dataset_available:
            raise ValueError("No training sample is available")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        if not self.val_dataset_available:
            raise ValueError("No validation dataset is available")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        if not self.test_dataset_available:
            raise ValueError("No test dataset is available")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
