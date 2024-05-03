# pylint: disable=unsubscriptable-object
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from quadra.datamodules.classification import ClassificationDataModule
from quadra.datasets import TwoAugmentationDataset, TwoSetAugmentationDataset
from quadra.utils import utils

log = utils.get_logger(__name__)


class SSLDataModule(ClassificationDataModule):
    """Base class for all data modules for self supervised learning data modules.

    Args:
        data_path: Path to the data main folder.
        augmentation_dataset: Augmentation dataset
            for training dataset.
        name: The name for the data module. Defaults to  "ssl_datamodule".
        split_validation: Whether to split the validation set if . Defaults to True.
        **kwargs: The keyword arguments for the classification data module. Defaults to None.
    """

    def __init__(
        self,
        data_path: str,
        augmentation_dataset: TwoAugmentationDataset | TwoSetAugmentationDataset,
        name: str = "ssl_datamodule",
        split_validation: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            data_path=data_path,
            name=name,
            **kwargs,
        )
        self.augmentation_dataset = augmentation_dataset
        self.classifier_train_dataset: torch.utils.data.Dataset | None = None
        self.split_validation = split_validation

    def setup(self, stage: str | None = None) -> None:
        """Setup data module based on stages of training."""
        if stage == "fit":
            self.train_dataset = self.dataset(
                samples=self.train_data["samples"].tolist(),
                targets=self.train_data["targets"].tolist(),
                transform=self.train_transform,
            )

            if np.unique(self.train_data["targets"]).shape[0] > 1 and not self.split_validation:
                self.classifier_train_dataset = self.dataset(
                    samples=self.train_data["samples"].tolist(),
                    targets=self.train_data["targets"].tolist(),
                    transform=self.val_transform,
                )
                self.val_dataset = self.dataset(
                    samples=self.val_data["samples"].tolist(),
                    targets=self.val_data["targets"].tolist(),
                    transform=self.val_transform,
                )
            else:
                train_classifier_samples, val_samples, train_classifier_targets, val_targets = train_test_split(
                    self.val_data["samples"],
                    self.val_data["targets"],
                    test_size=0.3,
                    random_state=self.seed,
                    stratify=self.val_data["targets"],
                )

                self.classifier_train_dataset = self.dataset(
                    samples=train_classifier_samples,
                    targets=train_classifier_targets,
                    transform=self.test_transform,
                )

                self.val_dataset = self.dataset(
                    samples=val_samples,
                    targets=val_targets,
                    transform=self.val_transform,
                )

                log.warning(
                    "The training set contains only one class and cannot be used to train a classifier. To overcome "
                    "this issue 70% of the validation set is used to train the classifier. The remaining will be used "
                    "as standard validation. To disable this behaviour set the `split_validation` parameter to False."
                )
                self._check_train_dataset_config()
        if stage == "test":
            self.test_dataset = self.dataset(
                samples=self.test_data["samples"].tolist(),
                targets=self.test_data["targets"].tolist(),
                transform=self.test_transform,
            )

    def _check_train_dataset_config(self):
        """Check if train dataset is configured correctly."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized")
        if self.augmentation_dataset is None:
            raise ValueError("Augmentation dataset is not initialized")
        if self.train_dataset.transform is not None:
            log.warning("Train dataset transform is not None. It will be applied before SSL augmentations")

    def train_dataloader(self) -> DataLoader:
        """Returns train dataloader."""
        if not isinstance(self.train_dataset, torch.utils.data.Dataset):
            raise ValueError("Train dataset is not a subclass of `torch.utils.data.Dataset`")
        self.augmentation_dataset.dataset = self.train_dataset
        loader = DataLoader(
            self.augmentation_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        return loader

    def classifier_train_dataloader(self) -> DataLoader:
        """Returns classifier train dataloader."""
        if self.classifier_train_dataset is None:
            raise ValueError("Classifier train dataset is not initialized")

        loader = DataLoader(
            self.classifier_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        return loader
