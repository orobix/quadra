from __future__ import annotations

import os
import pathlib

import albumentations
import pandas as pd
from torch.utils.data import DataLoader

from quadra.datamodules.base import BaseDataModule
from quadra.datasets import AnomalyDataset
from quadra.datasets.anomaly import make_anomaly_dataset
from quadra.utils import utils

log = utils.get_logger(__name__)


class AnomalyDataModule(BaseDataModule):
    """Anomalib-like Lightning Data Module.

    Args:
        data_path: Path to the dataset
        category: Name of the sub category to use.
        image_size: Variable to which image is resized.
        train_batch_size: Training batch size.
        test_batch_size: Testing batch size.
        train_transform: transformations for training. Defaults to None.
        val_transform: transformations for validation. Defaults to None.
        test_transform: transformations for testing. Defaults to None.
        num_workers: Number of workers.
        seed: seed used for the random subset splitting
        task: Whether we are interested in segmenting the anomalies (segmentation) or not (classification)
        mask_suffix: String to append to the base filename to get the mask name, by default for MVTec dataset masks
            are saved as imagename_mask.png in this case the parameter should be filled with "_mask"
        create_test_set_if_empty: If True, the test set is created from good images if it is empty.
        phase: Either train or test.
        name: Name of the data module.
        valid_area_mask: Optional path to the mask to use to filter out the valid area of the image. If None, the whole
            image is considered valid. The mask should match the image size even if the image is cropped.
        crop_area: Optional tuple of 4 integers (x1, y1, x2, y2) to crop the image to the specified area. If None, the
            whole image is considered valid.
    """

    def __init__(
        self,
        data_path: str,
        category: str | None = None,
        image_size: int | tuple[int, int] | None = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        train_transform: albumentations.Compose | None = None,
        val_transform: albumentations.Compose | None = None,
        test_transform: albumentations.Compose | None = None,
        seed: int = 0,
        task: str = "segmentation",
        mask_suffix: str | None = None,
        create_test_set_if_empty: bool = True,
        phase: str = "train",
        name: str = "anomaly_datamodule",
        valid_area_mask: str | None = None,
        crop_area: tuple[int, int, int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            data_path=data_path,
            name=name,
            seed=seed,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            num_workers=num_workers,
            **kwargs,
        )

        self.root = data_path
        self.category = category
        self.data_path = os.path.join(self.root, self.category) if self.category is not None else self.root
        self.image_size = image_size

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.task = task

        self.train_dataset: AnomalyDataset
        self.test_dataset: AnomalyDataset
        self.val_dataset: AnomalyDataset
        self.mask_suffix = mask_suffix
        self.create_test_set_if_empty = create_test_set_if_empty
        self.phase = phase
        self.valid_area_mask = valid_area_mask
        self.crop_area = crop_area

    @property
    def val_data(self) -> pd.DataFrame:
        """Get validation data."""
        _val_data = super().val_data
        if len(_val_data) == 0:
            return self.test_data
        return _val_data

    def _prepare_data(self) -> None:
        """Prepare data for training and testing."""
        self.data = make_anomaly_dataset(
            path=pathlib.Path(self.data_path),
            split=None,
            seed=self.seed,
            mask_suffix=self.mask_suffix,
            create_test_set_if_empty=self.create_test_set_if_empty,
        )

    def setup(self, stage: str | None = None) -> None:
        """Setup data module based on stages of training."""
        if stage == "fit" and self.phase == "train":
            self.train_dataset = AnomalyDataset(
                transform=self.train_transform,
                task=self.task,
                samples=self.train_data,
                valid_area_mask=self.valid_area_mask,
                crop_area=self.crop_area,
            )

            if len(self.val_data) == 0:
                log.info("Validation dataset is empty, using test set instead")

            self.val_dataset = AnomalyDataset(
                transform=self.test_transform,
                task=self.task,
                samples=self.val_data if len(self.val_data) > 0 else self.data,
                valid_area_mask=self.valid_area_mask,
                crop_area=self.crop_area,
            )
        if stage == "test" or self.phase == "test":
            self.test_dataset = AnomalyDataset(
                transform=self.test_transform,
                task=self.task,
                samples=self.test_data,
                valid_area_mask=self.valid_area_mask,
                crop_area=self.crop_area,
            )

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns a dataloader used for predictions."""
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
