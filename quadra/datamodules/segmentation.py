# pylint: disable=unsubscriptable-object,unsupported-assignment-operation,unsupported-membership-test
from __future__ import annotations

import glob
import os
import random
from typing import Any

import albumentations
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import DataLoader

from quadra.datamodules.base import BaseDataModule
from quadra.datasets.segmentation import SegmentationDataset, SegmentationDatasetMulticlass
from quadra.utils import utils

log = utils.get_logger(__name__)


class SegmentationDataModule(BaseDataModule):
    """Base class for segmentation datasets.

    Args:
        data_path: Path to the data main folder.
        name: The name for the data module. Defaults to "segmentation_datamodule".
        val_size: The validation split. Defaults to 0.2.
        test_size: The test split. Defaults to 0.2.
        seed: Random generator seed. Defaults to 42.
        dataset: Dataset class.
        batch_size: Batch size. Defaults to 32.
        num_workers: Number of workers for dataloaders. Defaults to 16.
        train_transform: Transformations for train dataset.
            Defaults to None.
        val_transform: Transformations for validation dataset.
            Defaults to None.
        test_transform: Transformations for test dataset.
            Defaults to None.
        num_data_class: The number of samples per class. Defaults to None.
        exclude_good: If True, exclude good samples from the dataset. Defaults to False.
    """

    def __init__(
        self,
        data_path: str,
        name: str = "segmentation_datamodule",
        test_size: float = 0.3,
        val_size: float = 0.3,
        seed: int = 42,
        dataset: type[SegmentationDataset] = SegmentationDataset,
        batch_size: int = 32,
        num_workers: int = 6,
        train_transform: albumentations.Compose | None = None,
        test_transform: albumentations.Compose | None = None,
        val_transform: albumentations.Compose | None = None,
        train_split_file: str | None = None,
        test_split_file: str | None = None,
        val_split_file: str | None = None,
        num_data_class: int | None = None,
        exclude_good: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            data_path=data_path,
            name=name,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            **kwargs,
        )
        self.test_size = test_size
        self.val_size = val_size
        self.num_data_class = num_data_class
        self.exclude_good = exclude_good
        self.train_split_file = train_split_file
        self.test_split_file = test_split_file
        self.val_split_file = val_split_file
        self.dataset = dataset
        self.train_dataset: SegmentationDataset
        self.val_dataset: SegmentationDataset
        self.test_dataset: SegmentationDataset

    def _preprocess_mask(self, mask) -> np.ndarray:
        """Binarize mask using 0 as threshold."""
        mask = (mask > 0).astype(np.uint8)
        return mask

    @staticmethod
    def _resolve_label(path: str) -> int:
        """Resolve label from mask.

        Args:
            path: Path to the mask.

        Returns:
            0 if the mask is empty, 1 otherwise
        """
        if cv2.imread(path).sum() == 0:
            return 0

        return 1

    def _read_folder(self, data_path: str) -> tuple[list[str], list[int], list[str]]:
        """Read a folder containing images and masks subfolders.

        Args:
            data_path: Path to the data folder.

        Returns:
            List of paths to the images, associated binary targets and list to paths to the masks.
        """
        samples = []
        targets = []
        masks = []

        for im in glob.glob(os.path.join(data_path, "images", "*")):
            if im[0] == ".":
                continue

            mask_path = glob.glob(os.path.splitext(im.replace("images", "masks"))[0] + ".*")

            if len(mask_path) == 0:
                log.debug("Mask not found: %s", os.path.basename(im))
                continue

            if len(mask_path) > 1:
                raise ValueError(f"Multiple masks found for image: {os.path.basename(im)}, this is not supported")

            target = self._resolve_label(mask_path[0])
            samples.append(im)
            targets.append(target)
            masks.append(mask_path[0])

        return samples, targets, masks

    def _read_split(self, split_file: str) -> tuple[list[str], list[int], list[str]]:
        """Reads split file.

        Args:
            split_file: Path to the split file.

        Returns:
            List of paths to images, List of labels.
        """
        samples, targets, masks = [], [], []
        with open(split_file) as f:
            split = f.read().splitlines()
        for sample in split:
            sample_path = os.path.join(self.data_path, sample)
            mask_path = glob.glob(os.path.splitext(sample_path.replace("images", "masks"))[0] + ".*")

            if len(mask_path) == 0:
                log.debug("Mask not found: %s", os.path.basename(sample_path))
                continue

            if len(mask_path) > 1:
                raise ValueError(
                    f"Multiple masks found for image: {os.path.basename(sample_path)}, this is not supported"
                )

            target = self._resolve_label(mask_path[0])
            samples.append(sample_path)
            targets.append(target)
            masks.append(mask_path[0])

        return samples, targets, masks

    def _prepare_data(self) -> None:
        """Prepare data for training and testing."""
        if not (self.test_split_file and self.train_split_file and self.val_split_file):
            all_samples, all_targets, all_masks = self._read_folder(self.data_path)
            samples_train, samples_test, targets_train, targets_test, masks_train, masks_test = train_test_split(
                all_samples,
                all_targets,
                all_masks,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=all_targets,
            )
        if self.test_split_file:
            samples_test, targets_test, masks_test = self._read_split(self.test_split_file)
            if not self.train_split_file:
                samples_train, targets_train, masks_train = [], [], []
                for sample, target, mask in zip(all_samples, all_targets, all_masks, strict=False):
                    if sample not in samples_test:
                        samples_train.append(sample)
                        targets_train.append(target)
                        masks_train.append(mask)

        if self.train_split_file:
            samples_train, targets_train, masks_train = self._read_split(self.train_split_file)
            if not self.test_split_file:
                samples_test, targets_test, masks_test = [], [], []
                for sample, target, mask in zip(all_samples, all_targets, all_masks, strict=False):
                    if sample not in samples_train:
                        samples_test.append(sample)
                        targets_test.append(target)
                        masks_test.append(mask)

        if self.val_split_file:
            if not self.test_split_file or not self.train_split_file:
                raise ValueError("Validation split file is specified but no train or test split file is specified.")
            samples_val, targets_val, masks_val = self._read_split(self.val_split_file)
        else:
            samples_train, samples_val, targets_train, targets_val, masks_train, masks_val = train_test_split(
                samples_train,
                targets_train,
                masks_train,
                test_size=self.val_size,
                random_state=self.seed,
                stratify=targets_train,
            )

        if self.exclude_good:
            samples_train = list(np.array(samples_train)[np.array(targets_train) != 0])
            masks_train = list(np.array(masks_train)[np.array(targets_train) != 0])
            targets_train = list(np.array(targets_train)[np.array(targets_train) != 0])

        if self.num_data_class is not None:
            samples_train_topick = []
            targets_train_topick = []
            masks_train_topick = []

            for cl in np.unique(targets_train):
                idx = np.where(np.array(targets_train) == cl)[0].tolist()
                random.seed(self.seed)
                random.shuffle(idx)
                to_pick = idx[: self.num_data_class]
                for i in to_pick:
                    samples_train_topick.append(samples_train[i])
                    targets_train_topick.append(cl)
                    masks_train_topick.append(masks_train[i])

            samples_train = samples_train_topick
            targets_train = targets_train_topick
            masks_train = masks_train_topick

        df_list = []
        for split_name, samples, targets, masks in [
            ("train", samples_train, targets_train, masks_train),
            ("val", samples_val, targets_val, masks_val),
            ("test", samples_test, targets_test, masks_test),
        ]:
            df = pd.DataFrame({"samples": samples, "targets": targets, "masks": masks})
            df["split"] = split_name
            df_list.append(df)

        self.data = pd.concat(df_list, axis=0)

    def setup(self, stage=None):
        """Setup data module based on stages of training."""
        if stage in ["fit", "train"]:
            self.train_dataset = self.dataset(
                image_paths=self.data[self.data["split"] == "train"]["samples"].tolist(),
                mask_paths=self.data[self.data["split"] == "train"]["masks"].tolist(),
                mask_preprocess=self._preprocess_mask,
                labels=self.data[self.data["split"] == "train"]["targets"].tolist(),
                object_masks=None,
                transform=self.train_transform,
                batch_size=None,
                defect_transform=None,
                resize=None,
            )
            self.val_dataset = self.dataset(
                image_paths=self.data[self.data["split"] == "val"]["samples"].tolist(),
                mask_paths=self.data[self.data["split"] == "val"]["masks"].tolist(),
                defect_transform=None,
                labels=self.data[self.data["split"] == "val"]["targets"].tolist(),
                object_masks=None,
                batch_size=None,
                mask_preprocess=self._preprocess_mask,
                transform=self.test_transform,
                resize=None,
            )
        elif stage == "test":
            self.test_dataset = self.dataset(
                image_paths=self.data[self.data["split"] == "test"]["samples"].tolist(),
                mask_paths=self.data[self.data["split"] == "test"]["masks"].tolist(),
                labels=self.data[self.data["split"] == "test"]["targets"].tolist(),
                object_masks=None,
                batch_size=None,
                mask_preprocess=self._preprocess_mask,
                transform=self.test_transform,
                resize=None,
            )
        elif stage == "predict":
            pass
        else:
            raise ValueError(f"Unknown stage {stage}")

    def train_dataloader(self) -> DataLoader:
        """Returns the train dataloader.

        Raises:
            ValueError: If train dataset is not initialized.

        Returns:
            Train dataloader.
        """
        if not self.train_dataset_available:
            raise ValueError("Train dataset is not initialized")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader.

        Raises:
            ValueError: If validation dataset is not initialized.

        Returns:
            val dataloader.
        """
        if not self.val_dataset_available:
            raise ValueError("Validation dataset is not initialized")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader.

        Raises:
            ValueError: If test dataset is not initialized.


        Returns:
            test dataloader.
        """
        if not self.test_dataset_available:
            raise ValueError("Test dataset is not initialized")

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        return loader

    def predict_dataloader(self) -> DataLoader:
        """Returns a dataloader used for predictions."""
        return self.test_dataloader()


class SegmentationMulticlassDataModule(BaseDataModule):
    """Base class for segmentation datasets with multiple classes.

    Args:
        data_path : Path to the data main folder.
        idx_to_class: dict with corrispondence btw mask index and classes: {1: class_1, 2: class_2, ..., N: class_N}
            except background class which is 0.
        name : The name for the data module. Defaults to "multiclass_segmentation_datamodule".
        dataset: Dataset class.
        batch_size : Batch size. Defaults to 32.
        val_size : The validation split. Defaults to 0.3.
        test_size : The test split. Defaults to 0.3.
        seed : Random generator seed. Defaults to 42.
        num_workers: Number of workers for dataloaders. Defaults to 6.
        train_transform: Transformations for train dataset.
            Defaults to None.
        val_transform : Transformations for validation dataset.
            Defaults to None.
        test_transform : Transformations for test dataset.
            Defaults to None.
        train_split_file: path to txt file with training samples list
        val_split_file: path to txt file with validation samples list
        test_split_file: path to txt file with test samples list
        exclude_good : If True, exclude good samples from the dataset. Defaults to False.
        num_data_train: number of samples to use in the train split (shuffle the samples and pick the
            first num_data_train)
        one_hot_encoding: if True, the labels are one-hot encoded to N channels, where N is the number of classes.
            If False, masks are single channel that contains values as class indexes. Defaults to True.
    """

    def __init__(
        self,
        data_path: str,
        idx_to_class: dict,
        name: str = "multiclass_segmentation_datamodule",
        dataset: type[SegmentationDatasetMulticlass] = SegmentationDatasetMulticlass,
        batch_size: int = 32,
        test_size: float = 0.3,
        val_size: float = 0.3,
        seed: int = 42,
        num_workers: int = 6,
        train_transform: albumentations.Compose | None = None,
        test_transform: albumentations.Compose | None = None,
        val_transform: albumentations.Compose | None = None,
        train_split_file: str | None = None,
        test_split_file: str | None = None,
        val_split_file: str | None = None,
        exclude_good: bool = False,
        num_data_train: int | None = None,
        one_hot_encoding: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            data_path=data_path,
            name=name,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            **kwargs,
        )
        self.test_size = test_size
        self.val_size = val_size
        self.exclude_good = exclude_good
        self.train_split_file = train_split_file
        self.test_split_file = test_split_file
        self.val_split_file = val_split_file
        self.dataset = dataset
        self.idx_to_class = idx_to_class
        self.num_data_train = num_data_train
        self.one_hot_encoding = one_hot_encoding
        self.train_dataset: SegmentationDataset
        self.val_dataset: SegmentationDataset
        self.test_dataset: SegmentationDataset

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Function to preprocess the mask.

        Args:
            mask: a numpy array of dimension HxW with values in [0] + self.idx_to_class.

        Output:
            a binary numpy array with dims len(self.idx_to_class+1)xHxW
        """
        # For each class we must have a channel
        multilayer_mask = np.zeros((len(self.idx_to_class) + 1, *mask.shape[:2]))
        for idx in self.idx_to_class:
            multilayer_mask[int(idx)] = (mask == int(idx)).astype(np.uint8)

        return multilayer_mask

    def _resolve_label(self, path: str) -> np.ndarray:
        """Return a binary array of 1 + len(self.idx_to_class) with 1 if that class is present in the mask."""
        one_hot = np.zeros([len(self.idx_to_class) + 1], np.uint8)  # add class 0
        mask = cv2.imread(path, 0)
        if mask.sum() == 0:
            one_hot[0] = 1
        else:
            indices = np.unique(mask)
            one_hot[indices] = 1
            one_hot[0] = 0

        return one_hot

    def _read_folder(self, data_path: str) -> tuple[list[str], list[np.ndarray], list[str]]:
        """Read a folder containing images and masks subfolders.

        Args:
            data_path: Path to the data folder.

        Returns:
            List of paths to the images, list of associated one-hot encoded targets and list of mask paths.
        """
        samples = []
        targets = []
        masks = []

        for im in glob.glob(os.path.join(data_path, "images", "*")):
            if im[0] == ".":
                continue

            mask_path = glob.glob(os.path.splitext(im.replace("images", "masks"))[0] + ".*")

            if len(mask_path) == 0:
                log.debug("Mask not found: %s", os.path.basename(im))
                continue

            if len(mask_path) > 1:
                raise ValueError(f"Multiple masks found for image: {os.path.basename(im)}, this is not supported")

            target = self._resolve_label(mask_path[0])
            samples.append(im)
            targets.append(target)
            masks.append(mask_path[0])

        return samples, targets, masks

    def _read_split(self, split_file: str) -> tuple[list[str], list[np.ndarray], list[str]]:
        """Reads split file.

        Args:
            split_file: Path to the split file.

        Returns:
            List of paths to images, labels and mask paths.
        """
        samples, targets, masks = [], [], []
        with open(split_file) as f:
            split = f.read().splitlines()
        for sample in split:
            sample_path = os.path.join(self.data_path, sample)
            mask_path = glob.glob(os.path.splitext(sample_path.replace("images", "masks"))[0] + ".*")

            if len(mask_path) == 0:
                log.debug("Mask not found: %s", os.path.basename(sample_path))
                continue

            if len(mask_path) > 1:
                raise ValueError(
                    f"Multiple masks found for image: {os.path.basename(sample_path)}, this is not supported"
                )

            target = self._resolve_label(mask_path[0])
            samples.append(sample_path)
            targets.append(target)
            masks.append(mask_path[0])

        return samples, targets, masks

    def _prepare_data(self) -> None:
        """Prepare data for training and testing."""
        if not (self.train_split_file and self.test_split_file and self.val_split_file):
            all_samples, all_targets, all_masks = self._read_folder(self.data_path)

            (
                samples_and_masks_train,
                targets_train,
                samples_and_masks_test,
                targets_test,
            ) = iterative_train_test_split(
                np.expand_dims(np.array(list(zip(all_samples, all_masks, strict=False))), 1),
                np.array(all_targets),
                test_size=self.test_size,
            )

            samples_train, samples_test = samples_and_masks_train[:, 0, 0], samples_and_masks_test[:, 0, 0]
            masks_train, masks_test = samples_and_masks_train[:, 0, 1], samples_and_masks_test[:, 0, 1]

        if self.test_split_file:
            samples_test, targets_test, masks_test = self._read_split(self.test_split_file)
            if not self.train_split_file:
                samples_train, targets_train, masks_train = [], [], []
                for sample, target, mask in zip(all_samples, all_targets, all_masks, strict=False):
                    if sample not in samples_test:
                        samples_train.append(sample)
                        targets_train.append(target)
                        masks_train.append(mask)

        if self.train_split_file:
            samples_train, targets_train, masks_train = self._read_split(self.train_split_file)
            if not self.test_split_file:
                samples_test, targets_test, masks_test = [], [], []
                for sample, target, mask in zip(all_samples, all_targets, all_masks, strict=False):
                    if sample not in samples_train:
                        samples_test.append(sample)
                        targets_test.append(target)
                        masks_test.append(mask)

        if self.val_split_file:
            samples_val, targets_val, masks_val = self._read_split(self.val_split_file)
            if not self.test_split_file or not self.train_split_file:
                raise ValueError("Validation split file is specified but no train or test split file is specified.")
        else:
            samples_and_masks_train, targets_train, samples_and_masks_val, targets_val = iterative_train_test_split(
                np.expand_dims(np.array(list(zip(samples_train, masks_train, strict=False))), 1),
                np.array(targets_train),
                test_size=self.val_size,
            )
            samples_train = samples_and_masks_train[:, 0, 0]
            samples_val = samples_and_masks_val[:, 0, 0]
            masks_train = samples_and_masks_train[:, 0, 1]
            masks_val = samples_and_masks_val[:, 0, 1]

        # Pre-ordering train and val samples for determinism
        # They will be shuffled (with a seed) during training
        sorting_indices_train = np.argsort(list(samples_train))
        samples_train = [samples_train[i] for i in sorting_indices_train]
        targets_train = [targets_train[i] for i in sorting_indices_train]
        masks_train = [masks_train[i] for i in sorting_indices_train]

        sorting_indices_val = np.argsort(samples_val)
        samples_val = [samples_val[i] for i in sorting_indices_val]
        targets_val = [targets_val[i] for i in sorting_indices_val]
        masks_val = [masks_val[i] for i in sorting_indices_val]

        if self.exclude_good:
            samples_train = list(np.array(samples_train)[np.array(targets_train)[:, 0] == 0])
            masks_train = list(np.array(masks_train)[np.array(targets_train)[:, 0] == 0])
            targets_train = list(np.array(targets_train)[np.array(targets_train)[:, 0] == 0])

        if self.num_data_train is not None:
            # Generate a random permutation
            random_permutation = list(range(len(samples_train)))
            random.seed(self.seed)
            random.shuffle(random_permutation)

            # Shuffle samples_train, targets_train, and masks_train using the same permutation
            samples_train = [samples_train[i] for i in random_permutation]
            targets_train = [targets_train[i] for i in random_permutation]
            masks_train = [masks_train[i] for i in random_permutation]

            samples_train = np.array(samples_train)[: self.num_data_train]
            targets_train = np.array(targets_train)[: self.num_data_train]
            masks_train = np.array(masks_train)[: self.num_data_train]

        df_list = []
        for split_name, samples, targets, masks in [
            ("train", samples_train, targets_train, masks_train),
            ("val", samples_val, targets_val, masks_val),
            ("test", samples_test, targets_test, masks_test),
        ]:
            df = pd.DataFrame({"samples": samples, "targets": list(targets), "masks": masks})
            df["split"] = split_name
            df_list.append(df)

        self.data = pd.concat(df_list, axis=0)

    def setup(self, stage=None):
        """Setup data module based on stages of training."""
        if stage in ["fit", "train"]:
            train_data = self.data[self.data["split"] == "train"]
            val_data = self.data[self.data["split"] == "val"]

            self.train_dataset = self.dataset(
                image_paths=train_data["samples"].tolist(),
                mask_paths=train_data["masks"].tolist(),
                idx_to_class=self.idx_to_class,
                transform=self.train_transform,
                one_hot=self.one_hot_encoding,
            )
            self.val_dataset = self.dataset(
                image_paths=val_data["samples"].tolist(),
                mask_paths=val_data["masks"].tolist(),
                transform=self.val_transform,
                idx_to_class=self.idx_to_class,
                one_hot=self.one_hot_encoding,
            )
        elif stage == "test":
            self.test_dataset = self.dataset(
                image_paths=self.data[self.data["split"] == "test"]["samples"].tolist(),
                mask_paths=self.data[self.data["split"] == "test"]["masks"].tolist(),
                transform=self.test_transform,
                idx_to_class=self.idx_to_class,
                one_hot=self.one_hot_encoding,
            )
        elif stage == "predict":
            pass
        else:
            raise ValueError(f"Unknown stage {stage}")

    def train_dataloader(self) -> DataLoader:
        """Returns the train dataloader.

        Raises:
            ValueError: If train dataset is not initialized.

        Returns:
            Train dataloader.
        """
        if not self.train_dataset_available:
            raise ValueError("Train dataset is not initialized")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader.

        Raises:
            ValueError: If validation dataset is not initialized.

        Returns:
            val dataloader.
        """
        if not self.val_dataset_available:
            raise ValueError("Validation dataset is not initialized")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader.

        Raises:
            ValueError: If test dataset is not initialized.


        Returns:
            test dataloader.
        """
        if not self.test_dataset_available:
            raise ValueError("Test dataset is not initialized")

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        return loader

    def predict_dataloader(self) -> DataLoader:
        """Returns a dataloader used for predictions."""
        return self.test_dataloader()
