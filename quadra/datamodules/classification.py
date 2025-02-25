# pylint: disable=unsupported-assignment-operation,unsubscriptable-object
from __future__ import annotations

import os
import random
from collections.abc import Callable
from typing import Any

import albumentations
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from timm.data.readers.reader_image_folder import find_images_and_targets
from torch.utils.data import DataLoader

from quadra.datamodules.base import BaseDataModule
from quadra.datasets import ImageClassificationListDataset
from quadra.datasets.classification import MultilabelClassificationDataset
from quadra.utils import utils
from quadra.utils.classification import find_test_image, get_split, group_labels, natural_key

log = utils.get_logger(__name__)


class ClassificationDataModule(BaseDataModule):
    """Base class single folder based classification datamodules. If there is no nested folders, use this class.

    Args:
        data_path: Path to the data main folder.
        name: The name for the data module. Defaults to "classification_datamodule".
        num_workers: Number of workers for dataloaders. Defaults to 16.
        batch_size: Batch size. Defaults to 32.
        seed: Random generator seed. Defaults to 42.
        dataset: Dataset class.
        val_size: The validation split. Defaults to 0.2.
        test_size: The test split. Defaults to 0.2.
        exclude_filter: The filter for excluding folders. Defaults to None.
        include_filter: The filter for including folders. Defaults to None.
        label_map: The mapping for labels. Defaults to None.
        num_data_class: The number of samples per class. Defaults to None.
        train_transform: Transformations for train dataset.
            Defaults to None.
        val_transform: Transformations for validation dataset.
            Defaults to None.
        test_transform: Transformations for test dataset.
            Defaults to None.
        train_split_file: The file with train split. Defaults to None.
        val_split_file: The file with validation split. Defaults to None.
        test_split_file: The file with test split. Defaults to None.
        class_to_idx: The mapping from class name to index. Defaults to None.
        **kwargs: Additional arguments for BaseDataModule.
    """

    def __init__(
        self,
        data_path: str,
        dataset: type[ImageClassificationListDataset] = ImageClassificationListDataset,
        name: str = "classification_datamodule",
        num_workers: int = 8,
        batch_size: int = 32,
        seed: int = 42,
        val_size: float | None = 0.2,
        test_size: float = 0.2,
        num_data_class: int | None = None,
        exclude_filter: list[str] | None = None,
        include_filter: list[str] | None = None,
        label_map: dict[str, Any] | None = None,
        load_aug_images: bool = False,
        aug_name: str | None = None,
        n_aug_to_take: int | None = 4,
        replace_str_from: str | None = None,
        replace_str_to: str | None = None,
        train_transform: albumentations.Compose | None = None,
        val_transform: albumentations.Compose | None = None,
        test_transform: albumentations.Compose | None = None,
        train_split_file: str | None = None,
        test_split_file: str | None = None,
        val_split_file: str | None = None,
        class_to_idx: dict[str, int] | None = None,
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
            load_aug_images=load_aug_images,
            aug_name=aug_name,
            n_aug_to_take=n_aug_to_take,
            replace_str_from=replace_str_from,
            replace_str_to=replace_str_to,
            **kwargs,
        )
        self.replace_str = None
        self.exclude_filter = exclude_filter
        self.include_filter = include_filter
        self.val_size = val_size
        self.test_size = test_size
        self.label_map = label_map
        self.num_data_class = num_data_class
        self.dataset = dataset
        self.train_split_file = train_split_file
        self.test_split_file = test_split_file
        self.val_split_file = val_split_file
        self.class_to_idx: dict[str, int] | None

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
            self.num_classes = len(self.class_to_idx)
        else:
            self.class_to_idx = self._find_classes_from_data_path(self.data_path)
            if self.class_to_idx is None:
                log.warning("Could not build a class_to_idx from the data_path subdirectories")
                self.num_classes = 0
            else:
                self.num_classes = len(self.class_to_idx)

    def _read_split(self, split_file: str) -> tuple[list[str], list[str]]:
        """Reads split file.

        Args:
            split_file: Path to the split file.

        Returns:
            List of paths to images.
        """
        samples, targets = [], []
        with open(split_file) as f:
            split = f.readlines()
        for row in split:
            csv_values = row.split(",")
            sample = str(",".join(csv_values[:-1])).strip()
            target = csv_values[-1].strip()
            sample_path = os.path.join(self.data_path, sample)
            if os.path.exists(sample_path):
                samples.append(sample_path)
                targets.append(target)
            else:
                continue
                # log.warning(f"{sample_path} does not exist")
        return samples, targets

    def _find_classes_from_data_path(self, data_path: str) -> dict[str, int] | None:
        """Given a data_path, build a random class_to_idx from the subdirectories.

        Args:
            data_path: Path to the data main folder.

        Returns:
            class_to_idx dictionary.
        """
        subdirectories = []

        # Check if the directory exists
        if os.path.exists(data_path) and os.path.isdir(data_path):
            # Iterate through the items in the directory
            for item in os.listdir(data_path):
                item_path = os.path.join(data_path, item)

                # Check if it's a directory and not starting with "."
                if (
                    os.path.isdir(item_path)
                    and not item.startswith(".")
                    # Check if there's at least one image file in the subdirectory
                    and any(
                        os.path.splitext(file)[1].lower().endswith(tuple(utils.IMAGE_EXTENSIONS))
                        for file in os.listdir(item_path)
                    )
                ):
                    subdirectories.append(item)

            if len(subdirectories) > 0:
                return {cl: idx for idx, cl in enumerate(sorted(subdirectories))}
            return None

        return None

    @staticmethod
    def _find_images_and_targets(
        root_folder: str, class_to_idx: dict[str, int] | None = None
    ) -> tuple[list[tuple[str, int]], dict[str, int]]:
        """Collects the samples from item folders."""
        images_and_targets, class_to_idx = find_images_and_targets(
            folder=root_folder, types=utils.IMAGE_EXTENSIONS, class_to_idx=class_to_idx
        )
        return images_and_targets, class_to_idx

    def _filter_images_and_targets(
        self, images_and_targets: list[tuple[str, int]], class_to_idx: dict[str, int]
    ) -> tuple[list[str], list[str]]:
        """Filters the images and targets."""
        samples: list[str] = []
        targets: list[str] = []
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        images_and_targets = [(str(image_path), target) for image_path, target in images_and_targets]
        for image_path, target in images_and_targets:
            target_class = idx_to_class[target]
            if self.exclude_filter is not None and any(
                exclude_filter in image_path for exclude_filter in self.exclude_filter
            ):
                continue
            if self.include_filter is not None:
                if any(include_filter in image_path for include_filter in self.include_filter):
                    samples.append(str(image_path))
                    targets.append(target_class)
            else:
                samples.append(str(image_path))
                targets.append(target_class)
        return (
            samples,
            targets,
        )

    def _prepare_data(self) -> None:
        """Prepares Classification data for the data module."""
        images_and_targets, class_to_idx = self._find_images_and_targets(self.data_path, self.class_to_idx)
        all_samples, all_targets = self._filter_images_and_targets(images_and_targets, class_to_idx)
        if self.label_map is not None:
            all_targets, _ = group_labels(all_targets, self.label_map)

        samples_train: list[str] = []
        targets_train: list[str] = []
        samples_test: list[str] = []
        targets_test: list[str] = []
        samples_val: list[str] = []
        targets_val: list[str] = []

        if self.test_size < 1.0:
            samples_train, samples_test, targets_train, targets_test = train_test_split(
                all_samples,
                all_targets,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=all_targets,
            )
            if self.test_split_file:
                samples_test, targets_test = self._read_split(self.test_split_file)
                if not self.train_split_file:
                    samples_train, targets_train = [], []
                    for sample, target in zip(all_samples, all_targets, strict=False):
                        if sample not in samples_test:
                            samples_train.append(sample)
                            targets_train.append(target)
            if self.train_split_file:
                samples_train, targets_train = self._read_split(self.train_split_file)
                if not self.test_split_file:
                    samples_test, targets_test = [], []
                    for sample, target in zip(all_samples, all_targets, strict=False):
                        if sample not in samples_train:
                            samples_test.append(sample)
                            targets_test.append(target)
            if self.val_split_file:
                samples_val, targets_val = self._read_split(self.val_split_file)
                if not self.test_split_file or not self.train_split_file:
                    raise ValueError("Validation split file is specified but no train or test split file is specified.")
            else:
                samples_train, samples_val, targets_train, targets_val = train_test_split(
                    samples_train,
                    targets_train,
                    test_size=self.val_size,
                    random_state=self.seed,
                    stratify=targets_train,
                )

            if self.num_data_class is not None:
                samples_train_topick = []
                targets_train_topick = []
                for cl in np.unique(targets_train):
                    idx = np.where(np.array(targets_train) == cl)[0]
                    random.seed(self.seed)
                    random.shuffle(idx)  # type: ignore[arg-type]
                    to_pick = idx[: self.num_data_class]
                    for i in to_pick:
                        samples_train_topick.append(samples_train[i])
                        targets_train_topick.append(cl)

                samples_train = samples_train_topick
                targets_train = targets_train_topick
        else:
            log.info("Test size is set to 1.0: all samples will be put in test-set")
            samples_test = all_samples
            targets_test = all_targets
        train_df = pd.DataFrame({"samples": samples_train, "targets": targets_train})
        train_df["split"] = "train"
        val_df = pd.DataFrame({"samples": samples_val, "targets": targets_val})
        val_df["split"] = "val"
        test_df = pd.DataFrame({"samples": samples_test, "targets": targets_test})
        test_df["split"] = "test"
        self.data = pd.concat([train_df, val_df, test_df], axis=0)

        # if self.load_aug_images:
        #    samples_train, targets_train = self.load_augmented_samples(
        #         samples_train, targets_train, self.replace_str, shuffle=True
        #     )
        #     samples_val, targets_val = self.load_augmented_samples(
        #         samples_val, targets_val , self.replace_str, shuffle=True
        #     )
        unique_targets = [str(t) for t in np.unique(targets_train)]
        if self.class_to_idx is None:
            sorted_targets = sorted(unique_targets, key=natural_key)
            class_to_idx = {c: idx for idx, c in enumerate(sorted_targets)}
            self.class_to_idx = class_to_idx
            log.info("Class_to_idx not provided in config, building it from targets: %s", class_to_idx)

        if len(unique_targets) == 0:
            log.warning("Unique_targets length is 0, training set is empty")
        else:
            if len(self.class_to_idx.keys()) != len(unique_targets):
                raise ValueError(
                    "The number of classes in the class_to_idx dictionary does not match the number of unique targets."
                    f" `class_to_idx`: {self.class_to_idx}, `unique_targets`: {unique_targets}"
                )
            if not all(c in unique_targets for c in self.class_to_idx):
                raise ValueError(
                    "The classes in the class_to_idx dictionary do not match the available unique targets in the"
                    " datasset. `class_to_idx`: {self.class_to_idx}, `unique_targets`: {unique_targets}"
                )

    def setup(self, stage: str | None = None) -> None:
        """Setup data module based on stages of training."""
        if stage in ["train", "fit"]:
            self.train_dataset = self.dataset(
                samples=self.data[self.data["split"] == "train"]["samples"].tolist(),
                targets=self.data[self.data["split"] == "train"]["targets"].tolist(),
                transform=self.train_transform,
                class_to_idx=self.class_to_idx,
            )
            self.val_dataset = self.dataset(
                samples=self.data[self.data["split"] == "val"]["samples"].tolist(),
                targets=self.data[self.data["split"] == "val"]["targets"].tolist(),
                transform=self.val_transform,
                class_to_idx=self.class_to_idx,
            )
        if stage in ["test", "predict"]:
            self.test_dataset = self.dataset(
                samples=self.data[self.data["split"] == "test"]["samples"].tolist(),
                targets=self.data[self.data["split"] == "test"]["targets"].tolist(),
                transform=self.test_transform,
                class_to_idx=self.class_to_idx,
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the train dataloader.

        Raises:
            ValueError: If train dataset is not initialized.

        Returns:
            Train dataloader.
        """
        if not self.train_dataset_available:
            raise ValueError("Train dataset is not initialized")
        if not isinstance(self.train_dataset, torch.utils.data.Dataset):
            raise ValueError("Train dataset has to be single `torch.utils.data.Dataset` instance.")
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
        if not isinstance(self.val_dataset, torch.utils.data.Dataset):
            raise ValueError("Validation dataset has to be single `torch.utils.data.Dataset` instance.")
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


class SklearnClassificationDataModule(BaseDataModule):
    """A generic Data Module for classification with frozen torch backbone and sklearn classifier.

    It can also handle k-fold cross validation.

    Args:
        name: The name for the data module. Defaults to "sklearn_classification_datamodule".
        data_path: Path to images main folder
        exclude_filter: List of string filter to be used to exclude images. If None no filter will be applied.
        include_filter: List of string filter to be used to include images. Only images that satisfied at list one of
                        the filter will be included.
        val_size: The validation split. Defaults to 0.2.
        class_to_idx: Dictionary of conversion btw folder name and index. Only file whose label is in dictionary key
            list will be considered. If None all files will be considered and a custom conversion is created.
        seed: Fixed seed for random operations
        batch_size: Dimension of batches for dataloader
        num_workers: Number of workers for dataloader
        train_transform: Albumentation transformations for training set
        val_transform: Albumentation transformations for validation set
        test_transform: Albumentation transformations for test set
        roi: Optional cropping region
        n_splits: Number of dataset subdivision (default 1 -> train/test). Use a value >= 2 for cross validation.
        phase: Either train or test
        cache: If true disable shuffling in all dataloader to enable feature caching
        limit_training_data: if defined, each class will be donwsampled to this number. It must be >= 2 to allow
            splitting
        label_map: Dictionary of conversion btw folder name and label.
        train_split_file: Optional path to a csv file containing the train split samples.
        test_split_file: Optional path to a csv file containing the test split samples.
        **kwargs: Additional arguments for BaseDataModule
    """

    def __init__(
        self,
        data_path: str,
        exclude_filter: list[str] | None = None,
        include_filter: list[str] | None = None,
        val_size: float = 0.2,
        class_to_idx: dict[str, int] | None = None,
        label_map: dict[str, Any] | None = None,
        seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 6,
        train_transform: albumentations.Compose | None = None,
        val_transform: albumentations.Compose | None = None,
        test_transform: albumentations.Compose | None = None,
        roi: tuple[int, int, int, int] | None = None,
        n_splits: int = 1,
        phase: str = "train",
        cache: bool = False,
        limit_training_data: int | None = None,
        train_split_file: str | None = None,
        test_split_file: str | None = None,
        name: str = "sklearn_classification_datamodule",
        dataset: type[ImageClassificationListDataset] = ImageClassificationListDataset,
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

        self.class_to_idx = class_to_idx
        self.roi = roi
        self.cache = cache
        self.limit_training_data = limit_training_data

        self.dataset = dataset
        self.phase = phase
        self.n_splits = n_splits
        self.train_split_file = train_split_file
        self.test_split_file = test_split_file
        self.exclude_filter = exclude_filter
        self.include_filter = include_filter
        self.val_size = val_size
        self.label_map = label_map
        self.full_dataset: ImageClassificationListDataset
        self.train_dataset: list[ImageClassificationListDataset]
        self.val_dataset: list[ImageClassificationListDataset]

    def _prepare_data(self) -> None:
        """Prepares the data for the data module."""
        assert os.path.isdir(self.data_path), f"Folder {self.data_path} does not exist."

        list_df = []
        if self.phase == "train":
            samples, targets, split_generator, self.class_to_idx = get_split(
                image_dir=self.data_path,
                exclude_filter=self.exclude_filter,
                include_filter=self.include_filter,
                test_size=self.val_size,
                random_state=self.seed,
                class_to_idx=self.class_to_idx,
                n_splits=self.n_splits,
                limit_training_data=self.limit_training_data,
                train_split_file=self.train_split_file,
                label_map=self.label_map,
            )

            for cv_idx, split in enumerate(split_generator):
                train_idx, val_idx = split
                train_val_df = pd.DataFrame({"samples": samples, "targets": targets})
                train_val_df["cv"] = 0
                train_val_df["split"] = "train"
                train_val_df.loc[val_idx, "split"] = "val"
                train_val_df.loc[train_idx, "cv"] = cv_idx
                train_val_df.loc[val_idx, "cv"] = cv_idx
                list_df.append(train_val_df)

        test_samples, test_targets = find_test_image(
            folder=self.data_path,
            exclude_filter=self.exclude_filter,
            include_filter=self.include_filter,
            test_split_file=self.test_split_file,
        )
        if self.label_map is not None:
            test_targets, _ = group_labels(test_targets, self.label_map)
        test_df = pd.DataFrame({"samples": test_samples, "targets": test_targets})
        test_df["split"] = "test"
        test_df["cv"] = np.nan

        list_df.append(test_df)
        self.data = pd.concat(list_df, axis=0)

    def setup(self, stage: str) -> None:
        """Setup data module based on stages of training."""
        if stage == "fit":
            self.train_dataset = []
            self.val_dataset = []

            for cv_idx in range(self.n_splits):
                cv_df = self.data[self.data["cv"] == cv_idx]
                train_samples = cv_df[cv_df["split"] == "train"]["samples"].tolist()
                train_targets = cv_df[cv_df["split"] == "train"]["targets"].tolist()
                val_samples = cv_df[cv_df["split"] == "val"]["samples"].tolist()
                val_targets = cv_df[cv_df["split"] == "val"]["targets"].tolist()
                self.train_dataset.append(
                    self.dataset(
                        class_to_idx=self.class_to_idx,
                        samples=train_samples,
                        targets=train_targets,
                        transform=self.train_transform,
                        roi=self.roi,
                    )
                )
                self.val_dataset.append(
                    self.dataset(
                        class_to_idx=self.class_to_idx,
                        samples=val_samples,
                        targets=val_targets,
                        transform=self.val_transform,
                        roi=self.roi,
                    )
                )
            all_samples = self.data[self.data["cv"] == 0]["samples"].tolist()
            all_targets = self.data[self.data["cv"] == 0]["targets"].tolist()
            self.full_dataset = self.dataset(
                class_to_idx=self.class_to_idx,
                samples=all_samples,
                targets=all_targets,
                transform=self.train_transform,
                roi=self.roi,
            )
        if stage == "test":
            test_samples = self.data[self.data["split"] == "test"]["samples"].tolist()
            test_targets = self.data[self.data["split"] == "test"]["targets"]
            self.test_dataset = self.dataset(
                class_to_idx=self.class_to_idx,
                samples=test_samples,
                targets=test_targets.tolist(),
                transform=self.test_transform,
                roi=self.roi,
                allow_missing_label=True,
            )

    def predict_dataloader(self) -> DataLoader:
        """Returns a dataloader used for predictions."""
        return self.test_dataloader()

    def train_dataloader(self) -> list[DataLoader]:
        """Returns a list of train dataloader.

        Raises:
            ValueError: If train dataset is not initialized.

        Returns:
            list of train dataloader.
        """
        if not self.train_dataset_available:
            raise ValueError("Train dataset is not initialized")

        loader = []
        for dataset in self.train_dataset:
            loader.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=not self.cache,
                    num_workers=self.num_workers,
                    drop_last=False,
                    pin_memory=True,
                )
            )
        return loader

    def val_dataloader(self) -> list[DataLoader]:
        """Returns a list of validation dataloader.

        Raises:
            ValueError: If validation dataset is not initialized.

        Returns:
            List of validation dataloader.
        """
        if not self.val_dataset_available:
            raise ValueError("Validation dataset is not initialized")

        loader = []
        for dataset in self.val_dataset:
            loader.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    drop_last=False,
                    pin_memory=True,
                )
            )

        return loader

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

    def full_dataloader(self) -> DataLoader:
        """Return a dataloader to perform training on the entire dataset.

        Returns:
            dataloader to perform training on the entire dataset after evaluation. This is useful
            to perform a final training on the entire dataset after the evaluation phase.

        """
        if self.full_dataset is None:
            raise ValueError("Full dataset is not initialized")

        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=not self.cache,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )


class MultilabelClassificationDataModule(BaseDataModule):
    """Base class for all multi-label modules.

    Args:
        data_path: Path to the data main folder.
        images_and_labels_file: a path to a txt file containing the relative (to `data_path`) path
            of images with their relative labels, in a comma-separated way.
            E.g.:

             * path1,l1,l2,l3
             * path2,l4,l5
             * ...

            One of `images_and_label` and both `train_split_file` and `test_split_file` must be set.
            Defaults to None.
        name: The name for the data module. Defaults to "multilabel_datamodule".
        dataset: a callable returning a torch.utils.data.Dataset class.
        num_classes: the number of classes in the dataset. This is used to create one-hot encoded
            targets. Defaults to None.
        num_workers: Number of workers for dataloaders. Defaults to 16.
        batch_size: Training batch size. Defaults to 64.
        test_batch_size: Testing batch size. Defaults to 64.
        seed: Random generator seed. Defaults to SegmentationEvalua2.
        val_size: The validation split. Defaults to 0.2.
        test_size: The test split. Defaults to 0.2.
        train_transform: Transformations for train dataset.
            Defaults to None.
        val_transform: Transformations for validation dataset.
            Defaults to None.
        test_transform: Transformations for test dataset.
            Defaults to None.
        train_split_file: The file with train split. Defaults to None.
        val_split_file: The file with validation split. Defaults to None.
        test_split_file: The file with test split. Defaults to None.
        class_to_idx: a clss to idx dictionary. Defaults to None.
    """

    def __init__(
        self,
        data_path: str,
        images_and_labels_file: str | None = None,
        train_split_file: str | None = None,
        test_split_file: str | None = None,
        val_split_file: str | None = None,
        name: str = "multilabel_datamodule",
        dataset: Callable = MultilabelClassificationDataset,
        num_classes: int | None = None,
        num_workers: int = 16,
        batch_size: int = 64,
        test_batch_size: int = 64,
        seed: int = 42,
        val_size: float | None = 0.2,
        test_size: float | None = 0.2,
        train_transform: albumentations.Compose | None = None,
        val_transform: albumentations.Compose | None = None,
        test_transform: albumentations.Compose | None = None,
        class_to_idx: dict[str, int] | None = None,
        **kwargs,
    ):
        super().__init__(
            data_path=data_path,
            name=name,
            num_workers=num_workers,
            batch_size=batch_size,
            seed=seed,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            **kwargs,
        )
        if not (images_and_labels_file is not None or (train_split_file is not None and test_split_file is not None)):
            raise ValueError(
                "Either `images_and_labels_file` or both `train_split_file` and `test_split_file` must be set"
            )
        self.images_and_labels_file = images_and_labels_file
        self.dataset = dataset
        self.num_classes = num_classes
        self.train_batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.train_split_file = train_split_file
        self.test_split_file = test_split_file
        self.val_split_file = val_split_file
        self.class_to_idx = class_to_idx
        self.train_dataset: MultilabelClassificationDataset
        self.val_dataset: MultilabelClassificationDataset
        self.test_dataset: MultilabelClassificationDataset

    def _read_split(self, split_file: str) -> tuple[list[str], list[list[str]]]:
        """Reads split file.

        Args:
            split_file: Path to the split file.

        Returns:
            Tuple containing list of paths to images and list of labels.
        """
        all_samples, all_targets = [], []
        with open(split_file) as f:
            for line in f.readlines():
                split_line = line.split(",")
                sample = os.path.join(self.data_path, split_line[0])
                targets = [t.strip() for t in split_line[1:]]
                if len(targets) == 0:
                    continue
                all_samples.append(sample)
                all_targets.append(targets)
        return all_samples, all_targets

    def _prepare_data(self) -> None:
        """Prepares the data for the data module."""
        if self.images_and_labels_file is not None:
            # Read all images and targets
            all_samples, all_targets = self._read_split(self.images_and_labels_file)
            all_samples = np.array(all_samples).reshape(-1, 1)

            # Targets to idx
            unique_targets = set(utils.flatten_list(all_targets))
            if self.class_to_idx is None:
                self.class_to_idx = {c: i for i, c in enumerate(unique_targets)}

            all_targets = [[self.class_to_idx[t] for t in targets] for targets in all_targets]

            # Transform targets to one-hot
            if self.num_classes is None:
                self.num_classes = len(unique_targets)
            all_targets = np.array([[i in targets for i in range(self.num_classes)] for targets in all_targets]).astype(
                int
            )

            # Create splits
            samples_train, targets_train, samples_test, targets_test = iterative_train_test_split(
                all_samples, all_targets, test_size=self.test_size
            )
        elif self.train_split_file is not None and self.test_split_file is not None:
            # Both train_split_file and test_split_file are set
            samples_train, targets_train = self._read_split(self.train_split_file)
            samples_test, targets_test = self._read_split(self.test_split_file)

            # Create class_to_idx from all targets
            unique_targets = set(utils.flatten_list(targets_test + targets_train))
            if self.class_to_idx is None:
                self.class_to_idx = {c: i for i, c in enumerate(unique_targets)}

            # Transform targets to one-hot
            if self.num_classes is None:
                self.num_classes = len(unique_targets)
            targets_test = [[self.class_to_idx[t] for t in targets] for targets in targets_test]
            targets_test = np.array(
                [[i in targets for i in range(self.num_classes)] for targets in targets_test]
            ).astype(int)
            targets_train = [[self.class_to_idx[t] for t in targets] for targets in targets_train]
            targets_train = np.array(
                [[i in targets for i in range(self.num_classes)] for targets in targets_train]
            ).astype(int)
        else:
            raise ValueError(
                "Either `images_and_labels_file` or both `train_split_file` and `test_split_file` must be set"
            )

        if self.val_split_file:
            if not self.test_split_file or not self.train_split_file:
                raise ValueError("Validation split file is specified but no train or test split file is specified.")
            samples_val, targets_val = self._read_split(self.val_split_file)
            targets_val = [[self.class_to_idx[t] for t in targets] for targets in targets_val]
            targets_val = np.array([[i in targets for i in range(self.num_classes)] for targets in targets_val]).astype(
                int
            )
        else:
            samples_train = np.array(samples_train).reshape(-1, 1)
            targets_train = np.array(targets_train).reshape(-1, self.num_classes)
            samples_train, targets_train, samples_val, targets_val = iterative_train_test_split(
                samples_train, targets_train, test_size=self.val_size
            )

        if isinstance(samples_train, np.ndarray):
            samples_train = samples_train.flatten().tolist()
        if isinstance(samples_val, np.ndarray):
            samples_val = samples_val.flatten().tolist()
        if isinstance(samples_test, np.ndarray):
            samples_test = samples_test.flatten().tolist()

        if isinstance(targets_train, np.ndarray):
            targets_train = list(targets_train)
        if isinstance(targets_val, np.ndarray):
            targets_val = list(targets_val)  # type: ignore[assignment]
        if isinstance(targets_test, np.ndarray):
            targets_test = list(targets_test)

        # Create data
        train_df = pd.DataFrame({"samples": samples_train, "targets": targets_train})
        train_df["split"] = "train"
        val_df = pd.DataFrame({"samples": samples_val, "targets": targets_val})
        val_df["split"] = "val"
        test_df = pd.DataFrame({"samples": samples_test, "targets": targets_test})
        test_df["split"] = "test"
        self.data = pd.concat([train_df, val_df, test_df], axis=0)

    def setup(self, stage: str | None = None) -> None:
        """Setup data module based on stages of training."""
        if stage in ["train", "fit"]:
            train_samples = self.data[self.data["split"] == "train"]["samples"].tolist()
            train_targets = self.data[self.data["split"] == "train"]["targets"].tolist()
            val_samples = self.data[self.data["split"] == "val"]["samples"].tolist()
            val_targets = self.data[self.data["split"] == "val"]["targets"].tolist()
            self.train_dataset = self.dataset(
                samples=train_samples,
                targets=train_targets,
                transform=self.train_transform,
                class_to_idx=self.class_to_idx,
            )
            self.val_dataset = self.dataset(
                samples=val_samples,
                targets=val_targets,
                transform=self.val_transform,
                class_to_idx=self.class_to_idx,
            )
        if stage == "test":
            test_samples = self.data[self.data["split"] == "test"]["samples"].tolist()
            test_targets = self.data[self.data["split"] == "test"]["targets"].tolist()
            self.test_dataset = self.dataset(
                samples=test_samples,
                targets=test_targets,
                transform=self.test_transform,
                class_to_idx=self.class_to_idx,
            )

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
