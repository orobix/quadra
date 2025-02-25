from __future__ import annotations

import multiprocessing as mp
import multiprocessing.pool as mpp
import os
import pickle as pkl
import typing
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from typing import Any, Literal, cast

import albumentations
import numpy as np
import pandas as pd
import torch
import xxhash
from pytorch_lightning import LightningDataModule
from tqdm import tqdm

from quadra.utils import utils

log = utils.get_logger(__name__)
TrainDataset = torch.utils.data.Dataset | Sequence[torch.utils.data.Dataset]
ValDataset = torch.utils.data.Dataset | Sequence[torch.utils.data.Dataset]
TestDataset = torch.utils.data.Dataset


def load_data_from_disk_dec(func):
    """Load data from disk if it exists."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function to load data from disk if it exists."""
        self = cast(BaseDataModule, args[0])
        self.restore_checkpoint()
        return func(*args, **kwargs)

    return wrapper


class DecorateParentMethod(type):
    """Metaclass to decorate methods of subclasses."""

    def __new__(cls, name, bases, dct):
        """Create new  decorator for parent class methods."""
        method_decorator_mapper = {
            "setup": load_data_from_disk_dec,
        }
        for method_name, decorator in method_decorator_mapper.items():
            if method_name in dct:
                dct[method_name] = decorator(dct[method_name])

        return super().__new__(cls, name, bases, dct)


def compute_file_content_hash(path: str, hash_size: Literal[32, 64, 128] = 64) -> str:
    """Get hash of a file based on its content.

    Args:
        path: Path to the file.
        hash_size: Size of the hash. Must be one of [32, 64, 128].

    Returns:
        The hash of the file.
    """
    with open(path, "rb") as f:
        data = f.read()

        if hash_size == 32:
            file_hash = xxhash.xxh32(data, seed=42).hexdigest()
        elif hash_size == 64:
            file_hash = xxhash.xxh64(data, seed=42).hexdigest()
        elif hash_size == 128:
            file_hash = xxhash.xxh128(data, seed=42).hexdigest()
        else:
            raise ValueError(f"Invalid hash size {hash_size}. Must be one of [32, 64, 128].")

    return file_hash


def compute_file_size_hash(path: str, hash_size: Literal[32, 64, 128] = 64) -> str:
    """Get hash of a file based on its size.

    Args:
        path: Path to the file.
        hash_size: Size of the hash. Must be one of [32, 64, 128].

    Returns:
        The hash of the file.
    """
    data = str(os.path.getsize(path))

    if hash_size == 32:
        file_hash = xxhash.xxh32(data, seed=42).hexdigest()
    elif hash_size == 64:
        file_hash = xxhash.xxh64(data, seed=42).hexdigest()
    elif hash_size == 128:
        file_hash = xxhash.xxh128(data, seed=42).hexdigest()
    else:
        raise ValueError(f"Invalid hash size {hash_size}. Must be one of [32, 64, 128].")

    return file_hash


@typing.no_type_check
def istarmap(self, func: Callable, iterable: Iterable, chunksize: int = 1):
    # pylint: disable=all
    """Starmap-version of imap."""
    self._check_running()
    if chunksize < 1:
        raise ValueError(f"Chunksize must be 1+, not {chunksize:n}")

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put((self._guarded_task_generation(result._job, mpp.starmapstar, task_batches), result._set_length))
    return (item for chunk in result for item in chunk)


# Patch Pool class to include istarmap
mpp.Pool.istarmap = istarmap  # type: ignore[attr-defined]


class BaseDataModule(LightningDataModule, metaclass=DecorateParentMethod):
    """Base class for all data modules.

    Args:
        data_path: Path to the data main folder.
        name: The name for the data module. Defaults to "base_datamodule".
        num_workers: Number of workers for dataloaders. Defaults to 16.
        batch_size: Batch size. Defaults to 32.
        seed: Random generator seed. Defaults to 42.
        train_transform: Transformations for train dataset.
            Defaults to None.
        val_transform: Transformations for validation dataset.
            Defaults to None.
        test_transform: Transformations for test dataset.
            Defaults to None.
        enable_hashing: Whether to enable hashing of images. Defaults to True.
        hash_size: Size of the hash. Must be one of [32, 64, 128]. Defaults to 64.
        hash_type: Type of hash to use, if content hash is used, the hash is computed on the file content, otherwise
            the hash is computed on the file size which is faster but less safe. Defaults to "content".
    """

    def __init__(
        self,
        data_path: str,
        name: str = "base_datamodule",
        num_workers: int = 16,
        batch_size: int = 32,
        seed: int = 42,
        load_aug_images: bool = False,
        aug_name: str | None = None,
        n_aug_to_take: int | None = None,
        replace_str_from: str | None = None,
        replace_str_to: str | None = None,
        train_transform: albumentations.Compose | None = None,
        val_transform: albumentations.Compose | None = None,
        test_transform: albumentations.Compose | None = None,
        enable_hashing: bool = True,
        hash_size: Literal[32, 64, 128] = 64,
        hash_type: Literal["content", "size"] = "content",
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_path = data_path
        self.name = name
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.enable_hashing = enable_hashing
        self.hash_size = hash_size
        self.hash_type = hash_type

        if self.hash_size not in [32, 64, 128]:
            raise ValueError(f"Invalid hash size {self.hash_size}. Must be one of [32, 64, 128].")

        self.load_aug_images = load_aug_images
        self.aug_name = aug_name
        self.n_aug_to_take = n_aug_to_take
        self.replace_str_from = replace_str_from
        self.replace_str_to = replace_str_to
        self.extra_args: dict[str, Any] = {}
        self.train_dataset: TrainDataset
        self.val_dataset: ValDataset
        self.test_dataset: TestDataset
        self.data: pd.DataFrame
        self.data_folder = "data"
        os.makedirs(self.data_folder, exist_ok=True)
        self.datamodule_checkpoint_file = os.path.join(self.data_folder, "datamodule.pkl")
        self.dataset_file = os.path.join(self.data_folder, "dataset.csv")

    @property
    def train_data(self) -> pd.DataFrame:
        """Get train data."""
        if not hasattr(self, "data"):
            raise ValueError("`data` attribute is not set. Cannot load train data.")
        return self.data[self.data["split"] == "train"]

    @property
    def val_data(self) -> pd.DataFrame:
        """Get validation data."""
        if not hasattr(self, "data"):
            raise ValueError("`data` attribute is not set. Cannot load val data.")
        return self.data[self.data["split"] == "val"]

    @property
    def test_data(self) -> pd.DataFrame:
        """Get test data."""
        if not hasattr(self, "data"):
            raise ValueError("`data` attribute is not set. Cannot load test data.")
        return self.data[self.data["split"] == "test"]

    def _dataset_available(self, dataset_name: str) -> bool:
        """Checks if the dataset is available.

        Args:
            dataset_name : Name of the dataset attribute.

        Returns:
            True if the dataset is available, False otherwise.
        """
        available = hasattr(self, dataset_name) and getattr(self, dataset_name) is not None
        if available:
            dataset_attr = getattr(self, dataset_name)
            if isinstance(dataset_attr, list):
                available = all(len(d) > 0 for d in dataset_attr)
            else:
                available = len(dataset_attr) > 0
        return available

    @property
    def train_dataset_available(self) -> bool:
        """Checks if the train dataset is available."""
        return self._dataset_available("train_dataset")

    @property
    def val_dataset_available(self) -> bool:
        """Checks if the validation dataset is available."""
        return self._dataset_available("val_dataset")

    @property
    def test_dataset_available(self) -> bool:
        """Checks if the test dataset is available."""
        return self._dataset_available("test_dataset")

    def _prepare_data(self) -> None:
        """Prepares the data, this should have exactly the same logic as the prepare_data method
        of a LightningModule.
        """
        raise NotImplementedError(
            "This method must be implemented, it should contain all the logic that normally is "
            "contained in the prepare_data method of a LightningModule."
        )

    def hash_data(self) -> None:
        """Computes the hash of the files inside the datasets."""
        if not self.enable_hashing:
            return

        # TODO: We need to find a way to annotate the columns of data.
        paths_and_hash_length = zip(self.data["samples"], [self.hash_size] * len(self.data), strict=False)

        with mp.Pool(min(8, mp.cpu_count() - 1)) as pool:
            self.data["hash"] = list(
                tqdm(
                    pool.istarmap(  # type: ignore[attr-defined]
                        compute_file_content_hash if self.hash_type == "content" else compute_file_size_hash,
                        paths_and_hash_length,
                    ),
                    total=len(self.data),
                    desc="Computing hashes",
                )
            )

        self.data["hash_type"] = self.hash_type

    def prepare_data(self) -> None:
        """Prepares the data, should be overridden by subclasses."""
        if hasattr(self, "data"):
            return

        self._prepare_data()
        self.hash_data()
        self.save_checkpoint()

    def __getstate__(self) -> dict[str, Any]:
        """This method is called when pickling the object.
        It's useful to remove attributes that shouldn't be pickled.
        """
        state = self.__dict__.copy()
        if "trainer" in state:
            # Lightning injects the trainer in the datamodule, we don't want to pickle it.
            del state["trainer"]

        return state

    def save_checkpoint(self) -> None:
        """Saves the datamodule to disk, utility function that is called from prepare_data. We are required to save
        datamodule to disk because we can't assign attributes to the datamodule in prepare_data when working with
        multiple gpus.
        """
        if not os.path.exists(self.datamodule_checkpoint_file) and not os.path.exists(self.dataset_file):
            with open(self.datamodule_checkpoint_file, "wb") as f:
                pkl.dump(self, f)

            self.data.to_csv(self.dataset_file, index=False)
            log.info("Datamodule checkpoint saved to disk.")

        if "targets" in self.data:
            if isinstance(self.data["targets"].iloc[0], np.ndarray):
                # If we find a numpy array target it's very likely one hot encoded,
                # in that case we just print the number of train/val/test samples
                grouping = ["split"]
            else:
                grouping = ["split", "targets"]
            log.info("Dataset Info:")
            split_order = {"train": 0, "val": 1, "test": 2}
            log.info(
                "\n%s",
                self.data.groupby(grouping)
                .size()
                .to_frame()
                .reset_index()
                .sort_values(by=["split"], key=lambda x: x.map(split_order))
                .rename(columns={0: "count"})
                .to_string(index=False),
            )

    def restore_checkpoint(self) -> None:
        """Loads the data from disk, utility function that should be called from setup."""
        if hasattr(self, "data"):
            return

        if not os.path.isfile(self.datamodule_checkpoint_file):
            raise ValueError(f"Dataset file {self.datamodule_checkpoint_file} does not exist.")

        with open(self.datamodule_checkpoint_file, "rb") as f:
            checkpoint_datamodule = pkl.load(f)
            for key, value in checkpoint_datamodule.__dict__.items():
                setattr(self, key, value)

    # TODO: Check if this function can be removed
    def load_augmented_samples(
        self,
        samples: list[str],
        targets: list[Any],
        replace_str_from: str | None = None,
        replace_str_to: str | None = None,
        shuffle: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Loads augmented samples."""
        if self.n_aug_to_take is None:
            raise ValueError("`n_aug_to_take` is not set. Cannot load augmented samples.")
        aug_samples = []
        aug_labels = []
        for sample, label in zip(samples, targets, strict=False):
            aug_samples.append(sample)
            aug_labels.append(label)
            final_sample = sample
            if replace_str_from is not None and replace_str_to is not None:
                final_sample = final_sample.replace(replace_str_from, replace_str_to)
            base, ext = os.path.splitext(final_sample)
            for k in range(self.n_aug_to_take):
                aug_samples.append(base + "_" + str(k + 1) + ext)
                aug_labels.append(label)
        samples = aug_samples
        targets = aug_labels
        if shuffle:
            idexs = np.arange(len(aug_samples))
            np.random.shuffle(idexs)
            samples = np.array(samples)[idexs].tolist()
            targets = np.array(targets)[idexs].tolist()
        return samples, targets
