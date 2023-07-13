import os
import pickle as pkl
from functools import wraps
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import albumentations
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule

from quadra.utils import utils

log = utils.get_logger(__name__)
IMAGE_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm"]
TrainDataset = Union[torch.utils.data.Dataset, Sequence[torch.utils.data.Dataset]]
ValDataset = Union[torch.utils.data.Dataset, Sequence[torch.utils.data.Dataset]]
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
    """

    def __init__(
        self,
        data_path: str,
        name: str = "base_datamodule",
        num_workers: int = 16,
        batch_size: int = 32,
        seed: int = 42,
        load_aug_images: bool = False,
        aug_name: Optional[str] = None,
        n_aug_to_take: Optional[int] = None,
        replace_str_from: Optional[str] = None,
        replace_str_to: Optional[str] = None,
        train_transform: Optional[albumentations.Compose] = None,
        val_transform: Optional[albumentations.Compose] = None,
        test_transform: Optional[albumentations.Compose] = None,
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

        self.load_aug_images = load_aug_images
        self.aug_name = aug_name
        self.n_aug_to_take = n_aug_to_take
        self.replace_str_from = replace_str_from
        self.replace_str_to = replace_str_to
        self.extra_args: Dict[str, Any] = {}
        self.train_dataset: TrainDataset
        self.val_dataset: ValDataset
        self.test_dataset: TestDataset
        self.data: pd.DataFrame
        self.data_folder = "data"
        os.makedirs(self.data_folder, exist_ok=True)
        self.datamodule_checkpoint_file = os.path.join(self.data_folder, "datamodule.pkl")

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

    def prepare_data(self) -> None:
        """Prepares the data, should be overridden by subclasses."""
        if hasattr(self, "data"):
            return

        self._prepare_data()
        self.save_checkpoint()

    def __getstate__(self) -> Dict[str, Any]:
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
        saved_to_disk = False
        if not os.path.exists(self.datamodule_checkpoint_file):
            with open(self.datamodule_checkpoint_file, "wb") as f:
                pkl.dump(self, f)
            saved_to_disk = True

        if saved_to_disk:
            log.info("Datamodule checkpoint saved to disk.")

        if "targets" in self.data and not isinstance(self.data["targets"].iloc[0], np.ndarray):
            # If we find a numpy array target it's very likely one hot encoded, in that case we don't want to print
            log.info("Dataset Info:")
            split_order = {"train": 0, "val": 1, "test": 2}
            log.info(
                "\n%s",
                self.data.groupby(["split", "targets"])
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
        samples: List[str],
        targets: List[Any],
        replace_str_from: Optional[str] = None,
        replace_str_to: Optional[str] = None,
        shuffle: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Loads augmented samples."""
        if self.n_aug_to_take is None:
            raise ValueError("`n_aug_to_take` is not set. Cannot load augmented samples.")
        aug_samples = []
        aug_labels = []
        for sample, label in zip(samples, targets):
            aug_samples.append(sample)
            aug_labels.append(label)
            if replace_str_from is not None and replace_str_to is not None:
                sample = sample.replace(replace_str_from, replace_str_to)
            base, ext = os.path.splitext(sample)
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
