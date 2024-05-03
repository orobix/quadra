from __future__ import annotations

import os
import shutil
from typing import Any

import cv2
from torchvision.datasets.mnist import MNIST

from quadra.datamodules import AnomalyDataModule
from quadra.utils.utils import get_logger

log = get_logger(__name__)


class MNISTAnomalyDataModule(AnomalyDataModule):
    """Standard anomaly datamodule with automatic download of the MNIST dataset."""

    def __init__(
        self, data_path: str, good_number: int, limit_data: int = 100, category: str | None = None, **kwargs: Any
    ):
        """Initialize the MNIST anomaly datamodule.

        Args:
            data_path: Path to the dataset
            good_number: Which number to use as a good class, all other numbers are considered anomalies.
            category: The category of the dataset. For mnist this is always None.
            limit_data: Limit the number of images to use for training and testing. Defaults to 100.
            **kwargs: Additional arguments to pass to the AnomalyDataModule.
        """
        super().__init__(data_path=data_path, category=None, **kwargs)
        self.good_number = good_number
        self.limit_data = limit_data

    def download_data(self) -> None:
        """Download the MNIST dataset and move images in the right folders."""
        log.info("Generating MNIST anomaly dataset for good number %s", self.good_number)

        mnist_train_dataset = MNIST(root=self.data_path, train=True, download=True)
        mnist_test_dataset = MNIST(root=self.data_path, train=False, download=True)

        self.data_path = os.path.join(self.data_path, "quadra_mnist_anomaly")

        if os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)

        # Create the folder structure
        train_good_folder = os.path.join(self.data_path, "train", "good")
        test_good_folder = os.path.join(self.data_path, "test", "good")

        os.makedirs(train_good_folder, exist_ok=True)
        os.makedirs(test_good_folder, exist_ok=True)

        # Copy the good train images to the correct folder
        good_train_samples = mnist_train_dataset.data[mnist_train_dataset.targets == self.good_number]
        for i, image in enumerate(good_train_samples.numpy()):
            if i == self.limit_data:
                break
            cv2.imwrite(os.path.join(train_good_folder, f"{i}.png"), image)

        for number in range(10):
            if number == self.good_number:
                good_train_samples = mnist_test_dataset.data[mnist_test_dataset.targets == number]
                for i, image in enumerate(good_train_samples.numpy()):
                    if i == self.limit_data:
                        break
                    cv2.imwrite(os.path.join(test_good_folder, f"{number}_{i}.png"), image)
            else:
                test_bad_folder = os.path.join(self.data_path, "test", str(number))
                os.makedirs(test_bad_folder, exist_ok=True)
                bad_train_samples = mnist_train_dataset.data[mnist_train_dataset.targets == number]
                for i, image in enumerate(bad_train_samples.numpy()):
                    if i == self.limit_data:
                        break

                    cv2.imwrite(os.path.join(test_bad_folder, f"{number}_{i}.png"), image)

    def _prepare_data(self) -> None:
        """Prepare the MNIST dataset."""
        self.download_data()
        return super()._prepare_data()
