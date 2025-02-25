from __future__ import annotations

import os
import random
from pathlib import Path

import albumentations as alb
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset

from quadra.utils.utils import IMAGE_EXTENSIONS


def create_validation_set_from_test_set(samples: DataFrame, seed: int = 0) -> DataFrame:
    """Craete Validation Set from Test Set.

    This function creates a validation set from test set by splitting both
    normal and abnormal samples to two.

    Args:
        samples: Dataframe containing dataset info such as filenames, splits etc.
        seed: Random seed to ensure reproducibility. Defaults to 0.
    """
    if seed > 0:
        random.seed(seed)

    # Split normal images.
    normal_test_image_indices = samples.index[(samples.split == "test") & (samples.targets == "good")].to_list()
    num_normal_valid_images = len(normal_test_image_indices) // 2

    indices_to_sample = random.sample(population=normal_test_image_indices, k=num_normal_valid_images)
    samples.loc[indices_to_sample, "split"] = "val"

    # Split abnormal images.
    abnormal_test_image_indices = samples.index[(samples.split == "test") & (samples.targets != "good")].to_list()
    num_abnormal_valid_images = len(abnormal_test_image_indices) // 2

    indices_to_sample = random.sample(population=abnormal_test_image_indices, k=num_abnormal_valid_images)
    samples.loc[indices_to_sample, "split"] = "val"

    return samples


def split_normal_images_in_train_set(samples: DataFrame, split_ratio: float = 0.1, seed: int = 0) -> DataFrame:
    """Split normal images in train set.

        This function splits the normal images in training set and assigns the
        values to the test set. This is particularly useful especially when the
        test set does not contain any normal images.

        This is important because when the test set doesn't have any normal images,
        AUC computation fails due to having single class.

    Args:
        samples: Dataframe containing dataset info such as filenames, splits etc.
        split_ratio: Train-Test normal image split ratio. Defaults to 0.1.
        seed: Random seed to ensure reproducibility. Defaults to 0.

    Returns:
        Output dataframe where the part of the training set is assigned to test set.
    """
    if seed > 0:
        random.seed(seed)

    normal_train_image_indices = samples.index[(samples.split == "train") & (samples.targets == "good")].to_list()
    num_normal_train_images = len(normal_train_image_indices)
    num_normal_valid_images = int(num_normal_train_images * split_ratio)

    indices_to_split_from_train_set = random.sample(population=normal_train_image_indices, k=num_normal_valid_images)
    samples.loc[indices_to_split_from_train_set, "split"] = "test"

    return samples


def make_anomaly_dataset(
    path: Path,
    split: str | None = None,
    split_ratio: float = 0.1,
    seed: int = 0,
    mask_suffix: str | None = None,
    create_test_set_if_empty: bool = True,
) -> DataFrame:
    """Create dataframe by parsing a folder following the MVTec data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/label/image_filename.xyz
        path/to/dataset/ground_truth/label/mask_filename.png

    Masks MUST be png images, no other format is allowed
    Split can be either train/val/test

    This function creates a dataframe to store the parsed information based on the following format:
    |---|---------------|-------|---------|--------------|-----------------------------------------------|-------------|
    |   | path          | split | targets | samples      | mask_path                                     | label_index |
    |---|---------------|-------|---------|--------------|-----------------------------------------------|-------------|
    | 0 | datasets/name |  test |  defect | filename.xyz | ground_truth/defect/filename{mask_suffix}.png | 1           |
    |---|---------------|-------|---------|--------------|-----------------------------------------------|-------------|

    Args:
        path: Path to dataset
        split: Dataset split (i.e., either train or test). Defaults to None.
        split_ratio: Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.1.
        seed: Random seed to ensure reproducibility when splitting. Defaults to 0.
        mask_suffix: String to append to the base filename to get the mask name, by default for MVTec dataset masks
            are saved as imagename_mask.png in this case the parameter shoul be filled with "_mask"
        create_test_set_if_empty: If True, create a test set if the test set is empty.


    Example:
        The following example shows how to get training samples from MVTec bottle category:

        >>> root = Path('./MVTec')
        >>> category = 'bottle'
        >>> path = root / category
        >>> path
        PosixPath('MVTec/bottle')

        >>> samples = make_anomaly_dataset(path, split='train', split_ratio=0.1, seed=0)
        >>> samples.head()
           path         split label image_path                           mask_path                   label_index
        0  MVTec/bottle train good MVTec/bottle/train/good/105.png MVTec/bottle/ground_truth/good/105_mask.png 0
        1  MVTec/bottle train good MVTec/bottle/train/good/017.png MVTec/bottle/ground_truth/good/017_mask.png 0
        2  MVTec/bottle train good MVTec/bottle/train/good/137.png MVTec/bottle/ground_truth/good/137_mask.png 0
        3  MVTec/bottle train good MVTec/bottle/train/good/152.png MVTec/bottle/ground_truth/good/152_mask.png 0
        4  MVTec/bottle train good MVTec/bottle/train/good/109.png MVTec/bottle/ground_truth/good/109_mask.png 0

    Returns:
        An output dataframe containing samples for the requested split (ie., train or test)
    """
    samples_list = [
        (str(path),) + filename.parts[-3:]
        for filename in path.glob("**/*")
        if filename.is_file()
        and os.path.splitext(filename)[-1].lower() in IMAGE_EXTENSIONS
        and ".ipynb_checkpoints" not in str(filename)
    ]

    if len(samples_list) == 0:
        raise RuntimeError(f"Found 0 images in {path}")

    samples_list.sort()

    data = pd.DataFrame(samples_list, columns=["path", "split", "targets", "samples"])
    data = data[data.split != "ground_truth"]

    # Create mask_path column, masks MUST have png extension
    data["mask_path"] = (
        data.path
        + "/ground_truth/"
        + data.targets
        + "/"
        + data.samples.apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        + (f"{mask_suffix}.png" if mask_suffix is not None else ".png")
    )

    # Modify image_path column by converting to absolute path
    data["samples"] = data.path + "/" + data.split + "/" + data.targets + "/" + data.samples

    # Split the normal images in training set if test set doesn't
    # contain any normal images. This is needed because AUC score
    # cannot be computed based on 1-class
    if sum((data.split == "test") & (data.targets == "good")) == 0 and create_test_set_if_empty:
        data = split_normal_images_in_train_set(data, split_ratio, seed)

    # Good images don't have mask
    data.loc[(data.split == "test") & (data.targets == "good"), "mask_path"] = ""

    # Create label index for normal (0), anomalous (1) and unknown (-1) images.
    data.loc[data.targets == "good", "label_index"] = 0
    data.loc[~data.targets.isin(["good", "unknown"]), "label_index"] = 1
    data.loc[data.targets == "unknown", "label_index"] = -1
    data.label_index = data.label_index.astype(int)

    # Get the data frame for the split.
    if split is not None and split in ["train", "val", "test"]:
        data = data[data.split == split]
        data = data.reset_index(drop=True)

    return data


class AnomalyDataset(Dataset):
    """Anomaly Dataset.

    Args:
        transform: Albumentations compose.
        task: ``classification`` or ``segmentation``
        samples: Pandas dataframe containing samples following the same structure created by make_anomaly_dataset
        valid_area_mask: Optional path to the mask to use to filter out the valid area of the image. If None, the
            whole image is considered valid.
        crop_area: Optional tuple of 4 integers (x1, y1, x2, y2) to crop the image to the specified area. If None, the
            whole image is considered valid.
    """

    def __init__(
        self,
        transform: alb.Compose,
        samples: DataFrame,
        task: str = "segmentation",
        valid_area_mask: str | None = None,
        crop_area: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.task = task
        self.transform = transform

        self.samples = samples
        self.samples = self.samples.reset_index(drop=True)
        self.split = self.samples.split.unique()[0]

        self.crop_area = crop_area
        self.valid_area_mask: np.ndarray | None = None

        if valid_area_mask is not None:
            if not os.path.exists(valid_area_mask):
                raise RuntimeError(f"Valid area mask {valid_area_mask} does not exist.")

            self.valid_area_mask = cv2.imread(valid_area_mask, 0) > 0

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        """Get dataset item for the index ``index``.

        Args:
            index: Index to get the item.

        Returns:
            Dict of image tensor during training.
            Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        item: dict[str, str | Tensor] = {}

        image_path = self.samples.samples.iloc[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_image_shape = image.shape
        if self.valid_area_mask is not None:
            image = image * self.valid_area_mask[:, :, np.newaxis]

        if self.crop_area is not None:
            image = image[self.crop_area[1] : self.crop_area[3], self.crop_area[0] : self.crop_area[2]]

        label_index = self.samples.label_index[index]

        if self.split == "train":
            pre_processed = self.transform(image=image)
            item = {"image": pre_processed["image"], "label": label_index}
        elif self.split in ["val", "test"]:
            item["image_path"] = image_path
            item["label"] = label_index

            if self.task == "segmentation":
                mask_path = self.samples.mask_path[index]

                # If good images have no associated mask create an empty one
                if label_index == 0:
                    mask = np.zeros(shape=original_image_shape[:2])
                elif os.path.isfile(mask_path):
                    mask = cv2.imread(mask_path, flags=0) / 255.0
                else:
                    # We need ones in the mask to compute correctly at least image level f1 score
                    mask = np.ones(shape=original_image_shape[:2])

                if self.valid_area_mask is not None:
                    mask = mask * self.valid_area_mask

                if self.crop_area is not None:
                    mask = mask[self.crop_area[1] : self.crop_area[3], self.crop_area[0] : self.crop_area[2]]

                pre_processed = self.transform(image=image, mask=mask)

                item["mask_path"] = mask_path
                item["mask"] = pre_processed["mask"]
            else:
                pre_processed = self.transform(image=image)

            item["image"] = pre_processed["image"]
        return item
