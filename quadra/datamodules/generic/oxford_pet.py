from __future__ import annotations

import os
from typing import Any

import albumentations
import cv2
import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive

from quadra.datamodules import SegmentationMulticlassDataModule
from quadra.datasets.segmentation import SegmentationDatasetMulticlass
from quadra.utils import utils

log = utils.get_logger(__name__)


class OxfordPetSegmentationDataModule(SegmentationMulticlassDataModule):
    """OxfordPetSegmentationDataModule.

    Args:
        data_path: path to the oxford pet dataset
        idx_to_class: dict with corrispondence btw mask index and classes: {1: class_1, 2: class_2, ..., N: class_N}
            except background class which is 0.
        name: Defaults to "oxford_pet_segmentation_datamodule".
        dataset: Defaults to SegmentationDataset.
        batch_size:  batch size for training. Defaults to 32.
        test_size: Defaults to 0.3.
        val_size:  Defaults to 0.3.
        seed: Defaults to 42.
        num_workers: number of workers for data loading. Defaults to 6.
        train_transform: Train transform. Defaults to None.
        test_transform: Test transform. Defaults to None.
        val_transform: Validation transform. Defaults to None.
    """

    def __init__(
        self,
        data_path: str,
        idx_to_class: dict,
        name: str = "oxford_pet_segmentation_datamodule",
        dataset: type[SegmentationDatasetMulticlass] = SegmentationDatasetMulticlass,
        batch_size: int = 32,
        test_size: float = 0.3,
        val_size: float = 0.3,
        seed: int = 42,
        num_workers: int = 6,
        train_transform: albumentations.Compose | None = None,
        test_transform: albumentations.Compose | None = None,
        val_transform: albumentations.Compose | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            data_path=data_path,
            idx_to_class=idx_to_class,
            name=name,
            dataset=dataset,
            batch_size=batch_size,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
            num_workers=num_workers,
            train_transform=train_transform,
            test_transform=test_transform,
            val_transform=val_transform,
            **kwargs,
        )

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Preprocess mask function that is adapted from
        https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/.

        Args:
            mask: mask to be preprocessed

        Returns:
            binarized mask
        """
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        mask = (mask > 0).astype(np.uint8)
        return mask

    def _check_exists(self, image_folder: str, annotation_folder: str) -> bool:
        """Check if the dataset is already downloaded."""
        return all(os.path.exists(folder) and os.path.isdir(folder) for folder in (image_folder, annotation_folder))

    def download_data(self):
        """Download the dataset if it is not already downloaded."""
        image_folder = os.path.join(self.data_path, "images")
        annotation_folder = os.path.join(self.data_path, "annotations")
        if not self._check_exists(image_folder, annotation_folder):
            for url, md5 in self._RESOURCES:
                download_and_extract_archive(url, download_root=self.data_path, md5=md5, remove_finished=True)
            log.info("Fixing corrupted files...")
            images_filenames = sorted(os.listdir(image_folder))
            for filename in images_filenames:
                file_wo_ext = os.path.splitext(os.path.basename(filename))[0]
                try:
                    mask = cv2.imread(os.path.join(annotation_folder, "trimaps", file_wo_ext + ".png"))
                    mask = self._preprocess_mask(mask)
                    if np.sum(mask) == 0:
                        os.remove(os.path.join(image_folder, filename))
                        os.remove(os.path.join(annotation_folder, "trimaps", file_wo_ext + ".png"))
                        log.info("Removed %s", filename)
                    else:
                        img = cv2.imread(os.path.join(image_folder, filename))
                        cv2.imwrite(os.path.join(image_folder, file_wo_ext + ".jpg"), img)
                except Exception:
                    ip = os.path.join(image_folder, filename)
                    mp = os.path.join(annotation_folder, "trimaps", file_wo_ext + ".png")
                    if os.path.exists(ip):
                        os.remove(ip)
                    if os.path.exists(mp):
                        os.remove(mp)
                    log.info("Removed %s", filename)

    def _prepare_data(self) -> None:
        """Prepare the data to be used by the DataModule."""
        self.download_data()

        trainval_split_filepath = os.path.join(self.data_path, "annotations", "trainval.txt")
        with open(trainval_split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        trainval_filenames = [
            x.split(" ")[0]
            for x in split_data
            if os.path.exists(os.path.join(self.data_path, "images", x.split(" ")[0] + ".jpg"))
        ]
        train_filenames = [x for i, x in enumerate(trainval_filenames) if i % 10 != 0]
        val_filenames = [x for i, x in enumerate(trainval_filenames) if i % 10 == 0]

        test_split_filepath = os.path.join(self.data_path, "annotations", "test.txt")
        with open(test_split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        test_filenames = [
            x.split(" ")[0]
            for x in split_data
            if os.path.exists(os.path.join(self.data_path, "images", x.split(" ")[0] + ".jpg"))
        ]

        df_list = []
        for split_name, filenames in [
            ("train", train_filenames),
            ("val", val_filenames),
            ("test", test_filenames),
        ]:
            samples = [os.path.join(self.data_path, "images", f + ".jpg") for f in filenames]
            masks = [os.path.join(self.data_path, "annotations", "trimaps", f + ".png") for f in filenames]
            targets = [1] * len(filenames)

            df = pd.DataFrame({"samples": samples, "masks": masks, "targets": targets})
            df["split"] = split_name
            df_list.append(df)

        self.data = pd.concat(df_list, axis=0)
