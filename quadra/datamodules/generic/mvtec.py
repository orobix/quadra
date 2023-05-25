import os
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive

from quadra.datamodules import AnomalyDataModule
from quadra.utils.utils import get_logger

log = get_logger(__name__)


DATASET_BASE_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/"

DATASET_URL = {
    "bottle": DATASET_BASE_URL + "420937370-1629951468/bottle.tar.xz",
    "capsule": DATASET_BASE_URL + "420937454-1629951595/capsule.tar.xz",
    "carpet": DATASET_BASE_URL + "420937484-1629951672/carpet.tar.xz",
    "grid": DATASET_BASE_URL + "420937487-1629951814/grid.tar.xz",
    "hazelnut": DATASET_BASE_URL + "420937545-1629951845/hazelnut.tar.xz",
    "leather": DATASET_BASE_URL + "420937607-1629951964/leather.tar.xz",
    "metal_nut": DATASET_BASE_URL + "420937637-1629952063/metal_nut.tar.xz",
    "pill": DATASET_BASE_URL + "420938129-1629953099/pill.tar.xz",
    "screw": DATASET_BASE_URL + "420938130-1629953152/screw.tar.xz",
    "tile": DATASET_BASE_URL + "420938133-1629953189/tile.tar.xz",
    "toothbrush": DATASET_BASE_URL + "420938134-1629953256/toothbrush.tar.xz",
    "transistor": DATASET_BASE_URL + "420938166-1629953277/transistor.tar.xz",
    "wood": DATASET_BASE_URL + "420938383-1629953354/wood.tar.xz",
    "zipper": DATASET_BASE_URL + "420938385-1629953449/zipper.tar.xz",
}


class MVTecDataModule(AnomalyDataModule):
    """Standard anomaly datamodule with automatic download of the MVTec dataset."""

    def __init__(self, data_path: str, category: str, **kwargs):
        if category not in DATASET_URL:
            raise ValueError(f"Unknown category {category}. Available categories are {list(DATASET_URL.keys())}")

        super().__init__(data_path=data_path, category=category, **kwargs)

    def download_data(self) -> None:
        """Download the MVTec dataset."""
        if self.category is None:
            raise ValueError("Category must be specified for MVTec dataset.")

        if os.path.exists(self.data_path):
            log.info("The path %s already exists. Skipping download.", os.path.join(self.data_path, self.category))
            return

        log.info("Downloading and extracting MVTec dataset for category %s to %s", self.category, self.data_path)
        # self.data_path is the path to the category folder that will be created by the download_and_extract_archive
        data_path_no_category = str(Path(self.data_path).parent)
        download_and_extract_archive(DATASET_URL[self.category], data_path_no_category, remove_finished=True)

    def _prepare_data(self) -> None:
        """Prepare the MVTec dataset."""
        self.download_data()
        return super()._prepare_data()
