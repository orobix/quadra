# pylint: disable=redefined-outer-name,unsubscriptable-object
import shutil

import numpy as np
import pytest

from quadra.datamodules import SSLDataModule
from quadra.datasets import TwoAugmentationDataset
from quadra.utils.tests.fixtures.dataset import ClassificationDatasetArguments, classification_dataset


@pytest.mark.parametrize(
    "dataset_arguments",
    [ClassificationDatasetArguments(**{"samples": [15, 20, 25], "classes": ["a", "b", "c"]})],
)
def test_classification_data_module_phase_test(classification_dataset: classification_dataset):
    data_path, arguments = classification_dataset
    augmentation_dataset = TwoAugmentationDataset(dataset=None, transform=None)
    datamodule = SSLDataModule(data_path=data_path, augmentation_dataset=augmentation_dataset, num_workers=1)
    datamodule.prepare_data()
    datamodule.restore_checkpoint()

    datamodule_labels = []
    for split in ["train", "val", "test"]:
        datamodule_labels += datamodule.data[datamodule.data["split"] == split]["targets"].tolist()

    datamodule_labels = np.array(datamodule_labels)

    for class_name in arguments.classes:
        assert (datamodule_labels == class_name).sum() == arguments.samples[arguments.classes.index(class_name)]
    shutil.rmtree(data_path)
