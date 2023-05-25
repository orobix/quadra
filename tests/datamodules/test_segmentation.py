# pylint: disable=redefined-outer-name,unsubscriptable-object
import os

import numpy as np
import pytest

from quadra.datamodules import SegmentationDataModule, SegmentationMulticlassDataModule
from quadra.utils.tests.fixtures.dataset import SegmentationDatasetArguments, segmentation_dataset


@pytest.mark.parametrize("dataset_arguments", [SegmentationDatasetArguments(**{"train_samples": [15, 20]})])
def test_binary_segmentation_datamodule(segmentation_dataset: segmentation_dataset):
    data_path, arguments, _ = segmentation_dataset
    datamodule = SegmentationDataModule(data_path=data_path, num_workers=1)
    datamodule.prepare_data()
    datamodule.restore_checkpoint()

    datamodule_labels = []
    for split in ["train", "val", "test"]:
        datamodule_labels += datamodule.data[datamodule.data["split"] == split]["targets"].tolist()

    datamodule_labels = np.array(datamodule_labels)

    assert arguments.train_samples[0] == (datamodule_labels == 0).sum()
    assert arguments.train_samples[1] == (datamodule_labels == 1).sum()


@pytest.mark.parametrize(
    "dataset_arguments",
    [SegmentationDatasetArguments(**{"train_samples": [15, 20], "val_samples": [5, 10], "test_samples": [10, 10]})],
)
@pytest.mark.parametrize("exclude_good", [False, True])
def test_binary_segmentation_with_split_datamodule(segmentation_dataset: segmentation_dataset, exclude_good: bool):
    data_path, arguments, _ = segmentation_dataset
    datamodule = SegmentationDataModule(
        data_path=data_path,
        num_workers=1,
        train_split_file=os.path.join(data_path, "train.txt"),
        val_split_file=os.path.join(data_path, "val.txt"),
        test_split_file=os.path.join(data_path, "test.txt"),
        exclude_good=exclude_good,
    )
    datamodule.prepare_data()
    datamodule.restore_checkpoint()

    train_labels = np.array(datamodule.data[datamodule.data["split"] == "train"]["targets"])
    val_labels = np.array(datamodule.data[datamodule.data["split"] == "val"]["targets"])
    test_labels = np.array(datamodule.data[datamodule.data["split"] == "test"]["targets"])

    # Check that the splits taken from the files are correct
    for l in [0, 1]:
        if exclude_good and l == 0:
            assert (train_labels == l).sum() == 0
        else:
            assert arguments.train_samples[l] == (train_labels == l).sum()
        assert arguments.val_samples[l] == (val_labels == l).sum()
        assert arguments.test_samples[l] == (test_labels == l).sum()


@pytest.mark.parametrize(
    "dataset_arguments",
    [
        SegmentationDatasetArguments(
            **{"train_samples": [15, 20, 10], "val_samples": [5, 10, 10], "test_samples": [10, 10, 5]}
        )
    ],
)
@pytest.mark.parametrize("exclude_good", [False, True])
def test_segmentation_multilabel_datamodule(segmentation_dataset: segmentation_dataset, exclude_good: bool):
    data_path, arguments, class_to_idx = segmentation_dataset
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    datamodule = SegmentationMulticlassDataModule(
        data_path=data_path,
        idx_to_class=idx_to_class,
        train_split_file=os.path.join(data_path, "train.txt"),
        val_split_file=os.path.join(data_path, "val.txt"),
        test_split_file=os.path.join(data_path, "test.txt"),
        exclude_good=exclude_good,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("test")
    num_train_samples = len(datamodule.data[datamodule.data["split"] == "train"]["targets"])
    num_val_samples = len(datamodule.data[datamodule.data["split"] == "val"]["targets"])
    num_test_samples = len(datamodule.data[datamodule.data["split"] == "test"]["targets"])
    if not exclude_good:
        assert num_train_samples == sum(arguments.train_samples)
    else:
        assert num_train_samples == sum(arguments.train_samples[1:])
    assert num_val_samples == sum(arguments.val_samples)
    assert num_test_samples == sum(arguments.test_samples)
