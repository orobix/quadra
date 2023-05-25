# pylint: disable=redefined-outer-name
from pathlib import Path

import albumentations as alb
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2

from quadra.datasets import AnomalyDataset
from quadra.datasets.anomaly import make_anomaly_dataset
from quadra.utils.tests.fixtures.dataset.anomaly import (
    AnomalyDatasetArguments,
    anomaly_dataset,
    base_anomaly_dataset,
)


@pytest.mark.parametrize(
    "dataset_arguments",
    [
        AnomalyDatasetArguments(**{"train_samples": 10, "val_samples": (10, 10), "test_samples": (10, 10)}),
        AnomalyDatasetArguments(**{"train_samples": 10, "val_samples": (10, 10), "test_samples": (0, 10)}),
    ],
)
@pytest.mark.parametrize("create_test_set_if_empty", [True, False])
def test_make_anomaly_dataset(anomaly_dataset: anomaly_dataset, create_test_set_if_empty: bool):
    data_path, arguments = anomaly_dataset

    pandas_dataset = make_anomaly_dataset(
        path=Path(data_path), split_ratio=0.1, split=None, create_test_set_if_empty=create_test_set_if_empty
    )

    assert len(pandas_dataset) == arguments.train_samples + sum(arguments.val_samples) + sum(arguments.test_samples)

    assert (pandas_dataset["split"] == "val").sum() == sum(arguments.val_samples)

    if not create_test_set_if_empty:
        assert (pandas_dataset["split"] == "train").sum() == arguments.train_samples
        assert (pandas_dataset["split"] == "test").sum() == sum(arguments.test_samples)
    else:
        # If this flag is true and the test set contains no good samples then good samples are moved from the train set
        if arguments.test_samples[0] == 0:
            good_test_samples = pandas_dataset[
                (pandas_dataset["split"] == "test") & (pandas_dataset["label_index"] == 0)
            ]
            assert (pandas_dataset["split"] == "train").sum() == arguments.train_samples - len(good_test_samples)
            assert len(good_test_samples) > 0


@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_anomaly_dataset(base_anomaly_dataset: base_anomaly_dataset, task: str):
    data_path, _ = base_anomaly_dataset

    pandas_dataset = make_anomaly_dataset(
        path=Path(data_path),
        split=None,
        create_test_set_if_empty=True,
    )

    transform = alb.Compose(
        [
            alb.Resize(224, 224),
            ToTensorV2(),
        ]
    )

    for split in ["train", "val", "test"]:
        dataset = AnomalyDataset(
            samples=pandas_dataset[pandas_dataset["split"] == split],
            transform=transform,
            task=task,
        )

        assert len(dataset) == (pandas_dataset["split"] == split).sum()

        for item in dataset:
            assert isinstance(item["image"], torch.Tensor)
            assert isinstance(item["label"], np.int64)

            if split == "train":
                assert item.keys() == {"image", "label"}
            else:
                assert isinstance(item["image_path"], str)

                if task == "classification":
                    assert item.keys() == {"image_path", "image", "label"}
                else:
                    assert item.keys() == {"image_path", "image", "label", "mask", "mask_path"}
                    assert isinstance(item["mask"], torch.Tensor)
                    assert isinstance(item["mask_path"], str)
