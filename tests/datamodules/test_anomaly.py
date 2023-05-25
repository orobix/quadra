# pylint: disable=redefined-outer-name
import pandas as pd
import pytest

from quadra.datamodules import AnomalyDataModule
from quadra.utils.tests.fixtures.dataset import AnomalyDatasetArguments, anomaly_dataset


def _check_total_samples(samples: pd.DataFrame, dataset_arguments: AnomalyDatasetArguments):
    train_samples = dataset_arguments.train_samples
    val_samples_good, val_samples_bad = dataset_arguments.val_samples
    test_samples_good, test_samples_bad = dataset_arguments.test_samples
    assert len(samples) == train_samples + val_samples_good + val_samples_bad + test_samples_good + test_samples_bad
    assert len(samples[samples.split == "train"]) == train_samples
    assert len(samples[samples.split == "val"]) == val_samples_good + val_samples_bad
    assert len(samples[samples.split == "test"]) == test_samples_good + test_samples_bad
    assert len(samples[(samples.split == "val") & (samples.targets == "good")]) == val_samples_good
    assert len(samples[(samples.split == "val") & (samples.targets == "bad")]) == val_samples_bad
    assert len(samples[(samples.split == "test") & (samples.targets == "good")]) == test_samples_good
    assert len(samples[(samples.split == "test") & (samples.targets == "bad")]) == test_samples_bad


@pytest.mark.parametrize(
    "dataset_arguments",
    [AnomalyDatasetArguments(**{"train_samples": 10, "val_samples": (5, 5), "test_samples": (5, 5)})],
)
def test_anomaly_datamodule_full_phase_train(anomaly_dataset: anomaly_dataset):
    data_path, arguments = anomaly_dataset
    datamodule = AnomalyDataModule(data_path=data_path, phase="train")
    datamodule.prepare_data()
    datamodule.setup("fit")

    _check_total_samples(datamodule.data, arguments)


@pytest.mark.parametrize(
    "dataset_arguments",
    [AnomalyDatasetArguments(**{"train_samples": 0, "val_samples": (0, 0), "test_samples": (0, 0)})],
)
def test_anomaly_datamodule_empty(anomaly_dataset: anomaly_dataset):
    data_path, _ = anomaly_dataset
    datamodule = AnomalyDataModule(data_path=data_path)

    with pytest.raises(RuntimeError):
        datamodule.prepare_data()


@pytest.mark.parametrize(
    "dataset_arguments",
    [AnomalyDatasetArguments(**{"train_samples": 10, "val_samples": (0, 0), "test_samples": (5, 5)})],
)
def test_anomaly_datamodule_no_val_phase_train(anomaly_dataset: anomaly_dataset):
    data_path, arguments = anomaly_dataset
    datamodule = AnomalyDataModule(data_path=data_path)
    datamodule.prepare_data()
    datamodule.setup("fit")

    _check_total_samples(datamodule.data, arguments)
    assert len(datamodule.val_dataset) == sum(arguments.test_samples)
    assert not hasattr(datamodule, "test_dataset")

    # If there is no val data, test data is used for validation
    assert len(datamodule.val_dataloader()) > 0


@pytest.mark.parametrize(
    "dataset_arguments",
    [AnomalyDatasetArguments(**{"train_samples": 10, "val_samples": (5, 5), "test_samples": (5, 5)})],
)
def test_anomaly_datamodule_full_phase_test(anomaly_dataset: anomaly_dataset):
    data_path, arguments = anomaly_dataset
    datamodule = AnomalyDataModule(data_path=data_path, phase="test")
    datamodule.prepare_data()
    datamodule.setup("test")

    _check_total_samples(datamodule.data, arguments)

    assert not hasattr(datamodule, "train_dataset")
    assert not hasattr(datamodule, "val_dataset")
    assert len(datamodule.test_dataloader()) > 0
