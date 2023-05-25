# pylint: disable=redefined-outer-name
import os
import shutil
from pathlib import Path

import pytest

from quadra.utils.tests.fixtures import base_anomaly_dataset
from quadra.utils.tests.helpers import execute_quadra_experiment

BASE_EXPERIMENT_OVERRIDES = [
    "trainer=lightning_cpu",
    "trainer.devices=1",
    "datamodule.num_workers=1",
    "datamodule.train_batch_size=1",
    "datamodule.test_batch_size=1",
    "task.report=true",
    "logger=csv",
    "trainer.max_epochs=1",
    "~logger.mlflow",
]


def _check_deployment_model(invert: bool = False):
    """Check that the runtime model is present and valid.

    Args:
        invert: If true check that the runtime model is not present.
    """
    if invert:
        assert not os.path.exists("deployment_model/model.pt")
        assert not os.path.exists("deployment_model/model.json")
    else:
        assert os.path.exists("deployment_model/model.pt")
        assert os.path.exists("deployment_model/model.json")


def _check_report(invert: bool = False):
    """Check that the report is present and valid.

    Args:
        invert: If true check that the report is not present.
    """
    if invert:
        assert not os.path.exists("cumulative_histogram.png")
        assert not os.path.exists("test_confusion_matrix.png")
        assert not os.path.exists("avg_score_by_label.csv")
    else:
        assert os.path.exists("cumulative_histogram.png")
        assert os.path.exists("test_confusion_matrix.png")
        assert os.path.exists("avg_score_by_label.csv")


def _run_inference_experiment(data_path: str, train_path: str, test_path: str):
    test_overrides = [
        "task.device=cpu",
        "experiment=base/anomaly/inference",
        f"datamodule.data_path={data_path}",
        "datamodule.num_workers=1",
        "datamodule.test_batch_size=32",
        "logger=csv",
        f"task.model_path={os.path.join(train_path, 'deployment_model', 'model.pt')}",
    ]

    execute_quadra_experiment(overrides=test_overrides, experiment_path=test_path)


@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_padim_training(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    data_path, _ = base_anomaly_dataset

    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    overrides = [
        "experiment=base/anomaly/padim",
        f"datamodule.data_path={data_path}",
        "model.model.backbone=resnet18",
        f"model.dataset.task={task}",
        "task.export_type=[torchscript]",
    ]
    overrides += BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_deployment_model()
    _check_report()

    _run_inference_experiment(data_path, train_path, test_path)

    shutil.rmtree(tmp_path)


@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_patchcore_training(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    data_path, _ = base_anomaly_dataset

    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    overrides = [
        "experiment=base/anomaly/patchcore",
        f"datamodule.data_path={data_path}",
        "model.model.backbone=resnet18",
        f"model.dataset.task={task}",
        "task.export_type=[torchscript]",
    ]
    overrides += BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_deployment_model()
    _check_report()

    _run_inference_experiment(data_path, train_path, test_path)

    shutil.rmtree(tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_cflow_training(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    data_path, _ = base_anomaly_dataset

    overrides = [
        "experiment=base/anomaly/cflow",
        f"datamodule.data_path={data_path}",
        "model.model.backbone=resnet18",
        f"model.dataset.task={task}",
        "task.export_type=null",
    ]
    overrides += BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_deployment_model(invert=True)  # cflow does not support runtime model
    _check_report()

    # cflow does not support inference with jitted model
    shutil.rmtree(tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_csflow_training(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    data_path, _ = base_anomaly_dataset
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    overrides = [
        "experiment=base/anomaly/csflow",
        f"datamodule.data_path={data_path}",
        f"model.dataset.task={task}",
        "task.export_type=[torchscript]",
    ]
    overrides += BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_deployment_model()
    _check_report()

    _run_inference_experiment(data_path, train_path, test_path)

    shutil.rmtree(tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_fastflow_training(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    data_path, _ = base_anomaly_dataset
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    overrides = [
        "experiment=base/anomaly/fastflow",
        f"datamodule.data_path={data_path}",
        "model.model.backbone=resnet18",
        f"model.dataset.task={task}",
        "task.export_type=[torchscript]",
    ]
    overrides += BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_deployment_model()
    _check_report()

    _run_inference_experiment(data_path, train_path, test_path)

    shutil.rmtree(tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_draem_training(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    data_path, _ = base_anomaly_dataset
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    overrides = [
        "experiment=base/anomaly/draem",
        f"datamodule.data_path={data_path}",
        f"model.dataset.task={task}",
        "task.export_type=[torchscript]",
    ]
    overrides += BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_deployment_model()
    _check_report()

    _run_inference_experiment(data_path, train_path, test_path)

    shutil.rmtree(tmp_path)
