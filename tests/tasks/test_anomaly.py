# pylint: disable=redefined-outer-name
import os
import shutil
from pathlib import Path
from typing import List

import pytest

from quadra.utils.export import get_export_extension
from quadra.utils.tests.fixtures import base_anomaly_dataset
from quadra.utils.tests.helpers import check_deployment_model, execute_quadra_experiment, setup_trainer_for_lightning

try:
    import onnx  # noqa
    import onnxruntime  # noqa
    import onnxsim  # noqa

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

BASE_EXPERIMENT_OVERRIDES = [
    "trainer.devices=1",
    "datamodule.num_workers=1",
    "datamodule.train_batch_size=1",
    "datamodule.test_batch_size=1",
    "task.report=true",
    "logger=csv",
    "trainer.max_epochs=1",
    "~logger.mlflow",
]

BASE_EXPORT_TYPES = ["torchscript"] if not ONNX_AVAILABLE else ["torchscript", "onnx"]


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


def _run_inference_experiment(data_path: str, train_path: str, test_path: str, export_type: str):
    """Run an inference experiment for the given export type."""
    extension = get_export_extension(export_type)

    test_overrides = [
        "task.device=cpu",
        "experiment=base/anomaly/inference",
        f"datamodule.data_path={data_path}",
        "datamodule.num_workers=1",
        "datamodule.test_batch_size=32",
        "logger=csv",
        f"task.model_path={os.path.join(train_path, 'deployment_model', f'model.{extension}')}",
    ]

    execute_quadra_experiment(overrides=test_overrides, experiment_path=test_path)


def run_inference_experiments(data_path: str, train_path: str, test_path: str, export_types: List[str]):
    """Run inference experiments for the given export types."""
    for export_type in export_types:
        cwd = os.getcwd()
        check_deployment_model(export_type=export_type)

        _run_inference_experiment(
            data_path=data_path, train_path=train_path, test_path=test_path, export_type=export_type
        )

        # Change back to the original working directory
        os.chdir(cwd)


@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_padim(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    """Test the training and evaluation of the PADIM model."""
    data_path, _ = base_anomaly_dataset

    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    overrides = [
        "experiment=base/anomaly/padim",
        f"datamodule.data_path={data_path}",
        "model.model.backbone=resnet18",
        f"model.dataset.task={task}",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_report()

    run_inference_experiments(
        data_path=data_path, train_path=train_path, test_path=test_path, export_types=BASE_EXPORT_TYPES
    )

    shutil.rmtree(tmp_path)


@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_patchcore(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    """Test the training and evaluation of the PatchCore model."""
    data_path, _ = base_anomaly_dataset

    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    overrides = [
        "experiment=base/anomaly/patchcore",
        f"datamodule.data_path={data_path}",
        "model.model.backbone=resnet18",
        f"model.dataset.task={task}",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_report()

    run_inference_experiments(
        data_path=data_path, train_path=train_path, test_path=test_path, export_types=BASE_EXPORT_TYPES
    )

    shutil.rmtree(tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_cflow(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    """Test the training and evaluation of the cflow model."""
    data_path, _ = base_anomaly_dataset

    overrides = [
        "experiment=base/anomaly/cflow",
        f"datamodule.data_path={data_path}",
        "model.model.backbone=resnet18",
        f"model.dataset.task={task}",
        "export.types=[]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_report()

    # cflow does not support exporting to torchscript and onnx at the moment
    shutil.rmtree(tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_csflow(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    """Test the training and evaluation of the csflow model."""
    data_path, _ = base_anomaly_dataset
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    overrides = [
        "experiment=base/anomaly/csflow",
        f"datamodule.data_path={data_path}",
        f"model.dataset.task={task}",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_report()

    run_inference_experiments(
        data_path=data_path, train_path=train_path, test_path=test_path, export_types=BASE_EXPORT_TYPES
    )

    shutil.rmtree(tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_fastflow(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    """Test the training and evaluation of the fastflow model."""
    data_path, _ = base_anomaly_dataset
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    export_types = ["torchscript"]  # fastflow does not support exporting to onnx at the moment

    overrides = [
        "experiment=base/anomaly/fastflow",
        f"datamodule.data_path={data_path}",
        "model.model.backbone=resnet18",
        f"model.dataset.task={task}",
        f"export.types=[{','.join(export_types)}]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_report()

    run_inference_experiments(
        data_path=data_path, train_path=train_path, test_path=test_path, export_types=export_types
    )

    shutil.rmtree(tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_draem(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset, task: str):
    """Test the training and evaluation of the draem model."""
    data_path, _ = base_anomaly_dataset
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    overrides = [
        "experiment=base/anomaly/draem",
        f"datamodule.data_path={data_path}",
        f"model.dataset.task={task}",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    assert os.path.exists("checkpoints/final_model.ckpt")

    _check_report()

    run_inference_experiments(
        data_path=data_path, train_path=train_path, test_path=test_path, export_types=BASE_EXPORT_TYPES
    )

    shutil.rmtree(tmp_path)
