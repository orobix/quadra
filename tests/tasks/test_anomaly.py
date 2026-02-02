# pylint: disable=redefined-outer-name
from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from quadra.utils.export import get_export_extension
from quadra.utils.tests.fixtures import base_anomaly_dataset, imagenette_dataset
from quadra.utils.tests.fixtures.models.anomaly import _initialize_patchcore_model
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


def run_inference_experiments(data_path: str, train_path: str, test_path: str, export_types: list[str]):
    """Run inference experiments for the given export types."""
    for export_type in export_types:
        cwd = os.getcwd()
        check_deployment_model(export_type=export_type)

        _run_inference_experiment(
            data_path=data_path, train_path=train_path, test_path=test_path, export_type=export_type
        )

        # Change back to the original working directory
        os.chdir(cwd)


@pytest.mark.usefixtures("mock_training")
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
        "export.input_shapes=[[3,224,224]]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides
    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    _check_report()

    run_inference_experiments(
        data_path=data_path, train_path=train_path, test_path=test_path, export_types=BASE_EXPORT_TYPES
    )

    shutil.rmtree(tmp_path)


@pytest.fixture
def mock_patchcore_training(pytestconfig: pytest.Config, mocker: Callable[..., Generator[MockerFixture, None, None]]):
    def setup_patchcore_model(self):
        self.module.model = _initialize_patchcore_model(self.module.model)

    if pytestconfig.getoption("mock_training"):
        mocker.patch("quadra.tasks.base.LightningTask.train", setup_patchcore_model)


@pytest.mark.usefixtures("mock_patchcore_training")
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

    _check_report()

    run_inference_experiments(
        data_path=data_path, train_path=train_path, test_path=test_path, export_types=BASE_EXPORT_TYPES
    )

    shutil.rmtree(tmp_path)


@pytest.mark.usefixtures("mock_training")
@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_efficientad(
    tmp_path: Path,
    base_anomaly_dataset: base_anomaly_dataset,
    imagenette_dataset: imagenette_dataset,
    task: str,
):
    """Test the training and evaluation of the EfficientAD model."""
    data_path, _ = base_anomaly_dataset
    imagenette_path = imagenette_dataset

    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    overrides = [
        "experiment=base/anomaly/efficient_ad",
        f"datamodule.data_path={data_path}",
        "transforms.input_height=256",
        "transforms.input_width=256",
        "model.model.train_batch_size=1",
        "datamodule.test_batch_size=1",
        "model.model.input_size=[256, 256]",
        "trainer.check_val_every_n_epoch= ${trainer.max_epochs}",
        f"model.model.imagenette_dir= {imagenette_path}",
        f"model.dataset.task={task}",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
        "export.input_shapes=[[3,256,256],[3,256,256]]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    _check_report()

    run_inference_experiments(
        data_path=data_path, train_path=train_path, test_path=test_path, export_types=BASE_EXPORT_TYPES
    )

    shutil.rmtree(tmp_path)


@pytest.mark.usefixtures("mock_training")
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

    _check_report()

    # cflow does not support exporting to torchscript and onnx at the moment
    shutil.rmtree(tmp_path)


def test_custom_normalized_threshold(tmp_path: Path, base_anomaly_dataset: base_anomaly_dataset):
    """Test that custom_normalized_threshold parameter works correctly."""
    from omegaconf import OmegaConf
    from quadra.tasks.anomaly import AnomalibEvaluation
    import json

    # Create a mock model with known threshold
    model_path = tmp_path / "model"
    model_path.mkdir()
    
    # Create a mock model.json with known threshold
    model_data = {
        "image_threshold": 20.0,  # Training threshold is 20
        "pixel_threshold": 20.0,
        "anomaly_method": "test"
    }
    
    with open(model_path / "model.json", "w") as f:
        json.dump(model_data, f)
    
    # Create a minimal config
    config = OmegaConf.create({
        "datamodule": {
            "_target_": "quadra.datamodules.AnomalyDataModule",
        }
    })
    
    # Test 1: Custom normalized threshold of 110 should convert to unnormalized 22
    # normalized = (raw / training) * 100
    # 110 = (raw / 20) * 100 => raw = 22
    task = AnomalibEvaluation(
        config=config,
        model_path=str(model_path),
        custom_normalized_threshold=110.0,
        training_threshold_type="image"
    )
    
    assert task.custom_normalized_threshold == 110.0
    
    # Test 2: Verify that invalid threshold raises error
    with pytest.raises(ValueError, match="Custom normalized threshold must be greater than 0"):
        AnomalibEvaluation(
            config=config,
            model_path=str(model_path),
            custom_normalized_threshold=-10.0,
            training_threshold_type="image"
        )
    
    # Test 3: Verify that zero threshold raises error
    with pytest.raises(ValueError, match="Custom normalized threshold must be greater than 0"):
        AnomalibEvaluation(
            config=config,
            model_path=str(model_path),
            custom_normalized_threshold=0.0,
            training_threshold_type="image"
        )


# TODO: This test seems to crash on not so powerful machines
@pytest.mark.slow
@pytest.mark.usefixtures("mock_training")
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

    _check_report()

    run_inference_experiments(
        data_path=data_path, train_path=train_path, test_path=test_path, export_types=BASE_EXPORT_TYPES
    )

    shutil.rmtree(tmp_path)


@pytest.mark.usefixtures("mock_training")
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
        "export.input_shapes=[[3,224,224]]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    _check_report()

    run_inference_experiments(
        data_path=data_path, train_path=train_path, test_path=test_path, export_types=export_types
    )

    shutil.rmtree(tmp_path)


@pytest.mark.skip(
    reason="This test requires anomalib with imgaug installed which we don't want to include in the dependencies as it "
    "requires the non headless version of opencv-python"
)
@pytest.mark.slow
@pytest.mark.usefixtures("mock_training")
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

    _check_report()

    run_inference_experiments(
        data_path=data_path, train_path=train_path, test_path=test_path, export_types=BASE_EXPORT_TYPES
    )

    shutil.rmtree(tmp_path)
