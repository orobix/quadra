# pylint: disable=redefined-outer-name
import os
import shutil
from pathlib import Path
from typing import List

import pytest

from quadra.utils.export import get_export_extension
from quadra.utils.tests.fixtures import (
    base_classification_dataset,
    base_multilabel_classification_dataset,
    base_patch_classification_dataset,
)
from quadra.utils.tests.helpers import (
    check_deployment_model,
    execute_quadra_experiment,
    get_quadra_test_device,
    setup_trainer_for_lightning,
)

try:
    import onnx  # noqa
    import onnxruntime  # noqa
    import onnxsim  # noqa

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

BASE_EXPERIMENT_OVERRIDES = [
    "datamodule.num_workers=1",
    "logger=csv",
]

BASE_EXPORT_TYPES = ["pytorch", "torchscript"] if not ONNX_AVAILABLE else ["pytorch", "torchscript", "onnx"]


def _run_inference_experiment(
    test_overrides: List[str], data_path: str, train_path: str, test_path: str, export_type: str
):
    """Run an inference experiment for the given export type."""
    extension = get_export_extension(export_type)

    test_overrides.append(f"datamodule.data_path={data_path}")
    test_overrides.append(f"task.model_path={os.path.join(train_path, 'deployment_model', f'model.{extension}')}")

    execute_quadra_experiment(overrides=test_overrides, experiment_path=test_path)


def run_inference_experiments(
    test_overrides: List[str], data_path: str, train_path: str, test_path: str, export_types: List[str]
):
    """Run inference experiments for the given export types."""
    for export_type in export_types:
        cwd = os.getcwd()
        check_deployment_model(export_type=export_type)

        _run_inference_experiment(
            test_overrides=test_overrides,
            data_path=data_path,
            train_path=train_path,
            test_path=test_path,
            export_type=export_type,
        )

        # Change back to the original working directory
        os.chdir(cwd)


def test_sklearn_classification(tmp_path: Path, base_classification_dataset: base_classification_dataset):
    """Test the training and evaluation of a sklearn classification model."""
    data_path, _ = base_classification_dataset

    device = get_quadra_test_device()

    train_overrides = [
        "experiment=base/classification/sklearn_classification",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
        f"task.device={device}",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
    ] + BASE_EXPERIMENT_OVERRIDES

    train_path = tmp_path / "train"
    test_path = tmp_path / "test"
    train_path.mkdir()
    test_path.mkdir()

    execute_quadra_experiment(overrides=train_overrides, experiment_path=train_path)

    inference_overrides = [
        "experiment=base/classification/sklearn_classification_test",
        "backbone=resnet18",
        f"task.device={device}",
        "task.gradcam=true",
    ] + BASE_EXPERIMENT_OVERRIDES

    run_inference_experiments(
        test_overrides=inference_overrides,
        data_path=data_path,
        train_path=train_path,
        test_path=test_path,
        export_types=BASE_EXPORT_TYPES,
    )

    shutil.rmtree(tmp_path)


def test_sklearn_classification_patch(
    tmp_path: Path, base_patch_classification_dataset: base_patch_classification_dataset
):
    """Test the training and evaluation of a sklearn classification model with patches."""
    data_path, _, class_to_idx = base_patch_classification_dataset
    device = get_quadra_test_device()

    class_to_idx_parameter = str(class_to_idx).replace(
        "'", ""
    )  # Remove single quotes so that it can be parsed by hydra

    train_experiment_path = tmp_path / "train"
    test_experiment_path = tmp_path / "test"
    train_experiment_path.mkdir()
    test_experiment_path.mkdir()

    backbone = "resnet18"

    train_overrides = [
        "experiment=base/classification/sklearn_classification_patch",
        f"datamodule.data_path={data_path}",
        f"datamodule.class_to_idx={class_to_idx_parameter}",
        "trainer.iteration_over_training=1",
        f"backbone={backbone}",
        f"task.device={device}",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
    ] + BASE_EXPERIMENT_OVERRIDES
    execute_quadra_experiment(overrides=train_overrides, experiment_path=train_experiment_path)

    test_overrides = [
        "experiment=base/classification/sklearn_classification_patch_test",
        f"backbone={backbone}",
        "task.device=cpu",
    ] + BASE_EXPERIMENT_OVERRIDES

    run_inference_experiments(
        test_overrides=test_overrides,
        data_path=data_path,
        train_path=train_experiment_path,
        test_path=test_experiment_path,
        export_types=BASE_EXPORT_TYPES,
    )

    shutil.rmtree(tmp_path)


@pytest.mark.usefixtures("mock_training")
@pytest.mark.parametrize(
    "backbone, gradcam, freeze",
    [("resnet18", True, False), ("dino_vits8", False, True)],
)
def test_classification(
    tmp_path: Path,
    base_classification_dataset: base_classification_dataset,
    backbone: str,
    gradcam: bool,
    freeze: bool,
):
    """Test the training and evaluation of a torch based classification model."""
    data_path, arguments = base_classification_dataset

    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    num_classes = len(arguments.samples)

    overrides = [
        "experiment=base/classification/classification",
        "trainer.devices=1",
        f"datamodule.data_path={data_path}",
        f"model.num_classes={num_classes}",
        f"backbone={backbone}",
        f"backbone.model.freeze={freeze}",
        f"task.gradcam={gradcam}",
        "trainer.max_epochs=1",
        "task.report=True",
        f"task.run_test=true",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    test_overrides = [
        "experiment=base/classification/classification_evaluation",
        "datamodule.num_workers=1",
        "datamodule.batch_size=16",
        "logger=csv",
        "task.device=cpu",
    ]

    run_inference_experiments(
        test_overrides=test_overrides,
        data_path=data_path,
        train_path=train_path,
        test_path=test_path,
        export_types=BASE_EXPORT_TYPES,
    )

    shutil.rmtree(tmp_path)


def test_multilabel_classification(
    tmp_path: Path, base_multilabel_classification_dataset: base_multilabel_classification_dataset
):
    """Test the training and evaluation of a torch based multilabel classification model."""
    data_path, arguments = base_multilabel_classification_dataset

    overrides = [
        "experiment=base/classification/multilabel_classification",
        "trainer.devices=1",
        f"datamodule.data_path={data_path}",
        f"datamodule.num_classes={len(arguments.samples)}",
        f"datamodule.images_and_labels_file={Path(data_path) / 'samples.txt'}",
        "backbone=resnet18",
        "trainer.max_epochs=1",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)
