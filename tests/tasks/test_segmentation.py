# pylint: disable=redefined-outer-name
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from quadra.utils.export import get_export_extension
from quadra.utils.tests.fixtures import base_binary_segmentation_dataset, base_multiclass_segmentation_dataset
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
    "trainer.max_epochs=1",
    "datamodule.num_workers=1",
    "datamodule.batch_size=32",
    "+trainer.limit_train_batches=1",
    "+trainer.limit_val_batches=1",
    "+trainer.limit_test_batches=1",
    "logger=csv",
]

BASE_EXPORT_TYPES = ["torchscript"] if not ONNX_AVAILABLE else ["torchscript", "onnx"]


def _run_inference_experiment(
    test_overrides: list[str], data_path: str, train_path: str, test_path: str, export_type: str
):
    """Run an inference experiment for the given export type."""
    extension = get_export_extension(export_type)

    test_overrides.append(f"datamodule.data_path={data_path}")
    test_overrides.append(f"task.model_path={os.path.join(train_path, 'deployment_model', f'model.{extension}')}")

    execute_quadra_experiment(overrides=test_overrides, experiment_path=test_path)


def run_inference_experiments(
    test_overrides: list[str], data_path: str, train_path: str, test_path: str, export_types: list[str]
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


@pytest.mark.usefixtures("mock_training")
@pytest.mark.parametrize("generate_report", [True, False])
def test_smp_binary(
    tmp_path: Path,
    base_binary_segmentation_dataset: base_binary_segmentation_dataset,
    generate_report: bool,
):
    data_path, _, _ = base_binary_segmentation_dataset

    train_path = tmp_path / "train"
    test_path = tmp_path / "test"
    train_path.mkdir()
    test_path.mkdir()

    overrides = [
        "experiment=base/segmentation/smp",
        f"datamodule.data_path={data_path}",
        f"task.report={generate_report}",
        "task.evaluate.analysis=false",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    inference_overrides = [
        "experiment=base/segmentation/smp_evaluation",
        "task.device=cpu",
    ] + BASE_EXPERIMENT_OVERRIDES

    run_inference_experiments(
        test_overrides=inference_overrides,
        data_path=data_path,
        train_path=train_path,
        test_path=test_path,
        export_types=BASE_EXPORT_TYPES,
    )

    shutil.rmtree(tmp_path)


@pytest.mark.usefixtures("mock_training")
def test_smp_multiclass(tmp_path: Path, base_multiclass_segmentation_dataset: base_multiclass_segmentation_dataset):
    data_path, _, class_to_idx = base_multiclass_segmentation_dataset
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_path = tmp_path / "train"
    test_path = tmp_path / "test"
    train_path.mkdir()
    test_path.mkdir()

    idx_to_class_parameter = str(idx_to_class).replace(
        "'", ""
    )  # Remove single quotes so that it can be parsed by hydra

    overrides = [
        "experiment=base/segmentation/smp_multiclass",
        f"datamodule.data_path={data_path}",
        f"datamodule.idx_to_class={idx_to_class_parameter}",
        "task.evaluate.analysis=false",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    inference_overrides = [
        "experiment=base/segmentation/smp_multiclass_evaluation",
        f"datamodule.idx_to_class={idx_to_class_parameter}",
        "task.device=cpu",
    ] + BASE_EXPERIMENT_OVERRIDES

    run_inference_experiments(
        test_overrides=inference_overrides,
        data_path=data_path,
        train_path=train_path,
        test_path=test_path,
        export_types=BASE_EXPORT_TYPES,
    )

    shutil.rmtree(tmp_path)


@pytest.mark.usefixtures("mock_training")
def test_smp_multiclass_with_binary_dataset(
    tmp_path: Path, base_binary_segmentation_dataset: base_binary_segmentation_dataset
):
    data_path, _, class_to_idx = base_binary_segmentation_dataset
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    idx_to_class_parameter = str(idx_to_class).replace(
        "'", ""
    )  # Remove single quotes so that it can be parsed by hydra

    overrides = [
        "experiment=base/segmentation/smp_multiclass",
        f"datamodule.data_path={data_path}",
        f"datamodule.idx_to_class={idx_to_class_parameter}",
        "task.evaluate.analysis=false",
        f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)
