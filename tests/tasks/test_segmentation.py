# pylint: disable=redefined-outer-name
import shutil
from pathlib import Path

import pytest

from quadra.utils.tests.fixtures import base_binary_segmentation_dataset, base_multiclass_segmentation_dataset
from quadra.utils.tests.helpers import execute_quadra_experiment

# TODO: NEED TO ADD SEGMENTATION EVALUATION TESTS

BASE_EXPERIMENT_OVERRIDES = [
    "trainer=lightning_cpu",
    "trainer.devices=1",
    "trainer.max_epochs=1",
    "datamodule.num_workers=1",
    "datamodule.batch_size=32",
    "+trainer.limit_train_batches=1",
    "+trainer.limit_val_batches=1",
    "+trainer.limit_test_batches=1",
    "logger=csv",
    "task.evaluate.analysis=false",
]


@pytest.mark.parametrize("generate_report", [True, False])
def test_smp_binary(
    tmp_path: Path, base_binary_segmentation_dataset: base_binary_segmentation_dataset, generate_report: bool
):
    data_path, _, _ = base_binary_segmentation_dataset

    overrides = [
        "experiment=base/segmentation/smp",
        f"datamodule.data_path={data_path}",
        f"task.report={generate_report}",
    ]
    overrides += BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)


def test_smp_multiclass(tmp_path: Path, base_multiclass_segmentation_dataset: base_multiclass_segmentation_dataset):
    data_path, _, class_to_idx = base_multiclass_segmentation_dataset
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    idx_to_class_parameter = str(idx_to_class).replace(
        "'", ""
    )  # Remove single quotes so that it can be parsed by hydra

    overrides = [
        "experiment=base/segmentation/smp_multiclass",
        f"datamodule.data_path={data_path}",
        f"datamodule.idx_to_class={idx_to_class_parameter}",
    ]
    overrides += BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)


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
    ]
    overrides += BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)
