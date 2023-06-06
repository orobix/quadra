# pylint: disable=redefined-outer-name
import os
import shutil
from pathlib import Path

import pytest

from quadra.utils.tests.fixtures import (
    base_classification_dataset,
    base_multilabel_classification_dataset,
    base_patch_classification_dataset,
)
from quadra.utils.tests.helpers import execute_quadra_experiment

BASE_EXPERIMENT_OVERRIDES = [
    "datamodule.num_workers=1",
    "logger=csv",
]


def test_train_sklearn_classification(tmp_path: Path, base_classification_dataset: base_classification_dataset):
    data_path, _ = base_classification_dataset

    overrides = [
        "experiment=base/classification/sklearn_classification",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
        "task.device=cpu",
    ] + BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)


def test_inference_sklearn_classification(tmp_path: Path, base_classification_dataset: base_classification_dataset):
    data_path, _ = base_classification_dataset

    train_overrides = [
        "experiment=base/classification/sklearn_classification",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
        "task.device=cpu",
    ] + BASE_EXPERIMENT_OVERRIDES

    train_path = tmp_path / "train"
    test_path = tmp_path / "test"
    train_path.mkdir()
    test_path.mkdir()

    execute_quadra_experiment(overrides=train_overrides, experiment_path=train_path)

    inference_overrides = [
        "experiment=base/classification/sklearn_classification_test",
        f"datamodule.data_path={data_path}",
        f"task.experiment_path={train_path}",
        "backbone=resnet18",
        "task.device=cpu",
    ] + BASE_EXPERIMENT_OVERRIDES
    execute_quadra_experiment(overrides=inference_overrides, experiment_path=test_path)

    shutil.rmtree(tmp_path)


def test_train_patches(tmp_path: Path, base_patch_classification_dataset):
    data_path, _, class_to_idx = base_patch_classification_dataset

    class_to_idx_parameter = str(class_to_idx).replace(
        "'", ""
    )  # Remove single quotes so that it can be parsed by hydra

    overrides = [
        "experiment=base/classification/sklearn_classification_patch",
        f"datamodule.data_path={data_path}",
        f"datamodule.class_to_idx={class_to_idx_parameter}",
        "trainer.iteration_over_training=1",
        "backbone=resnet18",
        "task.device=cpu",
    ] + BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)


def test_inference_patches(tmp_path: Path, base_patch_classification_dataset: base_patch_classification_dataset):
    data_path, _, class_to_idx = base_patch_classification_dataset

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
        "task.device=cpu",
    ] + BASE_EXPERIMENT_OVERRIDES
    execute_quadra_experiment(overrides=train_overrides, experiment_path=train_experiment_path)

    test_overrides = [
        "experiment=base/classification/sklearn_classification_patch_test",
        f"datamodule.data_path={data_path}",
        f"task.experiment_path={train_experiment_path}",
        f"backbone={backbone}",
        "task.device=cpu",
    ] + BASE_EXPERIMENT_OVERRIDES
    execute_quadra_experiment(overrides=test_overrides, experiment_path=test_experiment_path)

    shutil.rmtree(tmp_path)


def _run_inference_experiment(data_path: str, train_path: str, test_path: str):
    test_overrides = [
        "experiment=base/classification/classification_evaluation",
        f"datamodule.data_path={data_path}",
        "datamodule.num_workers=1",
        "datamodule.batch_size=16",
        "logger=csv",
        "task.device=cpu",
        f"task.model_path={os.path.join(train_path, 'deployment_model', 'model.pth')}",
    ]

    execute_quadra_experiment(overrides=test_overrides, experiment_path=test_path)


@pytest.mark.parametrize(
    "run_test, backbone, gradcam, freeze",
    [(True, "resnet18", True, False), (False, "resnet18", False, False), (True, "dino_vits8", False, True)],
)
def test_train_classification(
    tmp_path: Path,
    base_classification_dataset: base_classification_dataset,
    run_test: bool,
    backbone: str,
    gradcam: bool,
    freeze: bool,
):
    data_path, arguments = base_classification_dataset

    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    num_classes = len(arguments.samples)
    overrides = [
        "experiment=base/classification/classification",
        "trainer=lightning_cpu",
        "trainer.devices=1",
        f"datamodule.data_path={data_path}",
        f"model.num_classes={num_classes}",
        f"backbone={backbone}",
        f"backbone.model.freeze={freeze}",
        f"task.gradcam={gradcam}",
        "trainer.max_epochs=1",
        "task.report=True",
        f"task.run_test={run_test}",
    ] + BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=train_path)

    _run_inference_experiment(data_path, train_path, test_path)

    shutil.rmtree(tmp_path)


def test_train_multilabel_classification(
    tmp_path: Path, base_multilabel_classification_dataset: base_multilabel_classification_dataset
):
    data_path, arguments = base_multilabel_classification_dataset

    overrides = [
        "experiment=base/classification/multilabel_classification",
        "trainer=lightning_cpu",
        "trainer.devices=1",
        f"datamodule.data_path={data_path}",
        f"datamodule.images_and_labels_file={Path(data_path) / 'samples.txt'}",
        f"model.classifier.out_features={len(arguments.samples)}",
        "backbone=resnet18",
        "trainer.max_epochs=1",
    ] + BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)
