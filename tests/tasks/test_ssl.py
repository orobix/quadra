# pylint: disable=redefined-outer-name
import shutil
from pathlib import Path

from quadra.utils.tests.fixtures import base_classification_dataset
from quadra.utils.tests.helpers import execute_quadra_experiment

BASE_EXPERIMENT_OVERRIDES = [
    "trainer=lightning_cpu",
    "trainer.devices=1",
    "trainer.max_epochs=1",
    "trainer.check_val_every_n_epoch=1",
    "datamodule=base/ssl",
    "model.classifier.n_neighbors=2",
    "model.classifier.n_jobs=1",
    "datamodule.num_workers=1",
    "datamodule.batch_size=32",
    "+trainer.limit_train_batches=1",
    "+trainer.limit_val_batches=1",
    "+trainer.limit_test_batches=1",
    "logger=csv",
]


def test_simsiam(tmp_path: Path, base_classification_dataset: base_classification_dataset):
    data_path, _ = base_classification_dataset

    overrides = [
        "experiment=base/ssl/simsiam",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
    ] + BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)


def test_dino(tmp_path: Path, base_classification_dataset: base_classification_dataset):
    data_path, _ = base_classification_dataset

    # Warmup set to 1 and max_epochs set to 2 to avoid issues with the loss initialization
    overrides = BASE_EXPERIMENT_OVERRIDES + [
        "experiment=base/ssl/dino",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
        "loss.warmup_teacher_temp_epochs=1",
        "trainer.max_epochs=2",
    ]

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)


def test_simclr(tmp_path: Path, base_classification_dataset: base_classification_dataset):
    data_path, _ = base_classification_dataset

    overrides = [
        "experiment=base/ssl/simclr",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
    ] + BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)


def test_byol(tmp_path: Path, base_classification_dataset: base_classification_dataset):
    data_path, _ = base_classification_dataset

    overrides = [
        "experiment=base/ssl/byol",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
    ] + BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)


def test_barlow(tmp_path: Path, base_classification_dataset: base_classification_dataset):
    data_path, _ = base_classification_dataset

    overrides = [
        "experiment=base/ssl/barlow",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
    ] + BASE_EXPERIMENT_OVERRIDES

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)
