# pylint: disable=redefined-outer-name
import shutil
from pathlib import Path

from quadra.utils.tests.fixtures import base_classification_dataset
from quadra.utils.tests.helpers import execute_quadra_experiment, setup_trainer_for_lightning

try:
    import onnx  # noqa
    import onnxruntime  # noqa
    import onnxsim  # noqa

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

BASE_EXPORT_TYPES = ["torchscript"] if not ONNX_AVAILABLE else ["torchscript", "onnx"]

BASE_EXPERIMENT_OVERRIDES = [
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
    f"export.types=[{','.join(BASE_EXPORT_TYPES)}]",
]


def test_simsiam(tmp_path: Path, base_classification_dataset: base_classification_dataset):
    data_path, _ = base_classification_dataset

    overrides = [
        "experiment=base/ssl/simsiam",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

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
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)


def test_byol(tmp_path: Path, base_classification_dataset: base_classification_dataset):
    data_path, _ = base_classification_dataset

    overrides = [
        "experiment=base/ssl/byol",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)


def test_barlow(tmp_path: Path, base_classification_dataset: base_classification_dataset):
    data_path, _ = base_classification_dataset

    overrides = [
        "experiment=base/ssl/barlow",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
    ]
    trainer_overrides = setup_trainer_for_lightning()
    overrides += BASE_EXPERIMENT_OVERRIDES
    overrides += trainer_overrides

    execute_quadra_experiment(overrides=overrides, experiment_path=tmp_path)

    shutil.rmtree(tmp_path)
