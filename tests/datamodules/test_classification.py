import json
import os
import shutil

import numpy as np
import pytest

from quadra.datamodules import (
    ClassificationDataModule,
    MultilabelClassificationDataModule,
    PatchSklearnClassificationDataModule,
    SklearnClassificationDataModule,
)
from quadra.utils.patch.dataset import PatchDatasetInfo
from quadra.utils.tests.fixtures.dataset import (
    ClassificationDatasetArguments,
    ClassificationMultilabelDatasetArguments,
    ClassificationPatchDatasetArguments,
    classification_dataset,
    classification_patch_dataset,
    multilabel_classification_dataset,
)


@pytest.mark.parametrize(
    "dataset_arguments",
    [ClassificationDatasetArguments(**{"samples": [15, 20, 25], "classes": ["a", "b", "c"]})],
)
@pytest.mark.parametrize("val_size", [0.2, 0.5, 0.8])
def test_classification_data_module_holdout_phase_train(
    classification_dataset: classification_dataset, val_size: float
):
    data_path, arguments = classification_dataset
    datamodule = SklearnClassificationDataModule(data_path=data_path, val_size=val_size, n_splits=1, phase="train")
    datamodule.prepare_data()
    datamodule.setup("fit")

    n_samples = sum(arguments.samples)

    n_train = int(n_samples * (1 - val_size))
    n_val = n_samples - n_train

    cv_filter = datamodule.data["cv"] == 0
    train_split = datamodule.data["split"] == "train"
    val_split = datamodule.data["split"] == "val"
    val_samples = datamodule.data["samples"][cv_filter & val_split]
    train_samples = datamodule.data["samples"][cv_filter & train_split]
    assert (len(train_samples) - n_train) <= 1
    assert (len(val_samples) - n_val) <= 1
    shutil.rmtree(data_path)


@pytest.mark.parametrize(
    "dataset_arguments",
    [
        ClassificationDatasetArguments(
            **{"samples": [15, 20, 25], "classes": ["a", "b", "c"], "val_size": 0.2, "test_size": 0.1}
        ),
        ClassificationDatasetArguments(**{"samples": [15, 20, 25, 43], "val_size": 0.4, "test_size": 0.1}),
    ],
)
def test_classification_datamodule_with_splits_phase_test(classification_dataset: classification_dataset):
    data_path, _ = classification_dataset

    datamodule = SklearnClassificationDataModule(
        data_path=data_path,
        phase="test",
        train_split_file=os.path.join(data_path, "train.txt"),
        test_split_file=os.path.join(data_path, "test.txt"),
        n_splits=1,
        num_workers=1,
    )
    datamodule.prepare_data()
    datamodule.setup("test")

    with open(os.path.join(data_path, "test.txt"), "r") as f:
        test_samples_txt = f.read().splitlines()

    test_split = datamodule.data["split"] == "test"
    test_samples_df = datamodule.data["samples"][test_split]
    assert len(test_samples_df) == len(test_samples_txt)


@pytest.mark.parametrize(
    "dataset_arguments",
    [ClassificationDatasetArguments(**{"samples": [15, 20, 25], "classes": ["a", "b", "c"]})],
)
def test_classification_data_module_phase_test(classification_dataset: classification_dataset):
    data_path, arguments = classification_dataset
    datamodule = SklearnClassificationDataModule(data_path=data_path, phase="test", num_workers=1)
    datamodule.prepare_data()
    datamodule.setup("test")

    n_samples = sum(arguments.samples)

    test_split = datamodule.data["split"] == "test"
    test_samples_df = datamodule.data["samples"][test_split]
    assert len(test_samples_df) == n_samples
    shutil.rmtree(data_path)


@pytest.mark.parametrize(
    "dataset_arguments",
    [
        ClassificationPatchDatasetArguments(
            **{
                "samples": [15, 20, 25],
                "classes": ["a", "b", "c"],
                "patch_size": [32, 32],
                "overlap": 0.5,
                "val_size": 0.2,
                "test_size": 0.1,
            }
        ),
        ClassificationPatchDatasetArguments(
            **{
                "samples": [15, 20, 25],
                "classes": ["a", "b", "c"],
                "patch_number": [12, 18],
                "overlap": 0.5,
                "val_size": 0.2,
                "test_size": 0.1,
            }
        ),
    ],
)
def test_classification_patch_datamodule(classification_patch_dataset: classification_patch_dataset):
    data_path, _, class_to_idx = classification_patch_dataset
    datamodule = PatchSklearnClassificationDataModule(
        data_path=data_path,
        class_to_idx=class_to_idx,
        num_workers=1,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("test")

    with open(os.path.join(data_path, "info.json"), "r") as f:
        info = PatchDatasetInfo(**json.load(f))

    # train samples are named like imagename_class.h5
    train_samples_df = datamodule.train_data["samples"].tolist()
    datamodule_train_samples = set(
        [os.path.splitext(os.path.basename("_".join(s.split("_")[0:-1])))[0] for s in train_samples_df]
    )
    # val samples are named like imagename_patchnumber.xyz
    val_samples_df = datamodule.val_data["samples"].tolist()
    datamodule_val_samples = set(
        [os.path.splitext("_".join(os.path.basename(s).split("_")[0:-1]))[0] for s in val_samples_df]
    )
    # test samples are named like imagename_patchnumber.xyz and may contain #DISCARD# in the name
    test_samples_df = datamodule.test_data["samples"].tolist()
    datamodule_test_samples = set(
        [
            os.path.splitext("_".join(os.path.basename(s).replace("#DISCARD#", "").split("_")[0:-1]))[0]
            for s in test_samples_df
        ]
    )

    train_filenames = set([os.path.splitext(os.path.basename(s.image_path))[0] for s in info.train_files])
    val_filenames = set([os.path.splitext(os.path.basename(s.image_path))[0] for s in info.val_files])
    test_filenames = set([os.path.splitext(os.path.basename(s.image_path))[0] for s in info.test_files])

    assert datamodule_train_samples == train_filenames
    assert datamodule_val_samples == val_filenames
    assert datamodule_test_samples == test_filenames


@pytest.mark.parametrize(
    "dataset_arguments",
    [ClassificationDatasetArguments(**{"samples": [25, 35, 15, 25], "classes": ["d", "c", "b", "a"]})],
)
@pytest.mark.parametrize("val_size, test_size", [(0.1, 0.2), (0.2, 0.1)])
def test_new_classification_data_module_holdout_phase_train(
    classification_dataset: classification_dataset, val_size: float, test_size: float
):
    data_path, arguments = classification_dataset
    datamodule = ClassificationDataModule(data_path=data_path, val_size=val_size, test_size=test_size)
    datamodule.prepare_data()

    datamodule.setup("fit")
    datamodule.setup("test")
    # Verify that prepare_data builds class_to_idx if it isn't passed as parameter to the datamodule
    assert datamodule.class_to_idx == {"a": 0, "b": 1, "c": 2, "d": 3}

    # Verify train/test split
    test_samples_df = datamodule.data[datamodule.data["split"] == "test"]["samples"].tolist()
    test_targets_df = datamodule.data[datamodule.data["split"] == "test"]["targets"].tolist()

    assert len(test_samples_df) == test_size * sum(arguments.samples)
    assert len(test_targets_df) == test_size * sum(arguments.samples)

    # Verify remaining train / val split
    val_samples_df = datamodule.data[datamodule.data["split"] == "val"]["samples"].tolist()
    val_targets_df = datamodule.data[datamodule.data["split"] == "val"]["targets"].tolist()
    assert len(val_samples_df) == val_size * ((1 - test_size) * sum(arguments.samples))
    assert len(val_targets_df) == val_size * ((1 - test_size) * sum(arguments.samples))
    # Verify train dataset's getitem
    assert isinstance(datamodule.train_dataset[0][0], np.ndarray)
    assert isinstance(datamodule.train_dataset[0][1], int)

    shutil.rmtree(data_path)


@pytest.mark.parametrize(
    "dataset_arguments",
    [
        ClassificationMultilabelDatasetArguments(
            **{
                "samples": [15, 20, 25],
                "classes": ["a", "b", "c"],
                "val_size": 0.2,
                "test_size": 0.1,
                "percentage_other_classes": 0.5,
            }
        )
    ],
)
def test_multilabel_classification_datamodule(multilabel_classification_dataset: multilabel_classification_dataset):
    data_path, _ = multilabel_classification_dataset
    datamodule = MultilabelClassificationDataModule(
        data_path=data_path,
        num_workers=1,
        train_split_file=os.path.join(data_path, "train.txt"),
        val_split_file=os.path.join(data_path, "val.txt"),
        test_split_file=os.path.join(data_path, "test.txt"),
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("test")

    with open(os.path.join(data_path, "train.txt"), "r") as f:
        train_samples = f.read().splitlines()
        train_labels = [s.split(",")[1:] for s in train_samples]
        train_labels = np.array([datamodule.class_to_idx[x] for l in train_labels for x in l])

    with open(os.path.join(data_path, "val.txt"), "r") as f:
        val_samples = f.read().splitlines()
        val_labels = [s.split(",")[1:] for s in val_samples]
        val_labels = np.array([datamodule.class_to_idx[x] for l in val_labels for x in l])

    with open(os.path.join(data_path, "test.txt"), "r") as f:
        test_samples = f.read().splitlines()
        test_labels = [s.split(",")[1:] for s in test_samples]
        test_labels = np.array([datamodule.class_to_idx[x] for l in test_labels for x in l])

    # Verify that the one hot encoded labels count matches the number of labels in the split
    for split, labels in zip(["train", "val", "test"], [train_labels, val_labels, test_labels]):
        for l in np.unique(labels):
            train_targets = np.vstack(datamodule.data[datamodule.data["split"] == split]["targets"])
            assert train_targets[:, l].sum() == (labels == l).sum()

    shutil.rmtree(data_path)
