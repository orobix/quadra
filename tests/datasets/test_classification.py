# pylint: disable=redefined-outer-name
import glob
import os
from collections import Counter

import numpy as np
import pytest
import torch

from quadra.datasets import (
    ImageClassificationListDataset,
    MultilabelClassificationDataset,
    PatchSklearnClassificationTrainDataset,
)
from quadra.utils.patch.dataset import load_train_file
from quadra.utils.tests.fixtures.dataset.classification import (
    base_classification_dataset,
    base_multilabel_classification_dataset,
    base_patch_classification_dataset,
)


def test_classification_dataset(base_classification_dataset: base_classification_dataset):
    data_path, _ = base_classification_dataset

    samples = glob.glob(os.path.join(data_path, "**", "*"))
    targets = [os.path.basename(os.path.dirname(s)) for s in samples]

    dataset = ImageClassificationListDataset(
        samples=samples,
        targets=targets,
        class_to_idx=None,
        allow_missing_label=False,
    )

    assert dataset.class_to_idx is not None
    assert len(dataset.class_to_idx) == 2

    for i, item in enumerate(dataset):
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], np.ndarray)
        assert isinstance(item[1], int)
        assert item[1] == dataset.class_to_idx[targets[i]]


def test_multilabel_classification_dataset(
    base_multilabel_classification_dataset: base_multilabel_classification_dataset,
):
    data_path, _ = base_multilabel_classification_dataset

    samples = glob.glob(os.path.join(data_path, "images", "*"))
    with open(os.path.join(data_path, "samples.txt"), "r") as f:
        samples_and_targets = [line.strip().split(",") for line in f.readlines()]
        samples_mapping = {os.path.basename(st[0]): st[1:] for st in samples_and_targets}

    targets = set([item for sublist in list(samples_mapping.values()) for item in sublist])

    class_to_idx = {c: i for i, c in enumerate(targets)}
    one_hot_encoding = np.zeros((len(samples), len(targets)))
    for i, sample in enumerate(samples):
        for target in samples_mapping[os.path.basename(sample)]:
            one_hot_encoding[i, class_to_idx[target]] = 1

    dataset = MultilabelClassificationDataset(
        samples=samples,
        targets=one_hot_encoding,
        class_to_idx=class_to_idx,
        transform=None,
    )

    for i, item in enumerate(dataset):
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], np.ndarray)
        assert isinstance(item[1], torch.Tensor)
        reverted_classes = set([dataset.idx_to_class[c.item()] for c in torch.where(item[1] == 1)[0]])
        assert reverted_classes == set(samples_mapping[os.path.basename(dataset.x[i])])


@pytest.mark.parametrize("balance_classes", [True, False])
def test_patch_classification_dataset(
    base_patch_classification_dataset: base_patch_classification_dataset, balance_classes: bool
):
    data_path, _, class_to_idx = base_patch_classification_dataset

    samples, targets = load_train_file(os.path.join(data_path, "train", "dataset.txt"))

    dataset = PatchSklearnClassificationTrainDataset(
        data_path=data_path,
        samples=samples,
        targets=targets,
        class_to_idx=class_to_idx,
        balance_classes=balance_classes,
    )

    for i, item in enumerate(dataset):
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], np.ndarray)
        assert isinstance(item[1], int)
        if not balance_classes:
            assert item[1] == dataset.class_to_idx[targets[i]]

    if balance_classes:
        most_frequent_target_count = Counter(targets).most_common(1)[0][1]
        assert len(dataset) == len(class_to_idx) * most_frequent_target_count
