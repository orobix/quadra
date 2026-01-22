from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._base import ClassifierMixin
from torch.utils.data import DataLoader

from quadra.utils import utils
from quadra.utils.classification import get_results
from quadra.utils.models import get_feature

log = utils.get_logger(__name__)


class SklearnClassificationTrainer:
    """Class to configure and run a classification using torch for feature extraction and sklearn to fit a classifier.

    Args:
        input_shape: [H, W, C]
        random_state: seed to fix randomness
        classifier: classification model
        iteration_over_training: the number of iteration over training during feature extraction
        backbone: the feature extractor
    """

    def __init__(
        self,
        input_shape: list,
        backbone: torch.nn.Module,
        random_state: int = 42,
        classifier: ClassifierMixin = LogisticRegression,
        iteration_over_training: int = 1,
    ) -> None:
        super().__init__()

        try:
            self.classifier = classifier(max_iter=1e4, random_state=random_state)
        except Exception:
            self.classifier = classifier

        self.input_shape = input_shape
        self.random_state = random_state
        self.iteration_over_training = iteration_over_training
        self.backbone = backbone

    def change_backbone(self, backbone: torch.nn.Module):
        """Update feature extractor."""
        self.backbone = backbone
        self.backbone.eval()

    def change_classifier(self, classifier: ClassifierMixin):
        """Update classifier."""
        self.classifier = classifier

    def fit(
        self,
        train_dataloader: DataLoader | None = None,
        train_features: ndarray | None = None,
        train_labels: ndarray | None = None,
    ):
        """Fit classifier on training set."""
        # Extract feature
        if self.backbone is None:
            raise AssertionError("You must set a model before running execution")

        if train_dataloader is not None:  # train_features is None or train_labels is None:
            log.info("Extracting features from training set")
            train_features, train_labels, _ = get_feature(
                feature_extractor=self.backbone,
                dl=train_dataloader,
                iteration_over_training=self.iteration_over_training,
                gradcam=False,
            )
        else:
            log.info("Using cached features for training set")
            # With the current implementation cached features are not sorted
            # Even though it doesn't seem to change anything
            if train_features is None or train_labels is None:
                raise AssertionError("Train features and labels must be provided when using cached data")
            permuted_indices = np.random.RandomState(seed=self.random_state).permutation(train_features.shape[0])
            train_features = train_features[permuted_indices]
            train_labels = train_labels[permuted_indices]

        log.info("Fitting classifier on %d features", len(train_features))  # type: ignore[arg-type]
        self.classifier.fit(train_features, train_labels)

    def test(
        self,
        test_dataloader: DataLoader,
        test_labels: ndarray | None = None,
        test_features: ndarray | None = None,
        class_to_keep: list[int] | None = None,
        idx_to_class: dict[int, str] | None = None,
        predict_proba: bool = True,
        gradcam: bool = False,
    ) -> (
        tuple[str | dict, DataFrame, float, DataFrame, np.ndarray | None]
        | tuple[None, None, None, DataFrame, np.ndarray | None]
    ):
        """Test classifier on test set.

        Args:
            test_dataloader: Test dataloader
            test_labels: test labels
            test_features: Optional test features used when cache data is available
            class_to_keep: list of class to keep
            idx_to_class: dictionary mapping class index to class name
            predict_proba: if True, predict also probability for each test image
            gradcam: Whether to compute gradcam

        Returns:
            cl_rep: Classification report
            pd_cm: Confusion matrix dataframe
            accuracy: Test accuracy
            res: Test results
            cams: Gradcams
        """
        cams = None
        # Extract feature
        if test_features is None:
            log.info("Extracting features from test set")
            test_features, final_test_labels, cams = get_feature(
                feature_extractor=self.backbone,
                dl=test_dataloader,
                gradcam=gradcam,
                classifier=self.classifier,
                input_shape=(self.input_shape[2], self.input_shape[0], self.input_shape[1]),
            )
        else:
            if test_labels is None:
                raise ValueError("Test labels must be provided when using cached data")
            log.info("Using cached features for test set")
            final_test_labels = test_labels

        # Run classifier
        log.info("Predict classifier on test set")
        test_prediction_label = self.classifier.predict(test_features)
        if predict_proba:
            test_probability = self.classifier.predict_proba(test_features)
            test_probability = test_probability.max(axis=1)

        if class_to_keep is not None:
            if idx_to_class is None:
                raise ValueError("You must provide `idx_to_class` and `test_labels` when using `class_to_keep`")
            filtered_test_labels = [int(x) if idx_to_class[x] in class_to_keep else -1 for x in final_test_labels]
        else:
            filtered_test_labels = cast(list[int], final_test_labels.tolist())

        if not hasattr(test_dataloader.dataset, "x"):
            raise ValueError("Current dataset doesn't provide an `x` attribute")

        res = pd.DataFrame(
            {
                "sample": list(test_dataloader.dataset.x),
                "real_label": final_test_labels,
                "pred_label": test_prediction_label,
            }
        )

        if predict_proba:
            res["probability"] = test_probability

        if not all(t == -1 for t in filtered_test_labels):
            test_real_label_cm = np.array(filtered_test_labels)
            if cams is not None:
                cams = cams[test_real_label_cm != -1]  # TODO: Is class_to_keep still used?
            pred_labels_cm = np.array(test_prediction_label)[test_real_label_cm != -1]
            test_real_label_cm = test_real_label_cm[test_real_label_cm != -1].astype(pred_labels_cm.dtype)
            cl_rep, pd_cm, accuracy = get_results(test_real_label_cm, pred_labels_cm, idx_to_class)

            return cl_rep, pd_cm, accuracy, res, cams

        return None, None, None, res, cams
