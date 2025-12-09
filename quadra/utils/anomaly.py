"""Anomaly Score Normalization Callback that uses min-max normalization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

try:
    from typing import Any, TypeAlias
except ImportError:
    from typing import Any

    from typing_extensions import TypeAlias  # noqa


# MyPy wants TypeAlias, but pylint has problems dealing with it
import numpy as np  # pylint: disable=unused-import
import pytorch_lightning as pl
import torch  # pylint: disable=unused-import
from anomalib.models.components import AnomalyModule
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

# https://github.com/python/cpython/issues/90015#issuecomment-1172996118
MapOrValue: TypeAlias = "float | torch.Tensor | np.ndarray"


def normalize_anomaly_score(raw_score: MapOrValue, threshold: float) -> MapOrValue:
    """Normalize anomaly score value or map based on threshold.

    Args:
        raw_score: Raw anomaly score valure or map
        threshold: Threshold for anomaly detection

    Returns:
        Normalized anomaly score value or map clipped between 0 and 1000
    """
    if threshold > 0:
        normalized_score = (raw_score / threshold) * 100.0
    elif threshold == 0:
        # TODO: Is this the best way to handle this case?
        normalized_score = (raw_score + 1) * 100.0
    else:
        normalized_score = 200.0 - ((raw_score / threshold) * 100.0)

    # Ensures that the normalized scores are consistent with the raw scores
    # For all the items whose prediction changes after normalization, force the normalized score to be
    # consistent with the prediction made on the raw score by clipping the score:
    #   - to 100.0 if the prediction was "anomaly" on the raw score and "good" on the normalized score
    #   - to 99.99 if the prediction was "good" on the raw score and "anomaly" on the normalized score
    score = raw_score
    if isinstance(score, torch.Tensor):
        score = score.cpu().numpy()
    # Anomalib classify as anomaly if anomaly_score gte threshold
    is_anomaly_mask = score >= threshold
    is_not_anomaly_mask = np.bitwise_not(is_anomaly_mask)
    if isinstance(normalized_score, torch.Tensor):
        if normalized_score.dim() == 0:
            normalized_score = (
                normalized_score.clamp(min=100.0) if is_anomaly_mask else normalized_score.clamp(max=99.99)
            )
        else:
            normalized_score[is_anomaly_mask] = normalized_score[is_anomaly_mask].clamp(min=100.0)
            normalized_score[is_not_anomaly_mask] = normalized_score[is_not_anomaly_mask].clamp(max=99.99)
    elif isinstance(normalized_score, np.ndarray) or np.isscalar(normalized_score):
        if np.isscalar(normalized_score) or normalized_score.ndim == 0:  # type: ignore[union-attr]
            normalized_score = (
                np.clip(normalized_score, a_min=100.0, a_max=None)
                if is_anomaly_mask
                else np.clip(normalized_score, a_min=None, a_max=99.99)
            )
        else:
            normalized_score = cast(np.ndarray, normalized_score)
            normalized_score[is_anomaly_mask] = np.clip(normalized_score[is_anomaly_mask], a_min=100.0, a_max=None)
            normalized_score[is_not_anomaly_mask] = np.clip(
                normalized_score[is_not_anomaly_mask], a_min=None, a_max=99.99
            )

    if isinstance(normalized_score, torch.Tensor):
        return torch.clamp(normalized_score, 0.0, 1000.0)

    return np.clip(normalized_score, 0.0, 1000.0)


class ThresholdNormalizationCallback(Callback):
    """Callback that normalizes the image-level and pixel-level anomaly scores dividing by the threshold value.

    Args:
        threshold_type: Threshold used to normalize pixel level anomaly scores, either image or pixel (default)
    """

    def __init__(self, threshold_type: str = "pixel"):
        super().__init__()
        self.threshold_type = threshold_type

    def on_test_start(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the test begins."""
        del trainer  # `trainer` variable is not used.

        for metric in (pl_module.image_metrics, pl_module.pixel_metrics):
            if metric is not None:
                metric.set_threshold(100.0)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends, normalizes the predicted scores and anomaly maps."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        self._normalize_batch(outputs, pl_module)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the predict batch ends, normalizes the predicted scores and anomaly maps."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        self._normalize_batch(outputs, pl_module)

    def _normalize_batch(self, outputs, pl_module):
        """Normalize a batch of predictions."""
        image_threshold = pl_module.image_threshold.value.cpu()
        pixel_threshold = pl_module.pixel_threshold.value.cpu()
        outputs["pred_scores"] = normalize_anomaly_score(outputs["pred_scores"], image_threshold.item())

        threshold = pixel_threshold if self.threshold_type == "pixel" else image_threshold
        threshold = threshold.item()

        if "anomaly_maps" in outputs:
            outputs["anomaly_maps"] = normalize_anomaly_score(outputs["anomaly_maps"], threshold)

        if "box_scores" in outputs:
            outputs["box_scores"] = [normalize_anomaly_score(scores, threshold) for scores in outputs["box_scores"]]
