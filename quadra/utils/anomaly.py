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
from pydantic import BaseModel
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

# https://github.com/python/cpython/issues/90015#issuecomment-1172996118
MapOrValue: TypeAlias = "float | torch.Tensor | np.ndarray"


class EvalThreshold(BaseModel):
    """Pair of raw and normalized threshold values used for consistency enforcement.

    Attributes:
        raw: The unnormalized threshold.
        normalized: The corresponding normalized threshold.
    """

    raw: float
    normalized: float


def ensure_scores_consistency(
    normalized_score: MapOrValue,
    raw_score: MapOrValue,
    eval_threshold: EvalThreshold,
) -> MapOrValue:
    """Enforce that the classification based on normalized scores matches the raw classification.

    For every sample, if `raw_score >= eval_threshold.raw` (anomaly), the normalized score is
    clipped to be at least `eval_threshold.normalized`. If `raw_score < eval_threshold.raw`
    (normal), the normalized score is clipped to be strictly below `eval_threshold.normalized`
    using `np.nextafter` so that no hard-coded epsilon is required.

    Args:
        normalized_score: Normalized anomaly score value or map to adjust.
        raw_score: Original (unnormalized) anomaly score used to determine the ground-truth
            classification for each sample.
        eval_threshold: Threshold pair defining the decision boundary in both spaces.

    Returns:
        Normalized score with consistent predictions.
    """
    score = raw_score
    if isinstance(score, torch.Tensor):
        score = score.cpu().numpy()

    boundary = eval_threshold.normalized
    is_anomaly_mask = score >= eval_threshold.raw
    is_not_anomaly_mask = np.bitwise_not(is_anomaly_mask)

    below_boundary: torch.Tensor | np.ndarray
    anomaly_boundary: torch.Tensor | np.ndarray
    if isinstance(normalized_score, torch.Tensor):
        device = normalized_score.device
        # Work in scores dtype, cast boundaries to the same dype to ensure that casts take effect
        _inf = torch.tensor(float("inf"), dtype=normalized_score.dtype, device=device)
        anomaly_boundary = torch.tensor(boundary, dtype=normalized_score.dtype, device=device)
        if float(anomaly_boundary) < boundary:
            anomaly_boundary = torch.nextafter(anomaly_boundary, _inf)
        below_boundary = torch.nextafter(torch.tensor(boundary, dtype=normalized_score.dtype, device=device), -_inf)

        if normalized_score.dim() == 0:
            normalized_score = (
                normalized_score.clamp(min=anomaly_boundary)
                if is_anomaly_mask
                else normalized_score.clamp(max=below_boundary)
            )
        else:
            normalized_score[is_anomaly_mask] = normalized_score[is_anomaly_mask].clamp(min=anomaly_boundary)
            normalized_score[is_not_anomaly_mask] = normalized_score[is_not_anomaly_mask].clamp(max=below_boundary)
    elif isinstance(normalized_score, np.ndarray) or np.isscalar(normalized_score):
        # Work in scores dtype, cast boundaries to the same dype to ensure that casts take effect
        dtype = normalized_score.dtype if isinstance(normalized_score, np.ndarray) else np.float64
        anomaly_boundary = np.array(boundary, dtype=dtype)
        if float(anomaly_boundary) < boundary:
            anomaly_boundary = np.nextafter(anomaly_boundary, np.array(np.inf, dtype=dtype))
        below_boundary = np.nextafter(np.array(boundary, dtype=dtype), np.array(-np.inf, dtype=dtype))

        if np.isscalar(normalized_score) or normalized_score.ndim == 0:  # type: ignore[union-attr]
            normalized_score = (
                np.clip(normalized_score, a_min=anomaly_boundary, a_max=None)
                if is_anomaly_mask
                else np.clip(normalized_score, a_min=None, a_max=below_boundary)
            )
        else:
            normalized_score = cast(np.ndarray, normalized_score)
            normalized_score[is_anomaly_mask] = np.clip(
                normalized_score[is_anomaly_mask], a_min=anomaly_boundary, a_max=None
            )
            normalized_score[is_not_anomaly_mask] = np.clip(
                normalized_score[is_not_anomaly_mask], a_min=None, a_max=below_boundary
            )

    return normalized_score


def normalize_anomaly_score(
    raw_score: MapOrValue,
    threshold: float,
    eval_threshold: EvalThreshold | None = None,
) -> MapOrValue:
    """Normalize anomaly score value or map based on threshold.

    The training threshold maps to 100.0 in normalized space. After the linear scaling,
    `ensure_scores_consistency` is called to guarantee that every sample's normalized
    classification matches its raw classification.

    Args:
        raw_score: Raw anomaly score value or map.
        threshold: Threshold for anomaly detection, usually it is the training threshold.
        eval_threshold: Threshold used during evaluation. It is used for ensure consistency of raw scores
            and normalized scores. When `None`, an `EvalThreshold` with `raw=threshold` and `normalized=100.0` is used,
            which reproduces the original behaviour for the training-threshold case.

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

    _eval_threshold = eval_threshold if eval_threshold is not None else EvalThreshold(raw=threshold, normalized=100.0)
    normalized_score = ensure_scores_consistency(normalized_score, raw_score, _eval_threshold)

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
