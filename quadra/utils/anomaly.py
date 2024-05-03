"""Anomaly Score Normalization Callback that uses min-max normalization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
