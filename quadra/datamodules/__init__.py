from .anomaly import AnomalyDataModule
from .classification import (
    ClassificationDataModule,
    MultilabelClassificationDataModule,
    SklearnClassificationDataModule,
)
from .patch import PatchSklearnClassificationDataModule
from .segmentation import SegmentationDataModule, SegmentationMulticlassDataModule
from .ssl import SSLDataModule

__all__ = [
    "AnomalyDataModule",
    "ClassificationDataModule",
    "SklearnClassificationDataModule",
    "SegmentationDataModule",
    "SegmentationMulticlassDataModule",
    "PatchSklearnClassificationDataModule",
    "MultilabelClassificationDataModule",
    "SSLDataModule",
]
