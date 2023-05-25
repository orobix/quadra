from .anomaly import AnomalibDetection
from .base import Evaluation, LightningTask, PlaceholderTask, Task
from .classification import Classification, ClassificationEvaluation, SklearnClassification, SklearnTestClassification
from .patch import PatchSklearnClassification, PatchSklearnTestClassification
from .segmentation import Segmentation, SegmentationAnalysisEvaluation, SegmentationEvaluation
from .ssl import SSL

__all__ = [
    "Task",
    "LightningTask",
    "Classification",
    "ClassificationEvaluation",
    "Segmentation",
    "SegmentationEvaluation",
    "SegmentationAnalysisEvaluation",
    "SSL",
    "AnomalibDetection",
    "SklearnClassification",
    "PatchSklearnClassification",
    "PlaceholderTask",
    "Evaluation",
    "SklearnTestClassification",
    "PatchSklearnTestClassification",
]
