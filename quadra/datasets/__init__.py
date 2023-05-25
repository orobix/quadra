from .anomaly import AnomalyDataset
from .classification import ClassificationDataset, ImageClassificationListDataset, MultilabelClassificationDataset
from .patch import PatchSklearnClassificationTrainDataset
from .segmentation import SegmentationDataset, SegmentationDatasetMulticlass
from .ssl import TwoAugmentationDataset, TwoSetAugmentationDataset

__all__ = [
    "ImageClassificationListDataset",
    "ClassificationDataset",
    "SegmentationDataset",
    "SegmentationDatasetMulticlass",
    "PatchSklearnClassificationTrainDataset",
    "MultilabelClassificationDataset",
    "AnomalyDataset",
    "TwoAugmentationDataset",
    "TwoSetAugmentationDataset",
]
