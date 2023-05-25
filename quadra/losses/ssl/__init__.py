from .barlowtwins import BarlowTwinsLoss
from .byol import BYOLRegressionLoss
from .dino import DinoDistillationLoss
from .idmm import IDMMLoss
from .simclr import SimCLRLoss
from .simsiam import SimSIAMLoss
from .vicreg import VICRegLoss

__all__ = [
    "BarlowTwinsLoss",
    "BYOLRegressionLoss",
    "IDMMLoss",
    "SimCLRLoss",
    "SimSIAMLoss",
    "VICRegLoss",
    "DinoDistillationLoss",
]
