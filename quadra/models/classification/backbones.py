from __future__ import annotations

from typing import Any

import timm
import torch
from timm.models.helpers import load_checkpoint
from torch import nn
from torchvision import models

from quadra.models.classification.base import BaseNetworkBuilder
from quadra.utils.logger import get_logger

log = get_logger(__name__)


class TorchHubNetworkBuilder(BaseNetworkBuilder):
    """TorchHub feature extractor, with the possibility to map features to an hypersphere.

    Args:
        repo_or_dir: The name of the repository or the path to the directory containing the model.
        model_name: The name of the model within the repository.
        pretrained: Whether to load the pretrained weights for the model.
        pre_classifier: Pre classifier as a torch.nn.Module. Defaults to nn.Identity() if None.
        classifier: Classifier as a torch.nn.Module. Defaults to nn.Identity() if None.
        freeze: Whether to freeze the feature extractor. Defaults to True.
        hyperspherical: Whether to map features to an hypersphere. Defaults to False.
        flatten_features: Whether to flatten the features before the pre_classifier. Defaults to True.
        checkpoint_path: Path to a checkpoint to load after the model is initialized. Defaults to None.
        **torch_hub_kwargs: Additional arguments to pass to torch.hub.load
    """

    def __init__(
        self,
        repo_or_dir: str,
        model_name: str,
        pretrained: bool = True,
        pre_classifier: nn.Module | None = None,
        classifier: nn.Module | None = None,
        freeze: bool = True,
        hyperspherical: bool = False,
        flatten_features: bool = True,
        checkpoint_path: str | None = None,
        **torch_hub_kwargs: Any,
    ):
        self.pretrained = pretrained
        features_extractor = torch.hub.load(
            repo_or_dir=repo_or_dir, model=model_name, pretrained=self.pretrained, **torch_hub_kwargs
        )
        if checkpoint_path:
            log.info("Loading checkpoint from %s", checkpoint_path)
            load_checkpoint(model=features_extractor, checkpoint_path=checkpoint_path)

        super().__init__(
            features_extractor=features_extractor,
            pre_classifier=pre_classifier,
            classifier=classifier,
            freeze=freeze,
            hyperspherical=hyperspherical,
            flatten_features=flatten_features,
        )


class TorchVisionNetworkBuilder(BaseNetworkBuilder):
    """Torchvision feature extractor, with the possibility to map features to an hypersphere.

    Args:
        model_name: Torchvision model function that will be evaluated, for example: torchvision.models.resnet18.
        pretrained: Whether to load the pretrained weights for the model.
        pre_classifier: Pre classifier as a torch.nn.Module. Defaults to nn.Identity() if None.
        classifier: Classifier as a torch.nn.Module. Defaults to nn.Identity() if None.
        freeze: Whether to freeze the feature extractor. Defaults to True.
        hyperspherical: Whether to map features to an hypersphere. Defaults to False.
        flatten_features: Whether to flatten the features before the pre_classifier. Defaults to True.
        checkpoint_path: Path to a checkpoint to load after the model is initialized. Defaults to None.
        **torchvision_kwargs: Additional arguments to pass to the model function.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        pre_classifier: nn.Module | None = None,
        classifier: nn.Module | None = None,
        freeze: bool = True,
        hyperspherical: bool = False,
        flatten_features: bool = True,
        checkpoint_path: str | None = None,
        **torchvision_kwargs: Any,
    ):
        self.pretrained = pretrained
        model_function = models.__dict__[model_name]
        features_extractor = model_function(pretrained=self.pretrained, progress=True, **torchvision_kwargs)
        if checkpoint_path:
            log.info("Loading checkpoint from %s", checkpoint_path)
            load_checkpoint(model=features_extractor, checkpoint_path=checkpoint_path)

        # Remove classifier
        features_extractor.classifier = nn.Identity()
        super().__init__(
            features_extractor=features_extractor,
            pre_classifier=pre_classifier,
            classifier=classifier,
            freeze=freeze,
            hyperspherical=hyperspherical,
            flatten_features=flatten_features,
        )


class TimmNetworkBuilder(BaseNetworkBuilder):
    """Torchvision feature extractor, with the possibility to map features to an hypersphere.

    Args:
        model_name: Timm model name
        pretrained: Whether to load the pretrained weights for the model.
        pre_classifier: Pre classifier as a torch.nn.Module. Defaults to nn.Identity() if None.
        classifier: Classifier as a torch.nn.Module. Defaults to nn.Identity() if None.
        freeze: Whether to freeze the feature extractor. Defaults to True.
        hyperspherical: Whether to map features to an hypersphere. Defaults to False.
        flatten_features: Whether to flatten the features before the pre_classifier. Defaults to True.
        checkpoint_path: Path to a checkpoint to load after the model is initialized. Defaults to None.
        **timm_kwargs: Additional arguments to pass to timm.create_model
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        pre_classifier: nn.Module | None = None,
        classifier: nn.Module | None = None,
        freeze: bool = True,
        hyperspherical: bool = False,
        flatten_features: bool = True,
        checkpoint_path: str | None = None,
        **timm_kwargs: Any,
    ):
        self.pretrained = pretrained
        features_extractor = timm.create_model(
            model_name, pretrained=self.pretrained, num_classes=0, checkpoint_path=checkpoint_path, **timm_kwargs
        )

        super().__init__(
            features_extractor=features_extractor,
            pre_classifier=pre_classifier,
            classifier=classifier,
            freeze=freeze,
            hyperspherical=hyperspherical,
            flatten_features=flatten_features,
        )
