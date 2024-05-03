from __future__ import annotations

from torch import nn

from quadra.utils.models import L2Norm


class BaseNetworkBuilder(nn.Module):
    """Baseline Feature Extractor, with the possibility to map features to an hypersphere.
        If hypershperical is True the classifier is ignored.

    Args:
        features_extractor: Feature extractor as a toch.nn.Module.
        pre_classifier: Pre classifier as a torch.nn.Module. Defaults to nn.Identity() if None.
        classifier: Classifier as a torch.nn.Module. Defaults to nn.Identity() if None.
        freeze: Whether to freeze the feature extractor. Defaults to True.
        hyperspherical: Whether to map features to an hypersphere. Defaults to False.
        flatten_features: Whether to flatten the features before the pre_classifier. May be required if your model
            is outputting a feature map rather than a vector. Defaults to True.
    """

    def __init__(
        self,
        features_extractor: nn.Module,
        pre_classifier: nn.Module | None = None,
        classifier: nn.Module | None = None,
        freeze: bool = True,
        hyperspherical: bool = False,
        flatten_features: bool = True,
    ):
        super().__init__()
        if pre_classifier is None:
            pre_classifier = nn.Identity()

        if classifier is None:
            classifier = nn.Identity()

        self.features_extractor = features_extractor
        self.freeze = freeze
        self.hyperspherical = hyperspherical
        self.pre_classifier = pre_classifier
        self.classifier = classifier
        self.flatten: bool = False
        self._hyperspherical: bool = False
        self.l2: L2Norm | None = None
        self.flatten_features = flatten_features

        self.freeze = freeze
        self.hyperspherical = hyperspherical

        if self.freeze:
            for p in self.features_extractor.parameters():
                p.requires_grad = False

    @property
    def freeze(self) -> bool:
        """Whether to freeze the feature extractor."""
        return self._freeze

    @freeze.setter
    def freeze(self, value: bool) -> None:
        """Whether to freeze the feature extractor."""
        for p in self.features_extractor.parameters():
            p.requires_grad = not value

        self._freeze = value

    @property
    def hyperspherical(self) -> bool:
        """Whether to map the extracted features into an hypersphere."""
        return self._hyperspherical

    @hyperspherical.setter
    def hyperspherical(self, value: bool) -> None:
        """Whether to map the extracted features into an hypersphere."""
        self._hyperspherical = value
        self.l2 = L2Norm() if value else None

    def forward(self, x):
        x = self.features_extractor(x)

        if self.flatten_features:
            x = x.view(x.size(0), -1)

        x = self.pre_classifier(x)

        if self.hyperspherical:
            x = self.l2(x)

        x = self.classifier(x)

        return x
