import pytest

from quadra.models.classification import TimmNetworkBuilder, TorchHubNetworkBuilder


@pytest.fixture
def resnet18():
    """Yield a resnet18 model."""
    yield TimmNetworkBuilder("resnet18", pretrained=False, freeze=True, exportable=True)


@pytest.fixture
def resnet50():
    """Yield a resnet50 model."""
    yield TimmNetworkBuilder("resnet50", pretrained=False, freeze=True, exportable=True)


@pytest.fixture
def vit_tiny_patch16_224():
    """Yield a vit_tiny_patch16_224 model."""
    yield TimmNetworkBuilder("vit_tiny_patch16_224", pretrained=False, freeze=True, exportable=True)


@pytest.fixture
def dino_vits8():
    """Yield a dino_vits8 model."""
    yield TorchHubNetworkBuilder(
        repo_or_dir="facebookresearch/dino:main",
        model_name="dino_vits8",
        pretrained=False,
        freeze=True,
        exportable=True,
    )


@pytest.fixture
def dino_vitb8():
    """Yield a dino_vitb8 model."""
    yield TorchHubNetworkBuilder(
        repo_or_dir="facebookresearch/dino:main",
        model_name="dino_vitb8",
        pretrained=False,
        freeze=True,
        exportable=True,
    )
