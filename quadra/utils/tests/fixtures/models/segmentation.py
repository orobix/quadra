import pytest

from quadra.modules.backbone import create_smp_backbone
from quadra.utils.segmentation import patch_mix_transformer_encoder


@pytest.fixture
def smp_resnet18_unet():
    """Yield a unet with resnet18 encoder."""
    yield create_smp_backbone(
        arch="unet",
        encoder_name="resnet18",
        encoder_weights=None,
        encoder_depth=5,
        freeze_encoder=True,
        in_channels=3,
        num_classes=1,
        activation=None,
    )


@pytest.fixture
def smp_mit_b0_unet():
    """Yield a unet with patched mit_b0 encoder."""
    patch_mix_transformer_encoder()
    yield create_smp_backbone(
        arch="unet",
        encoder_name="mit_b0",
        encoder_weights=None,
        freeze_encoder=False,
        in_channels=3,
        num_classes=1,
        activation=None,
    )


@pytest.fixture
def smp_resnet18_unetplusplus():
    """Yield a unetplusplus with resnet18 encoder."""
    yield create_smp_backbone(
        arch="unetplusplus",
        encoder_name="resnet18",
        encoder_weights=None,
        encoder_depth=5,
        freeze_encoder=True,
        in_channels=3,
        num_classes=1,
        activation=None,
    )
