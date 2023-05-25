from typing import Any

import segmentation_models_pytorch as smp


def create_smp_backbone(
    arch: str,
    encoder_name: str,
    freeze_encoder: bool = False,
    in_channels: int = 3,
    num_classes: int = 0,
    **kwargs: Any,
):
    """Create Segmentation.models.pytorch model backbone
    Args:
        arch: architecture name
        encoder_name: architecture name
        freeze_encoder: freeze encoder or not
        in_channels: number of input channels
        num_classes: number of classes
        **kwargs: extra arguments for model (for example classification head).
    """
    model = smp.create_model(
        arch=arch, encoder_name=encoder_name, in_channels=in_channels, classes=num_classes, **kwargs
    )
    if freeze_encoder:
        for child in model.encoder.children():
            for param in child.parameters():
                param.requires_grad = False
    return model
