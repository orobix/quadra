import hydra
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize


def _check_output_dimension(model, cfg):
    input_size = cfg.backbone.metadata.input_size

    with torch.no_grad():
        out = model(torch.randn(1, 3, input_size, input_size))
        assert out.squeeze().shape[0] == cfg.backbone.metadata.output_dim


@pytest.mark.parametrize("model_name", ["dino_vits8", "dino_vitb8"])
def test_dino(model_name: str):
    with initialize(config_path="../../quadra/configs"):
        cfg = compose(config_name=f"backbone/{model_name}", overrides=["backbone.model.pretrained=False"])

    model = hydra.utils.instantiate(cfg.backbone.model)
    model = model.eval()

    _check_output_dimension(model, cfg)


def test_efficientnetv2_s():
    with initialize(config_path="../../quadra/configs"):
        cfg = compose(config_name="backbone/efficientnetv2_s", overrides=["backbone.model.pretrained=False"])

    model = hydra.utils.instantiate(cfg.backbone.model)
    model = model.eval()

    _check_output_dimension(model, cfg)


@pytest.mark.parametrize("model_name", ["resnet18", "resnet18_ssl", "resnet50", "resnet101"])
def test_resnet(model_name: str):
    with initialize(config_path="../../quadra/configs"):
        cfg = compose(config_name=f"backbone/{model_name}", overrides=["backbone.model.pretrained=False"])

    model = hydra.utils.instantiate(cfg.backbone.model)
    model = model.eval()

    _check_output_dimension(model, cfg)


@pytest.mark.parametrize("model_name", ["levit_128s", "vit16_base", "vit16_small", "vit16_tiny", "xcit_tiny_24_p8_224"])
def test_vit(model_name: str):
    with initialize(config_path="../../quadra/configs"):
        cfg = compose(config_name=f"backbone/{model_name}", overrides=["backbone.model.pretrained=False"])

    model = hydra.utils.instantiate(cfg.backbone.model)
    model = model.eval()

    _check_output_dimension(model, cfg)


def test_mnasnet0_5():
    with initialize(config_path="../../quadra/configs"):
        cfg = compose(config_name="backbone/mnasnet0_5", overrides=["backbone.model.pretrained=False"])

    model = hydra.utils.instantiate(cfg.backbone.model)
    model = model.eval()

    _check_output_dimension(model, cfg)


@pytest.mark.parametrize("apply_preclassifier", [True, False])
@pytest.mark.parametrize("apply_classifier", [True, False])
@pytest.mark.parametrize("hyperspherical", [True, False])
def test_model_customization(apply_preclassifier: bool, apply_classifier: bool, hyperspherical: bool):
    with initialize(config_path="../../quadra/configs"):
        cfg = compose(config_name="backbone/resnet18", overrides=["backbone.model.pretrained=False"])

    model = hydra.utils.instantiate(cfg.backbone.model)
    model = model.eval()

    final_dim = cfg.backbone.metadata.output_dim

    if hyperspherical:
        model.hyperspherical = True

    if apply_preclassifier:
        model.pre_classifier = nn.Sequential(
            nn.Linear(cfg.backbone.metadata.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        final_dim = 128

    if apply_classifier:
        model.classifier = nn.Linear(final_dim, 10)
        final_dim = 10

    with torch.no_grad():
        out = model(torch.randn(1, 3, cfg.backbone.metadata.input_size, cfg.backbone.metadata.input_size))
        assert out.squeeze().shape[0] == final_dim
