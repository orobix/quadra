from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from quadra.utils.export import export_onnx_model, export_torchscript_model, import_deployment_model
from quadra.utils.tests.fixtures.models.classification import (  # noqa
    dino_vitb8,
    dino_vits8,
    resnet18,
    resnet50,
    vit_tiny_patch16_224,
)

ONNX_CONFIG = DictConfig(
    {
        "opset_version": 16,
        "do_constant_folding": True,
        "export_params": True,
        "simplify": False,
    }
)


@pytest.mark.parametrize(
    "model",
    [
        pytest.lazy_fixture("dino_vitb8"),
        pytest.lazy_fixture("dino_vits8"),
        pytest.lazy_fixture("resnet18"),
        pytest.lazy_fixture("resnet50"),
        pytest.lazy_fixture("vit_tiny_patch16_224"),
    ],
)
def test_classification_models_export(tmp_path: Path, model: nn.Module):
    export_types = ["onnx", "torchscript"]

    input_shapes = [(3, 224, 224)]

    for export_type in export_types:
        if export_type == "torchscript":
            out = export_torchscript_model(
                model=model,
                input_shapes=input_shapes,
                output_path=tmp_path,
                half_precision=False,
            )

            torchscript_model_path, input_shapes = out
        else:
            out = export_onnx_model(
                model=model,
                output_path=tmp_path,
                onnx_config=ONNX_CONFIG,
                input_shapes=input_shapes,
                half_precision=False,
            )

            onnx_model_path, input_shapes = out

    inference_config = DictConfig({"onnx": {}, "torchscript": {}})

    onnx_model = import_deployment_model(model_path=onnx_model_path, inference_config=inference_config, device="cpu")

    torchscript_model = import_deployment_model(
        model_path=torchscript_model_path,
        inference_config=inference_config,
        device="cpu",
    )

    inp = torch.rand((1, *input_shapes[0]), dtype=torch.float32)

    onnx_out = onnx_model(inp)
    torchscript_out = torchscript_model(inp)

    assert torch.allclose(onnx_out, torchscript_out, atol=1e-3)
