from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from quadra.utils.export import export_onnx_model, export_torchscript_model, import_deployment_model
from quadra.utils.tests.fixtures.models import (  # noqa
    dino_vitb8,
    dino_vits8,
    draem,
    padim_resnet18,
    patchcore_resnet18,
    resnet18,
    resnet50,
    smp_resnet18_unet,
    smp_resnet18_unetplusplus,
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


@torch.inference_mode()
def check_export_model_outputs(tmp_path: Path, model: nn.Module, export_types: list[str], input_shapes: tuple[Any]):
    exported_models = {}

    for export_type in export_types:
        if export_type == "torchscript":
            out = export_torchscript_model(
                model=model,
                input_shapes=input_shapes,
                output_path=tmp_path,
                half_precision=False,
            )

            torchscript_model_path, input_shapes = out
            exported_models[export_type] = torchscript_model_path
        else:
            out = export_onnx_model(
                model=model,
                output_path=tmp_path,
                onnx_config=ONNX_CONFIG,
                input_shapes=input_shapes,
                half_precision=False,
            )

            onnx_model_path, input_shapes = out
            exported_models[export_type] = onnx_model_path

    inference_config = DictConfig({"onnx": {}, "torchscript": {}})

    models = []
    for export_type, model_path in exported_models.items():
        model = import_deployment_model(model_path=model_path, inference_config=inference_config, device="cpu")
        models.append(model)

    inp = torch.rand((1, *input_shapes[0]), dtype=torch.float32)

    outputs = []

    for model in models:
        outputs.append(model(inp))

    for i in range(len(outputs) - 1):
        if isinstance(outputs[i], Sequence):
            for j in range(len(outputs[i])):
                assert torch.allclose(outputs[i][j], outputs[i + 1][j], atol=1e-5)
        else:
            assert torch.allclose(outputs[i], outputs[i + 1], atol=1e-5)


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

    check_export_model_outputs(tmp_path=tmp_path, model=model, export_types=export_types, input_shapes=input_shapes)


@pytest.mark.parametrize(
    "model",
    [
        pytest.lazy_fixture("smp_resnet18_unet"),
        pytest.lazy_fixture("smp_resnet18_unetplusplus"),
    ],
)
def test_segmentation_models_export(tmp_path: Path, model: nn.Module):
    export_types = ["onnx", "torchscript"]

    input_shapes = [(3, 224, 224)]

    check_export_model_outputs(tmp_path=tmp_path, model=model, export_types=export_types, input_shapes=input_shapes)


@pytest.mark.parametrize(
    "model",
    [
        pytest.lazy_fixture("padim_resnet18"),
        pytest.lazy_fixture("patchcore_resnet18"),
        pytest.lazy_fixture("draem"),
    ],
)
def test_anomaly_detection_models_export(tmp_path: Path, model: nn.Module):
    export_types = ["onnx", "torchscript"]

    input_shapes = [(3, 224, 224)]

    check_export_model_outputs(tmp_path=tmp_path, model=model, export_types=export_types, input_shapes=input_shapes)
