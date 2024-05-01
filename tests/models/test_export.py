from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from anomalib.models.efficient_ad.torch_model import EfficientAdModel
from omegaconf import DictConfig
from torch import nn

from quadra.utils.export import export_onnx_model, export_torchscript_model, import_deployment_model
from quadra.utils.tests.fixtures.models import (  # noqa
    dino_vitb8,
    dino_vits8,
    draem,
    efficient_ad_small,
    padim_resnet18,
    patchcore_resnet18,
    resnet18,
    resnet50,
    smp_resnet18_unet,
    smp_resnet18_unetplusplus,
    vit_tiny_patch16_224,
)
from quadra.utils.tests.helpers import get_quadra_test_device

try:
    import onnx  # noqa
    import onnxruntime  # noqa
    import onnxsim  # noqa

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

ONNX_CONFIG = DictConfig(
    {
        "opset_version": 16,
        "do_constant_folding": True,
        "export_params": True,
        "simplify": False,
    }
)


@torch.inference_mode()
def check_export_model_outputs(
    tmp_path: Path, model: nn.Module, export_types: list[str], input_shapes: tuple[Any], half_precision: bool = False
):
    exported_models = {}
    device = get_quadra_test_device()

    for export_type in export_types:
        if export_type == "torchscript":
            out = export_torchscript_model(
                model=model,
                input_shapes=input_shapes,
                output_path=tmp_path,
                half_precision=half_precision,
            )

            torchscript_model_path, input_shapes = out
            exported_models[export_type] = torchscript_model_path
        else:
            out = export_onnx_model(
                model=model,
                output_path=tmp_path,
                onnx_config=ONNX_CONFIG,
                input_shapes=input_shapes,
                half_precision=half_precision,
            )

            onnx_model_path, input_shapes = out
            exported_models[export_type] = onnx_model_path

    inference_config = DictConfig(
        {
            "onnx": {
                "session_options": {
                    "inter_op_num_threads": 4,
                    "intra_op_num_threads": 4,
                }
            },
            "torchscript": {},
        }
    )

    models = []
    for _, model_path in exported_models.items():
        model = import_deployment_model(model_path=model_path, inference_config=inference_config, device=device)
        models.append(model)

    inp = torch.rand((1, *input_shapes[0]), dtype=torch.float32 if not half_precision else torch.float16, device=device)

    outputs = []

    for model in models:
        outputs.append(model(inp))

    tolerance = 1e-4 if not half_precision else 1e-2
    for i in range(len(outputs) - 1):
        if isinstance(outputs[i], Sequence):
            for j in range(len(outputs[i])):
                assert torch.allclose(outputs[i][j], outputs[i + 1][j], atol=tolerance)
        else:
            assert torch.allclose(outputs[i], outputs[i + 1], atol=tolerance)


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
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
@pytest.mark.parametrize("half_precision", [False, True])
def test_classification_models_export(tmp_path: Path, model: nn.Module, half_precision: bool):
    if half_precision and get_quadra_test_device() == "cpu":
        pytest.skip("Half precision not supported on CPU")

    export_types = ["onnx", "torchscript"]

    input_shapes = [(3, 224, 224)]

    check_export_model_outputs(
        tmp_path=tmp_path,
        model=model,
        export_types=export_types,
        input_shapes=input_shapes,
        half_precision=half_precision,
    )


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
@pytest.mark.parametrize(
    "model",
    [
        pytest.lazy_fixture("smp_resnet18_unet"),
        pytest.lazy_fixture("smp_resnet18_unetplusplus"),
    ],
)
@pytest.mark.parametrize("half_precision", [False, True])
def test_segmentation_models_export(tmp_path: Path, model: nn.Module, half_precision: bool):
    if half_precision and get_quadra_test_device() == "cpu":
        pytest.skip("Half precision not supported on CPU")

    export_types = ["onnx", "torchscript"]

    input_shapes = [(3, 224, 224)]

    check_export_model_outputs(
        tmp_path=tmp_path,
        model=model,
        export_types=export_types,
        input_shapes=input_shapes,
        half_precision=half_precision,
    )


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
@pytest.mark.parametrize(
    "model",
    [
        pytest.lazy_fixture("padim_resnet18"),
        pytest.lazy_fixture("patchcore_resnet18"),
        pytest.lazy_fixture("draem"),
        pytest.lazy_fixture("efficient_ad_small"),
    ],
)
@pytest.mark.parametrize("half_precision", [False, True])
def test_anomaly_detection_models_export(tmp_path: Path, model: nn.Module, half_precision: bool):
    if half_precision and get_quadra_test_device() == "cpu":
        pytest.skip("Half precision not supported on CPU")

    export_types = ["onnx", "torchscript"]

    if isinstance(model, EfficientAdModel):
        input_shapes = [(3, 256, 256)]
    else:
        input_shapes = [(3, 224, 224)]

    check_export_model_outputs(
        tmp_path=tmp_path,
        model=model,
        export_types=export_types,
        input_shapes=input_shapes,
        half_precision=half_precision,
    )
