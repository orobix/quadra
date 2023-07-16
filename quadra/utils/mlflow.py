from __future__ import annotations

try:
    from mlflow.models import infer_signature  # noqa
    from mlflow.models.signature import ModelSignature  # noqa

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from typing import Any, Sequence, TypeVar

import torch
from torch import nn

NnModuleT = TypeVar("NnModuleT", bound=nn.Module)


@torch.inference_mode()
def infer_signature_torch_model(model: NnModuleT, data: list[Any]) -> ModelSignature | None:
    """Infer input and output signature for a PyTorch/Torchscript model."""
    model = model.eval()
    model = model.cpu()
    model_output = model(*data)

    try:
        output_signature = infer_signature_input_torch(model_output)

        if len(data) == 1:
            signature_input = infer_signature_input_torch(data[0])
        else:
            signature_input = infer_signature_input_torch(data)
    except ValueError:
        # TODO: Solve circular import as it is not possible to import get_logger right now
        # log.warning("Unable to infer signature for model output type %s", type(model_output))
        return None

    return infer_signature(signature_input, output_signature)


def infer_signature_input_torch(input_tensor: Any) -> Any:
    """Recursively infer the signature input format to pass to mlflow.models.infer_signature.

    Raises:
        ValueError: If the input type is not supported or when nested dicts or sequences are encountered.
    """
    if isinstance(input_tensor, Sequence):
        # Mlflow currently does not support sequence outputs, so we use a dict instead
        signature = {}
        for i, x in enumerate(input_tensor):
            if isinstance(x, Sequence):
                # Nested signature is currently not supported by mlflow
                raise ValueError("Nested sequences are not supported")
                # TODO: Enable this once mlflow supports nested signatures
                # signature[f"output_{i}"] = {f"output_{j}": infer_signature_torch(y) for j, y in enumerate(x)}
            if isinstance(x, dict):
                # Nested dicts are not supported
                raise ValueError("Nested dicts are not supported")

            signature[f"output_{i}"] = infer_signature_input_torch(x)
    elif isinstance(input_tensor, torch.Tensor):
        signature = input_tensor.cpu().numpy()
    elif isinstance(input_tensor, dict):
        signature = {}
        for k, v in input_tensor.items():
            if isinstance(v, dict):
                # Nested dicts are not supported
                raise ValueError("Nested dicts are not supported")
            if isinstance(v, Sequence):
                # Nested signature is currently not supported by mlflow
                raise ValueError("Nested sequences are not supported")

            signature[k] = infer_signature_input_torch(v)
    else:
        raise ValueError(f"Unable to infer signature for model output type {type(input_tensor)}")

    return signature
