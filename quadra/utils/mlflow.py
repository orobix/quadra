from __future__ import annotations

try:
    from mlflow.models import infer_signature  # noqa
    from mlflow.models.signature import ModelSignature  # noqa

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from quadra.models.evaluation import BaseEvaluationModel


@torch.inference_mode()
def infer_signature_model(model: BaseEvaluationModel, data: list[Any]) -> ModelSignature | None:
    """Infer input and output signature for a PyTorch/Torchscript model."""
    model = model.eval()
    model_output = model(*data)

    try:
        output_signature = infer_signature_input(model_output)

        if len(data) == 1:
            signature_input = infer_signature_input(data[0])
        else:
            signature_input = infer_signature_input(data)
    except ValueError:
        # TODO: Solve circular import as it is not possible to import get_logger right now
        # log.warning("Unable to infer signature for model output type %s", type(model_output))
        return None

    return infer_signature(signature_input, output_signature)


def infer_signature_input(input_tensor: Any) -> Any:
    """Recursively infer the signature input format to pass to mlflow.models.infer_signature.

    Raises:
        ValueError: If the input type is not supported or when nested dicts or sequences are encountered.
    """
    signature: dict[str, Any] | np.ndarray
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

            signature[f"output_{i}"] = infer_signature_input(x)
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

            signature[k] = infer_signature_input(v)
    else:
        raise ValueError(f"Unable to infer signature for model output type {type(input_tensor)}")

    return signature


def get_mlflow_logger(trainer: Trainer) -> MLFlowLogger | None:
    """Safely get Mlflow logger from Trainer loggers.

    Args:
        trainer: Pytorch Lightning trainer.

    Returns:
        An mlflow logger if available, else None.
    """
    if isinstance(trainer.logger, MLFlowLogger):
        return trainer.logger

    if isinstance(trainer.logger, list):
        for logger in trainer.logger:
            if isinstance(logger, MLFlowLogger):
                return logger

    return None
