import os
from typing import Optional, Tuple, cast

import torch
from anomalib.models.cflow import CflowLightning
from torch import nn
from torch.jit._script import RecursiveScriptModule

from quadra.utils.utils import get_logger

log = get_logger(__name__)


def export_torchscript_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    half_precision: bool = False,
    model_name: str = "model.pt",
) -> Optional[str]:
    """Export a PyTorch model with TorchScript.

    Args:
        model: PyTorch model to be exported
        input_shape: Input shape for tracing
        output_path: Path to save the model
        half_precision: If True, the model will be exported with half precision
        model_name: Name of the exported model

    Returns:
        If the model is exported successfully, the path to the model is returned.

    """
    model.eval()
    if isinstance(model, CflowLightning):
        log.warning("Exporting cflow model with torchscript is not supported yet.")
        return None

    if half_precision:
        log.info("Jitting model with half precision on GPU")
        model.to("cuda:0")
        model = model.half()
        inp = torch.randn(input_shape, dtype=torch.float16, device="cuda:0")
    else:
        log.info("Jitting model with double precision")
        model.cpu()
        inp = torch.randn(input_shape, dtype=torch.float32)

    with torch.no_grad():
        model_jit = torch.jit.trace(model, inp)

    os.makedirs(output_path, exist_ok=True)

    model_path = os.path.join(output_path, model_name)
    model_jit.save(model_path)

    log.info("Torchscript model saved to %s", os.path.join(os.getcwd(), model_path))

    return os.path.join(os.getcwd(), model_path)


# TODO: Update signature when new models are added
def import_deployment_model(model_path: str, device: str) -> Tuple[RecursiveScriptModule, str]:
    """Try to import a model for deployment, currently only supports torchscript .pt files.

    Args:
        model_path: Path to the model
        device: Device to load the model on

    Returns:
        A tuple containing the model and the model type
    """
    file_extension = os.path.splitext(os.path.basename(model_path))[1]
    if file_extension == ".pt":
        model = cast(RecursiveScriptModule, torch.jit.load(model_path))
        model.eval()
        model.to(device)
        return model, "torchscript"

    raise ValueError(f"Unable to load model with extension {file_extension}, valid extensions are: ['.pt']")
