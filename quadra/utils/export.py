import os
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import torch
from anomalib.models.cflow import CflowLightning
from torch import nn
from torch.jit._script import RecursiveScriptModule

from quadra.models.base import ModelSignatureWrapper

# TODO: Solve circular import as it is not possible to import get_logger right now


def generate_torch_inputs(
    input_shapes: List[Any], device: str, half_precision: bool = False, dtype: torch.dtype = torch.float32
) -> Union[List[Any], Tuple[Any, ...], torch.Tensor]:
    """Given a list of input shapes that can contain either lists, tuples or dicts, with tuples being the input shapes
    of the model, generate a list of torch tensors with the given device and dtype.
    """
    if isinstance(input_shapes, list):
        if any(isinstance(inp, (Sequence, dict)) for inp in input_shapes):
            return [generate_torch_inputs(inp, device, half_precision, dtype) for inp in input_shapes]

        # Base case
        inp = torch.randn((1, *input_shapes), dtype=dtype, device=device)

    if isinstance(input_shapes, dict):
        return {k: generate_torch_inputs(v, device, half_precision, dtype) for k, v in input_shapes.items()}

    if isinstance(input_shapes, tuple):
        if any(isinstance(inp, (Sequence, dict)) for inp in input_shapes):
            # The tuple contains a list, tuple or dict
            return tuple(generate_torch_inputs(inp, device, half_precision, dtype) for inp in input_shapes)

        # Base case
        inp = torch.randn((1, *input_shapes), dtype=dtype, device=device)

    if half_precision:
        inp = inp.half()

    return inp


def export_torchscript_model(
    model: nn.Module,
    output_path: str,
    input_shapes: Optional[List[Any]] = None,
    half_precision: bool = False,
    model_name: str = "model.pt",
) -> Optional[Tuple[str, Any]]:
    """Export a PyTorch model with TorchScript.

    Args:
        model: PyTorch model to be exported
        input_shapes: Inputs shape for tracing
        output_path: Path to save the model
        half_precision: If True, the model will be exported with half precision
        model_name: Name of the exported model

    Returns:
        If the model is exported successfully, the path to the model and the input shape are returned.

    """
    if isinstance(model, ModelSignatureWrapper):
        if input_shapes is None:
            input_shapes = model.input_shapes
        model = model.instance

    if input_shapes is None:
        # log.warning(
        #    "Input shape is None, can not trace model! Please provide input_shapes in the task export configuration."
        # )
        return None

    if isinstance(model, CflowLightning):
        # log.warning("Exporting cflow model with torchscript is not supported yet.")
        return None

    model.eval()
    if half_precision:
        # log.info("Jitting model with half precision on GPU")
        model.to("cuda:0")
        model = model.half()
        inp = generate_torch_inputs(
            input_shapes=input_shapes, device="cuda:0", half_precision=True, dtype=torch.float16
        )
    else:
        # log.info("Jitting model with double precision")
        model.cpu()
        inp = generate_torch_inputs(input_shapes=input_shapes, device="cpu", half_precision=False, dtype=torch.float32)

    with torch.no_grad():
        model_jit = torch.jit.trace(model, inp)

    os.makedirs(output_path, exist_ok=True)

    model_path = os.path.join(output_path, model_name)
    model_jit.save(model_path)

    # log.info("Torchscript model saved to %s", os.path.join(os.getcwd(), model_path))

    return os.path.join(os.getcwd(), model_path), input_shapes


def export_pytorch_model(model: nn.Module, output_path: str, model_name: str = "model.pth") -> str:
    """Export pytorch model's parameter dictionary using a deserialized state_dict.

    Args:
        model: PyTorch model to be exported
        output_path: Path to save the model
        model_name: Name of the exported model

    Returns:
        If the model is exported successfully, the path to the model is returned.

    """
    os.makedirs(output_path, exist_ok=True)
    model.eval()
    model.cpu()
    model_path = os.path.join(output_path, model_name)
    torch.save(model.state_dict(), model_path)
    # log.info("Pytorch model saved to %s", os.path.join(output_path, model_name))

    return os.path.join(os.getcwd(), model_path)


# TODO: Update signature when new models are added
def import_deployment_model(
    model_path: str, device: str, model: Optional[nn.Module] = None
) -> Tuple[Union[RecursiveScriptModule, nn.Module], str]:
    """Try to import a model for deployment, currently only supports torchscript .pt files and
    state dictionaries .pth files.

    Args:
        model_path: Path to the model
        device: Device to load the model on
        model: Pytorch model needed to load the parameter dictionary

    Returns:
        A tuple containing the model and the model type
    """

    file_extension = os.path.splitext(os.path.basename(model_path))[1]
    if file_extension == ".pt":
        model = cast(RecursiveScriptModule, torch.jit.load(model_path))
        model.eval()
        model.to(device)

        return model, "torchscript"
    if file_extension == ".pth":
        if model is None:
            raise ValueError("Model is not defined, can not load state_dict!")

        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)

        return model, "torch"

    raise ValueError(f"Unable to load model with extension {file_extension}, valid extensions are: ['.pt', 'pth']")
