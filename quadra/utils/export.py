import os
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import torch
from anomalib.models.cflow import CflowLightning
from omegaconf import DictConfig
from torch import nn
from torch.jit._script import RecursiveScriptModule

from quadra.models.base import ModelSignatureWrapper
from quadra.utils.logger import get_logger

try:
    import onnx  # noqa
    from onnxsim import simplify as onnx_simplify  # noqa

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

log = get_logger(__name__)


def generate_torch_inputs(
    input_shapes: List[Any],
    device: str,
    half_precision: bool = False,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
) -> Union[List[Any], Tuple[Any, ...], torch.Tensor]:
    """Given a list of input shapes that can contain either lists, tuples or dicts, with tuples being the input shapes
    of the model, generate a list of torch tensors with the given device and dtype.
    """
    if isinstance(input_shapes, list):
        if any(isinstance(inp, (Sequence, dict)) for inp in input_shapes):
            return [generate_torch_inputs(inp, device, half_precision, dtype) for inp in input_shapes]

        # Base case
        inp = torch.randn((batch_size, *input_shapes), dtype=dtype, device=device)

    if isinstance(input_shapes, dict):
        return {k: generate_torch_inputs(v, device, half_precision, dtype) for k, v in input_shapes.items()}

    if isinstance(input_shapes, tuple):
        if any(isinstance(inp, (Sequence, dict)) for inp in input_shapes):
            # The tuple contains a list, tuple or dict
            return tuple(generate_torch_inputs(inp, device, half_precision, dtype) for inp in input_shapes)

        # Base case
        inp = torch.randn((batch_size, *input_shapes), dtype=dtype, device=device)

    if half_precision:
        inp = inp.half()

    return inp


def extract_torch_model_inputs(
    model: Union[nn.Module, ModelSignatureWrapper],
    input_shapes: Optional[List[Any]] = None,
    half_precision: bool = False,
    batch_size: int = 1,
) -> Optional[Tuple[Union[List[Any], Tuple[Any, ...], torch.Tensor], List[Any]]]:
    """Extract the input shapes from a model and generate a list of torch tensors with the given device and dtype.

    Args:
        model: PyTorch model to be exported
        input_shapes: Inputs shape for tracing
        half_precision: If True, the model will be exported with half precision
        batch_size: Batch size for the input shapes
    """
    if isinstance(model, ModelSignatureWrapper):
        if input_shapes is None:
            input_shapes = model.input_shapes
        model = model.instance

    if input_shapes is None:
        log.warning(
            "Input shape is None, can not trace model! Please provide input_shapes in the task export configuration."
        )
        return None

    if half_precision:
        model.to("cuda:0")
        model = model.half()
        inp = generate_torch_inputs(
            input_shapes=input_shapes, device="cuda:0", half_precision=True, dtype=torch.float16, batch_size=batch_size
        )
    else:
        model.cpu()
        inp = generate_torch_inputs(
            input_shapes=input_shapes, device="cpu", half_precision=False, dtype=torch.float32, batch_size=batch_size
        )

    return inp, input_shapes


@torch.inference_mode()
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
    if isinstance(model, CflowLightning):
        log.warning("Exporting cflow model with torchscript is not supported yet.")
        return None

    model.eval()

    model_inputs = extract_torch_model_inputs(model, input_shapes, half_precision)
    if model_inputs is None:
        return None

    inp, input_shapes = model_inputs

    model_jit = torch.jit.trace(model, inp)

    os.makedirs(output_path, exist_ok=True)

    model_path = os.path.join(output_path, model_name)
    model_jit.save(model_path)

    log.info("Torchscript model saved to %s", os.path.join(os.getcwd(), model_path))

    return os.path.join(os.getcwd(), model_path), input_shapes


@torch.inference_mode()
def export_onnx_model(
    model: nn.Module,
    output_path: str,
    onnx_config: DictConfig,
    input_shapes: Optional[List[Any]] = None,
    half_precision: bool = False,
    model_name: str = "model.onnx",
) -> Optional[Tuple[str, Any]]:
    """Export a PyTorch model with ONNX.

    Args:
        model: PyTorch model to be exported
        output_path: Path to save the model
        input_shapes: Input shapes for tracing
        onnx_config: ONNX export configuration
        half_precision: If True, the model will be exported with half precision
        model_name: Name of the exported model
    """
    batch_size = 1 if onnx_config.fixed_batch_size is None else onnx_config.fixed_batch_size

    model_inputs = extract_torch_model_inputs(
        model=model, input_shapes=input_shapes, half_precision=half_precision, batch_size=batch_size
    )
    if model_inputs is None:
        return None

    inp, input_shapes = model_inputs

    os.makedirs(output_path, exist_ok=True)

    model_path = os.path.join(output_path, model_name)

    input_names = onnx_config.input_names if hasattr(onnx_config, "input_names") else None

    if input_names is None:
        input_names = []
        for i, _ in enumerate(inp):
            input_names.append(f"input_{i}")

    output = [model(*inp)]
    output_names = onnx_config.output_names if hasattr(onnx_config, "output_names") else None

    if output_names is None:
        output_names = []
        for i, _ in enumerate(output):
            output_names.append(f"output_{i}")

    dynamic_axes = onnx_config.dynamic_axes if hasattr(onnx_config, "dynamic_axes") else None

    if onnx_config.fixed_batch_size is None:
        if dynamic_axes is None:
            dynamic_axes = {}
            for i, _ in enumerate(input_names):
                dynamic_axes[input_names[i]] = {0: "batch_size"}

            for i, _ in enumerate(output_names):
                dynamic_axes[output_names[i]] = {0: "batch_size"}
    else:
        dynamic_axes = None

    if len(inp) == 1:
        inp = inp[0]
    try:
        with torch.autocast("cuda"):
            torch.onnx.export(
                model=model,
                args=inp,
                f=model_path,
                export_params=onnx_config.export_params,
                opset_version=onnx_config.opset_version,
                do_constant_folding=onnx_config.do_constant_folding,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

        onnx_model = onnx.load(model_path)
        # Check if ONNX model is valid
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        log.warning("ONNX export failed with error: %s", e)
        return None

    log.info("ONNX model saved to %s", os.path.join(os.getcwd(), model_path))

    if onnx_config.simplify:
        log.info("Attempting to simplify ONNX model")
        onnx_model = onnx.load(model_path)
        simplified_model, check = onnx_simplify(onnx_model)

        if not check:
            log.warning("Simplified ONNX model could not be validated, using original ONNX model")
        else:
            model_filename, model_extension = os.path.splitext(model_name)
            model_name = f"{model_filename}_simplified{model_extension}"
            model_path = os.path.join(output_path, model_name)
            onnx.save(simplified_model, model_path)
            log.info("Simplified ONNX model saved to %s", os.path.join(os.getcwd(), model_path))

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
    log.info("Pytorch model saved to %s", os.path.join(output_path, model_name))

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
