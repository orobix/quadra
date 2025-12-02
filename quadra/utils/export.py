from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from typing import Any, Literal, TypeVar, cast

import torch
from anomalib.models.cflow import CflowLightning
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn

from quadra.models.base import ModelSignatureWrapper
from quadra.models.evaluation import (
    BaseEvaluationModel,
    ONNXEvaluationModel,
    TorchEvaluationModel,
    TorchscriptEvaluationModel,
)
from quadra.utils.logger import get_logger

try:
    import onnx  # noqa
    from onnxsim import simplify as onnx_simplify  # noqa
    from onnxconverter_common import auto_convert_mixed_precision  # noqa

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

log = get_logger(__name__)

BaseDeploymentModelT = TypeVar("BaseDeploymentModelT", bound=BaseEvaluationModel)


def generate_torch_inputs(
    input_shapes: list[Any],
    device: str | torch.device,
    half_precision: bool = False,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
) -> list[Any] | tuple[Any, ...] | torch.Tensor:
    """Given a list of input shapes that can contain either lists, tuples or dicts, with tuples being the input shapes
    of the model, generate a list of torch tensors with the given device and dtype.
    """
    inp = None

    if isinstance(input_shapes, ListConfig | DictConfig):
        input_shapes = OmegaConf.to_container(input_shapes)

    if isinstance(input_shapes, list):
        if any(isinstance(inp, Sequence | dict) for inp in input_shapes):
            return [generate_torch_inputs(inp, device, half_precision, dtype) for inp in input_shapes]

        # Base case
        inp = torch.randn((batch_size, *input_shapes), dtype=dtype, device=device)

    if isinstance(input_shapes, dict):
        return {k: generate_torch_inputs(v, device, half_precision, dtype) for k, v in input_shapes.items()}

    if isinstance(input_shapes, tuple):
        if any(isinstance(inp, Sequence | dict) for inp in input_shapes):
            # The tuple contains a list, tuple or dict
            return tuple(generate_torch_inputs(inp, device, half_precision, dtype) for inp in input_shapes)

        # Base case
        inp = torch.randn((batch_size, *input_shapes), dtype=dtype, device=device)

    if inp is None:
        raise RuntimeError("Something went wrong during model export, unable to parse input shapes")

    if half_precision:
        inp = inp.half()

    return inp


def extract_torch_model_inputs(
    model: nn.Module | ModelSignatureWrapper,
    input_shapes: list[Any] | None = None,
    half_precision: bool = False,
    batch_size: int = 1,
) -> tuple[list[Any] | tuple[Any, ...] | torch.Tensor, list[Any]] | None:
    """Extract the input shapes for the given model and generate a list of torch tensors with the
    given device and dtype.

    Args:
        model: Module or ModelSignatureWrapper
        input_shapes: Inputs shapes
        half_precision: If True, the model will be exported with half precision
        batch_size: Batch size for the input shapes
    """
    if isinstance(model, ModelSignatureWrapper) and input_shapes is None:
        input_shapes = model.input_shapes

    if input_shapes is None:
        log.warning(
            "Input shape is None, can not trace model! Please provide input_shapes in the task export configuration."
        )
        return None

    if half_precision:
        # TODO: This doesn't support bfloat16!!
        inp = generate_torch_inputs(
            input_shapes=input_shapes, device="cuda:0", half_precision=True, dtype=torch.float16, batch_size=batch_size
        )
    else:
        inp = generate_torch_inputs(
            input_shapes=input_shapes, device="cpu", half_precision=False, dtype=torch.float32, batch_size=batch_size
        )

    return inp, input_shapes


@torch.inference_mode()
def export_torchscript_model(
    model: nn.Module,
    output_path: str,
    input_shapes: list[Any] | None = None,
    half_precision: bool = False,
    model_name: str = "model.pt",
    example_inputs: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
) -> tuple[str, Any] | None:
    """Export a PyTorch model with TorchScript.

    Args:
        model: PyTorch model to be exported
        input_shapes: Inputs shape for tracing
        output_path: Path to save the model
        half_precision: If True, the model will be exported with half precision
        model_name: Name of the exported model
        example_inputs: If provided use this to evaluate the model instead of generating random inputs, it's expected to
            be a list of tensors or a single tensor without batch dimension

    Returns:
        If the model is exported successfully, the path to the model and the input shape are returned.

    """
    if isinstance(model, CflowLightning):
        log.warning("Exporting cflow model with torchscript is not supported yet.")
        return None

    model.eval()
    if half_precision:
        model.to("cuda:0")
        model = model.half()
    else:
        model.cpu()

    batch_size = 1
    model_inputs: tuple[list[Any] | tuple[Any, ...] | torch.Tensor, list[Any]] | None
    if example_inputs is not None:
        if isinstance(example_inputs, Sequence):
            model_input_tensors = []
            model_input_shapes = []

            for example_input in example_inputs:
                new_inp = example_input.to(
                    device="cuda:0" if half_precision else "cpu",
                    dtype=torch.float16 if half_precision else torch.float32,
                )
                new_inp = new_inp.unsqueeze(0).repeat(batch_size, *(1 for x in new_inp.shape))
                model_input_tensors.append(new_inp)
                model_input_shapes.append(new_inp[0].shape)

            model_inputs = (model_input_tensors, [model_input_shapes])
        else:
            new_inp = example_inputs.to(
                device="cuda:0" if half_precision else "cpu",
                dtype=torch.float16 if half_precision else torch.float32,
            )
            new_inp = new_inp.unsqueeze(0).repeat(batch_size, *(1 for x in new_inp.shape))
            model_inputs = (new_inp, [new_inp[0].shape])
    else:
        model_inputs = extract_torch_model_inputs(model, input_shapes, half_precision)

    if model_inputs is None:
        return None

    if isinstance(model, ModelSignatureWrapper):
        model = model.instance

    inp, input_shapes = model_inputs

    try:
        try:
            model_jit = torch.jit.trace(model, inp)
        except RuntimeError as e:
            log.warning("Standard tracing failed with exception %s, attempting tracing with strict=False", e)
            model_jit = torch.jit.trace(model, inp, strict=False)

        os.makedirs(output_path, exist_ok=True)

        model_path = os.path.join(output_path, model_name)
        model_jit.save(model_path)

        log.info("Torchscript model saved to %s", os.path.join(os.getcwd(), model_path))

        return os.path.join(os.getcwd(), model_path), input_shapes
    except Exception as e:
        log.debug("Failed to export torchscript model with exception: %s", e)
        return None


@torch.inference_mode()
def export_onnx_model(
    model: nn.Module,
    output_path: str,
    onnx_config: DictConfig,
    input_shapes: list[Any] | None = None,
    half_precision: bool = False,
    model_name: str = "model.onnx",
    example_inputs: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
    rtol: float = 0.01,
    atol: float = 5e-3,
) -> tuple[str, Any] | None:
    """Export a PyTorch model with ONNX.

    Args:
        model: PyTorch model to be exported
        output_path: Path to save the model
        input_shapes: Input shapes for tracing
        onnx_config: ONNX export configuration
        half_precision: If True, the model will be exported with half precision
        model_name: Name of the exported model
        example_inputs: If provided use this to evaluate the model instead of generating random inputs, it's expected to
            be a list of tensors or a single tensor without batch dimension
        rtol: Relative tolerance for the ONNX safe export in fp16
        atol: Absolute tolerance for the ONNX safe export in fp16
    """
    if not ONNX_AVAILABLE:
        log.warning("ONNX is not installed, can not export model in this format.")
        log.warning("Please install ONNX capabilities for quadra with: poetry install -E onnx")
        return None

    model.eval()
    if half_precision:
        model.to("cuda:0")
        model = model.half()
    else:
        model.cpu()

    if hasattr(onnx_config, "fixed_batch_size") and onnx_config.fixed_batch_size is not None:
        batch_size = onnx_config.fixed_batch_size
    else:
        batch_size = 1

    model_inputs: tuple[list[Any] | tuple[Any, ...] | torch.Tensor, list[Any]] | None
    if example_inputs is not None:
        if isinstance(example_inputs, Sequence):
            model_input_tensors = []
            model_input_shapes = []

            for example_input in example_inputs:
                new_inp = example_input.to(
                    device="cuda:0" if half_precision else "cpu",
                    dtype=torch.float16 if half_precision else torch.float32,
                )
                new_inp = new_inp.unsqueeze(0).repeat(batch_size, *(1 for x in new_inp.shape))
                model_input_tensors.append(new_inp)
                model_input_shapes.append(new_inp[0].shape)

            model_inputs = (model_input_tensors, [model_input_shapes])
        else:
            new_inp = example_inputs.to(
                device="cuda:0" if half_precision else "cpu",
                dtype=torch.float16 if half_precision else torch.float32,
            )
            new_inp = new_inp.unsqueeze(0).repeat(batch_size, *(1 for x in new_inp.shape))
            model_inputs = ([new_inp], [new_inp[0].shape])
    else:
        model_inputs = extract_torch_model_inputs(model, input_shapes, half_precision)

    if model_inputs is None:
        return None

    if isinstance(model, ModelSignatureWrapper):
        model = model.instance

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

    if hasattr(onnx_config, "fixed_batch_size") and onnx_config.fixed_batch_size is not None:
        dynamic_axes = None
    elif dynamic_axes is None:
        dynamic_axes = {}
        for i, _ in enumerate(input_names):
            dynamic_axes[input_names[i]] = {0: "batch_size"}

        for i, _ in enumerate(output_names):
            dynamic_axes[output_names[i]] = {0: "batch_size"}

    modified_onnx_config = cast(dict[str, Any], OmegaConf.to_container(onnx_config, resolve=True))

    modified_onnx_config["input_names"] = input_names
    modified_onnx_config["output_names"] = output_names
    modified_onnx_config["dynamic_axes"] = dynamic_axes

    simplify = modified_onnx_config.pop("simplify", False)
    _ = modified_onnx_config.pop("fixed_batch_size", None)

    if len(inp) == 1:
        inp = inp[0]

    if isinstance(inp, list):
        inp = tuple(inp)  # onnx doesn't like lists representing tuples of inputs
    elif isinstance(inp, torch.Tensor):
        inp = (inp,)

    if isinstance(inp, dict):
        raise ValueError("ONNX export does not support model with dict inputs")

    try:
        torch.onnx.export(model=model, args=inp, f=model_path, **modified_onnx_config)

        onnx_model = onnx.load(model_path)
        # Check if ONNX model is valid
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        log.debug("ONNX export failed with error: %s", e)
        return None

    log.info("ONNX model saved to %s", os.path.join(os.getcwd(), model_path))

    if half_precision:
        is_export_ok = _safe_export_half_precision_onnx(
            model=model,
            export_model_path=model_path,
            inp=inp,
            onnx_config=onnx_config,
            input_shapes=input_shapes,
            input_names=input_names,
            rtol=rtol,
            atol=atol,
        )

        if not is_export_ok:
            return None

    if simplify:
        log.info("Attempting to simplify ONNX model")
        onnx_model = onnx.load(model_path)

        try:
            simplified_model, check = onnx_simplify(onnx_model)
        except Exception as e:
            log.debug("ONNX simplification failed with error: %s", e)
            check = False

        if not check:
            log.warning("Something failed during model simplification, only original ONNX model will be exported")
        else:
            model_filename, model_extension = os.path.splitext(model_name)
            model_name = f"{model_filename}_simplified{model_extension}"
            model_path = os.path.join(output_path, model_name)
            onnx.save(simplified_model, model_path)
            log.info("Simplified ONNX model saved to %s", os.path.join(os.getcwd(), model_path))

    return os.path.join(os.getcwd(), model_path), input_shapes


def _safe_export_half_precision_onnx(
    model: nn.Module,
    export_model_path: str,
    inp: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    onnx_config: DictConfig,
    input_shapes: list[Any],
    input_names: list[str],
    rtol: float = 0.01,
    atol: float = 5e-3,
) -> bool:
    """Check that the exported half precision ONNX model does not contain NaN values. If it does, attempt to export
    the model with a more stable export and overwrite the original model.

    Args:
        model: PyTorch model to be exported
        export_model_path: Path to save the model
        inp: Input tensors for the model
        onnx_config: ONNX export configuration
        input_shapes: Input shapes for the model
        input_names: Input names for the model
        rtol: Relative tolerance to evaluate the model
        atol: Absolute tolerance to evaluate the model

    Returns:
        True if the model is stable or it was possible to export a more stable model, False otherwise.
    """
    test_fp_16_model: BaseEvaluationModel = import_deployment_model(
        export_model_path, OmegaConf.create({"onnx": {}}), "cuda:0"
    )
    if not isinstance(inp, Sequence):
        inp = [inp]

    test_output = test_fp_16_model(*inp)

    if not isinstance(test_output, Sequence):
        test_output = [test_output]

    # Check if there are nan values in any of the outputs
    is_broken_model = any(torch.isnan(out).any() for out in test_output)

    if is_broken_model:
        try:
            log.warning(
                "The exported half precision ONNX model contains NaN values, attempting with a more stable export..."
            )
            # Cast back the fp16 model to fp32 to simulate the export with fp32
            model = model.float()
            log.info("Starting to export model in full precision")
            export_output = export_onnx_model(
                model=model,
                output_path=os.path.dirname(export_model_path),
                # Force to not simplify fp32 model
                onnx_config=DictConfig({**onnx_config, "simplify": False}),
                input_shapes=input_shapes,
                half_precision=False,
                model_name=os.path.basename(export_model_path),
            )
            if export_output is None:
                # This should not happen
                raise RuntimeError("Failed to export model")

            model_fp32 = onnx.load(export_model_path)
            test_data = {input_names[i]: inp[i].float().cpu().numpy() for i in range(len(inp))}
            log.warning("Attempting to convert model in mixed precision, this may take a while...")
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                # This function prints a lot of information that is not useful for the user
                model_fp16 = auto_convert_mixed_precision(
                    model_fp32, test_data, rtol=rtol, atol=atol, keep_io_types=False
                )
            onnx.save(model_fp16, export_model_path)

            onnx_model = onnx.load(export_model_path)
            # Check if ONNX model is valid
            onnx.checker.check_model(onnx_model)
            return True
        except Exception as e:
            raise RuntimeError(
                "Failed to export model with automatic mixed precision, check your model or disable ONNX export"
            ) from e
    else:
        log.info("Exported half precision ONNX model does not contain NaN values, model is stable")
        return True


def export_pytorch_model(model: nn.Module, output_path: str, model_name: str = "model.pth") -> str:
    """Export pytorch model's parameter dictionary using a deserialized state_dict.

    Args:
        model: PyTorch model to be exported
        output_path: Path to save the model
        model_name: Name of the exported model

    Returns:
        If the model is exported successfully, the path to the model is returned.

    """
    if isinstance(model, ModelSignatureWrapper):
        model = model.instance

    os.makedirs(output_path, exist_ok=True)
    model.eval()
    model.cpu()
    model_path = os.path.join(output_path, model_name)
    torch.save(model.state_dict(), model_path)
    log.info("Pytorch model saved to %s", os.path.join(output_path, model_name))

    return os.path.join(os.getcwd(), model_path)


def export_model(
    config: DictConfig,
    model: Any,
    export_folder: str,
    half_precision: bool,
    input_shapes: list[Any] | None = None,
    idx_to_class: dict[int, str] | None = None,
    pytorch_model_type: Literal["backbone", "model"] = "model",
    example_inputs: list[Any] | tuple[Any, ...] | torch.Tensor | None = None,
    rtol: float = 0.01,
    atol: float = 5e-3,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Generate deployment models for the task.

    Args:
        config: Experiment config
        model: Model to be exported
        export_folder: Path to save the exported model
        half_precision: Whether to use half precision for the exported model
        input_shapes: Input shapes for the exported model
        idx_to_class: Mapping from class index to class name
        pytorch_model_type: Type of the pytorch model config to be exported, if it's backbone on disk we will save the
            config.backbone config, otherwise we will save the config.model
        example_inputs: If provided use this to evaluate the model instead of generating random inputs
        rtol: Relative tolerance for the ONNX safe export in fp16
        atol: Absolute tolerance for the ONNX safe export in fp16

    Returns:
        If the model is exported successfully, return a dictionary containing information about the exported model and
        a second dictionary containing the paths to the exported models. Otherwise, return two empty dictionaries.
    """
    if config.export is None or len(config.export.types) == 0:
        log.info("No export type specified skipping export")
        return {}, {}

    os.makedirs(export_folder, exist_ok=True)

    if input_shapes is None:
        # Try to get input shapes from config
        # If this is also None we will try to retrieve it from the ModelSignatureWrapper, if it fails we can't export
        input_shapes = config.export.input_shapes

    export_paths = {}

    for export_type in config.export.types:
        if export_type == "torchscript":
            out = export_torchscript_model(
                model=model,
                input_shapes=input_shapes,
                output_path=export_folder,
                half_precision=half_precision,
                example_inputs=example_inputs,
            )

            if out is None:
                log.warning("Torchscript export failed, enable debug logging for more details")
                continue

            export_path, input_shapes = out
            export_paths[export_type] = export_path
        elif export_type == "pytorch":
            export_path = export_pytorch_model(
                model=model,
                output_path=export_folder,
            )
            export_paths[export_type] = export_path
            with open(os.path.join(export_folder, "model_config.yaml"), "w") as f:
                OmegaConf.save(getattr(config, pytorch_model_type), f, resolve=True)
        elif export_type == "onnx":
            if not hasattr(config.export, "onnx"):
                log.warning("No onnx configuration found, skipping onnx export")
                continue

            out = export_onnx_model(
                model=model,
                output_path=export_folder,
                onnx_config=config.export.onnx,
                input_shapes=input_shapes,
                half_precision=half_precision,
                example_inputs=example_inputs,
                rtol=rtol,
                atol=atol,
            )

            if out is None:
                log.warning("ONNX export failed, enable debug logging for more details")
                continue

            export_path, input_shapes = out
            export_paths[export_type] = export_path
        else:
            log.warning("Export type: %s not implemented", export_type)

    if len(export_paths) == 0:
        log.warning("No export type was successful, no model will be available for deployment")
        return {}, export_paths

    model_json = {
        "input_size": input_shapes,
        "classes": idx_to_class,
        "mean": list(config.transforms.mean),
        "std": list(config.transforms.std),
    }

    return model_json, export_paths


def import_deployment_model(
    model_path: str,
    inference_config: DictConfig,
    device: str,
    model_architecture: nn.Module | None = None,
) -> BaseEvaluationModel:
    """Try to import a model for deployment, currently only supports torchscript .pt files and
    state dictionaries .pth files.

    Args:
        model_path: Path to the model
        inference_config: Inference configuration, should contain keys for the different deployment models
        device: Device to load the model on
        model_architecture: Optional model architecture to use for loading a plain pytorch model

    Returns:
        A tuple containing the model and the model type
    """
    log.info("Importing trained model")

    file_extension = os.path.splitext(os.path.basename(model_path))[1]
    deployment_model: BaseEvaluationModel | None = None

    if file_extension == ".pt":
        deployment_model = TorchscriptEvaluationModel(config=inference_config.torchscript)
    elif file_extension == ".pth":
        if model_architecture is None:
            raise ValueError("model_architecture must be specified when loading a .pth file")

        deployment_model = TorchEvaluationModel(config=inference_config.pytorch, model_architecture=model_architecture)
    elif file_extension == ".onnx":
        deployment_model = ONNXEvaluationModel(config=inference_config.onnx)

    if deployment_model is not None:
        deployment_model.load_from_disk(model_path=model_path, device=device)

        log.info("Imported %s model", deployment_model.__class__.__name__)

        return deployment_model

    raise ValueError(f"Unable to load model with extension {file_extension}, valid extensions are: ['.pt', 'pth']")


# This may be better as a dict?
def get_export_extension(export_type: str) -> str:
    """Get the extension of the exported model.

    Args:
        export_type: The type of the exported model.

    Returns:
        The extension of the exported model.
    """
    if export_type == "onnx":
        extension = "onnx"
    elif export_type == "torchscript":
        extension = "pt"
    elif export_type == "pytorch":
        extension = "pth"
    else:
        raise ValueError(f"Unsupported export type {export_type}")

    return extension
