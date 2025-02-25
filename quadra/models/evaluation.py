from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.jit import RecursiveScriptModule

from quadra.utils.logger import get_logger

try:
    import onnxruntime as ort  # noqa

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


log = get_logger(__name__)


class BaseEvaluationModel(ABC):
    """Base interface for all evaluation models."""

    def __init__(self, config: DictConfig) -> None:
        self.model: Any
        self.model_path: str | None
        self.device: str
        self.config = config
        self.is_loaded = False
        self.model_dtype: np.dtype | torch.dtype

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def load_from_disk(self, model_path: str, device: str = "cpu"):
        """Load model from disk."""

    @abstractmethod
    def to(self, device: str):
        """Move model to device."""

    @abstractmethod
    def eval(self):
        """Set model to evaluation mode."""

    @abstractmethod
    def half(self):
        """Convert model to half precision."""

    @abstractmethod
    def cpu(self):
        """Move model to cpu."""

    @property
    def training(self) -> bool:
        """Return whether model is in training mode."""
        return False

    @property
    def device(self) -> str:
        """Return the device of the model."""
        return self._device

    @device.setter
    def device(self, device: str):
        """Set the device of the model."""
        if device == "cuda" and ":" not in device:
            device = f"{device}:0"

        self._device = device


class TorchscriptEvaluationModel(BaseEvaluationModel):
    """Wrapper for torchscript models."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def load_from_disk(self, model_path: str, device: str = "cpu"):
        """Load model from disk."""
        self.model_path = model_path
        self.device = device

        model = cast(RecursiveScriptModule, torch.jit.load(self.model_path))
        model.eval()
        model.to(self.device)

        parameter_types = {param.dtype for param in model.parameters()}
        if len(parameter_types) == 2:
            # TODO: There could be models with mixed precision?
            raise ValueError(f"Expected only one type of parameters, found {parameter_types}")

        self.model_dtype = list(parameter_types)[0]
        self.model = model
        self.is_loaded = True

    def to(self, device: str):
        """Move model to device."""
        self.model.to(device)
        self.device = device

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()

    @property
    def training(self) -> bool:
        """Return whether model is in training mode."""
        return self.model.training

    def half(self):
        """Convert model to half precision."""
        self.model.half()

    def cpu(self):
        """Move model to cpu."""
        self.model.cpu()


class TorchEvaluationModel(TorchscriptEvaluationModel):
    """Wrapper for torch models.

    Args:
        model_architecture: Optional torch model architecture
    """

    def __init__(self, config: DictConfig, model_architecture: nn.Module) -> None:
        super().__init__(config=config)
        self.model = model_architecture
        self.model.eval()
        device = next(self.model.parameters()).device
        self.device = str(device)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def load_from_disk(self, model_path: str, device: str = "cpu"):
        """Load model from disk."""
        self.model_path = model_path
        self.device = device
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.model.to(self.device)

        parameter_types = {param.dtype for param in self.model.parameters()}
        if len(parameter_types) == 2:
            # TODO: There could be models with mixed precision?
            raise ValueError(f"Expected only one type of parameters, found {parameter_types}")

        self.model_dtype = list(parameter_types)[0]
        self.is_loaded = True


onnx_to_torch_dtype_dict = {
    "tensor(bool)": torch.bool,
    "tensor(uint8)": torch.uint8,
    "tensor(int8)": torch.int8,
    "tensor(int16)": torch.int16,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
    "tensor(float16)": torch.float16,
    "tensor(float32)": torch.float32,
    "tensor(float)": torch.float32,
    "tensor(float64)": torch.float64,
    "tensor(complex64)": torch.complex64,
    "tensor(complex128)": torch.complex128,
}


class ONNXEvaluationModel(BaseEvaluationModel):
    """Wrapper for ONNX models. It's designed to provide a similar interface to standard torch models."""

    def __init__(self, config: DictConfig) -> None:
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. Please install ONNX capabilities for quadra with: poetry install -E onnx"
            )
        super().__init__(config=config)
        self.session_options = self.generate_session_options()

    def generate_session_options(self) -> ort.SessionOptions:
        """Generate session options from the current config."""
        session_options = ort.SessionOptions()

        if hasattr(self.config, "session_options") and self.config.session_options is not None:
            session_options_dict = cast(
                dict[str, Any], OmegaConf.to_container(self.config.session_options, resolve=True)
            )
            for key, value in session_options_dict.items():
                final_value = value
                if isinstance(value, dict) and "_target_" in value:
                    final_value = instantiate(final_value)

                setattr(session_options, key, final_value)

        return session_options

    def __call__(self, *inputs: np.ndarray | torch.Tensor) -> Any:
        """Run inference on the model and return the output as torch tensors."""
        # TODO: Maybe we can support also kwargs
        use_pytorch = False

        onnx_inputs: dict[str, np.ndarray | torch.Tensor] = {}

        for onnx_input, current_input in zip(self.model.get_inputs(), inputs, strict=False):
            if isinstance(current_input, torch.Tensor):
                onnx_inputs[onnx_input.name] = current_input
                use_pytorch = True
            elif isinstance(current_input, np.ndarray):
                onnx_inputs[onnx_input.name] = current_input
            else:
                raise ValueError(f"Invalid input type: {type(inputs)}")

            if use_pytorch and isinstance(current_input, np.ndarray):
                raise ValueError("Cannot mix torch and numpy inputs")

        if use_pytorch:
            onnx_output = self._forward_from_pytorch(cast(dict[str, torch.Tensor], onnx_inputs))
        else:
            onnx_output = self._forward_from_numpy(cast(dict[str, np.ndarray], onnx_inputs))

        onnx_output = [torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x for x in onnx_output]

        if len(onnx_output) == 1:
            onnx_output = onnx_output[0]

        return onnx_output

    def _forward_from_pytorch(self, input_dict: dict[str, torch.Tensor]):
        """Run inference on the model and return the output as torch tensors."""
        io_binding = self.model.io_binding()
        device_type = self.device.split(":")[0]

        for k, v in input_dict.items():
            if not v.is_contiguous():
                # If not contiguous onnx give wrong results
                v = v.contiguous()  # noqa: PLW2901

            io_binding.bind_input(
                name=k,
                device_type=device_type,
                # Weirdly enough onnx wants 0 for cpu
                device_id=0 if device_type == "cpu" else int(self.device.split(":")[1]),
                element_type=np.float16 if v.dtype == torch.float16 else np.float32,
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr(),
            )

        for x in self.model.get_outputs():
            # TODO: Is it possible to also bind the output? We require info about output dimensions
            io_binding.bind_output(name=x.name)

        self.model.run_with_iobinding(io_binding)

        output = io_binding.copy_outputs_to_cpu()

        return output

    def _forward_from_numpy(self, input_dict: dict[str, np.ndarray]):
        """Run inference on the model and return the output as numpy array."""
        ort_outputs = [x.name for x in self.model.get_outputs()]

        onnx_output = self.model.run(ort_outputs, input_dict)

        return onnx_output

    def load_from_disk(self, model_path: str, device: str = "cpu"):
        """Load model from disk."""
        self.model_path = model_path
        self.device = device

        ort_providers = self._get_providers(device)
        self.model = ort.InferenceSession(self.model_path, providers=ort_providers, sess_options=self.session_options)
        self.model_dtype = self.cast_onnx_dtype(self.model.get_inputs()[0].type)
        self.is_loaded = True

    def _get_providers(self, device: str) -> list[tuple[str, dict[str, Any]] | str]:
        """Return the providers for the ONNX model based on the device."""
        ort_providers: list[tuple[str, dict[str, Any]] | str]

        if device == "cpu":
            ort_providers = ["CPUExecutionProvider"]
        else:
            ort_providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": int(device.split(":")[1]),
                    },
                )
            ]

        return ort_providers

    def to(self, device: str):
        """Move model to device."""
        self.device = device
        ort_providers = self._get_providers(device)
        self.model.set_providers(ort_providers)

    def eval(self):
        """Fake interface to match torch models."""
        return self

    def half(self):
        """Convert model to half precision."""
        raise NotImplementedError("At the moment ONNX models do not support half method.")

    def cpu(self):
        """Move model to cpu."""
        self.to("cpu")

    def cast_onnx_dtype(self, onnx_dtype: str) -> torch.dtype | np.dtype:
        """Cast ONNX dtype to numpy or pytorch dtype."""
        return onnx_to_torch_dtype_dict[onnx_dtype]
