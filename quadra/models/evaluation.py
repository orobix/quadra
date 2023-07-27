from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.jit import RecursiveScriptModule

try:
    import onnxruntime as ort  # noqa

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class BaseEvaluationModel(ABC):
    """Base interface for all evaluation models."""

    def __init__(self, config: DictConfig) -> None:
        self.model: Any
        self.model_path: str | None
        self.device: str
        self.config = config

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

    @property
    def training(self) -> bool:
        """Return whether model is in training mode."""
        return False


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

        self.model = model

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


class TorchEvaluationModel(TorchscriptEvaluationModel):
    """Wrapper for torch models.

    Args:
        model: Torch model to wrap.
    """

    def __init__(self, model: torch.nn.Module, config: DictConfig) -> None:
        self.model = model

        # Extract device from model
        self.device = str(next(self.model.parameters()).device)
        super().__init__(config=config)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def load_from_disk(self, model_path: str, device: str = "cpu"):
        """Load model from disk."""
        self.model_path = model_path
        self.device = device

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.model.to(self.device)


class ONNXEvaluationModel(BaseEvaluationModel):
    """Wrapper for ONNX models. It's designed to provide a similar interface to standard torch models."""

    def __init__(self, config: DictConfig) -> None:
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. Please install ONNX capabilities for quadra with: pip install .[onnx]"
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
                if isinstance(value, dict) and "_target_" in value:
                    value = instantiate(value)

                setattr(session_options, key, value)

        return session_options

    def __call__(self, inputs: list[np.ndarray] | np.ndarray | list[torch.Tensor] | torch.Tensor) -> Any:
        """Run inference on the model and return the output as torch tensors."""
        # TODO: If we get torch inputs we may be able to avoid the numpy conversion
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
        elif isinstance(inputs, list):
            inputs = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in inputs]

        if isinstance(inputs, np.ndarray):
            ort_inputs = {self.model.get_inputs()[0].name: inputs}
        else:
            ort_inputs = {x.name: y for x, y in zip(self.model.get_inputs(), inputs)}

        ort_outputs = [x.name for x in self.model.get_outputs()]

        onnx_output = self.model.run(ort_outputs, ort_inputs)

        onnx_output = [torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x for x in onnx_output]

        if len(onnx_output) == 1:
            onnx_output = onnx_output[0]

        return onnx_output

    def load_from_disk(self, model_path: str, device: str = "cpu"):
        """Load model from disk."""
        self.model_path = model_path
        self.device = device

        ort_providers = self._get_providers(device)
        self.model = ort.InferenceSession(self.model_path, providers=ort_providers, sess_options=self.session_options)

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

    def half(self):
        """Convert model to half precision."""
        raise NotImplementedError("At the moment ONNX models do not support half method.")
