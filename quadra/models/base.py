from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from quadra.utils.logger import get_logger

log = get_logger(__name__)


class ModelSignatureWrapper(nn.Module):
    """Model wrapper used to retrieve input shape. It can be used as a decorator of nn.Module, the first call to the
    forward method will retrieve the input shape and store it in the input_shapes attribute.
    It will also save the model summary in a file called model_summary.txt in the current working directory.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.instance = model
        self.input_shapes: Any = None
        self.disable = False

        if isinstance(self.instance, ModelSignatureWrapper):
            # Handle nested ModelSignatureWrapper
            self.input_shapes = self.instance.input_shapes
            self.instance = self.instance.instance

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Retrieve the input shape and forward the model, if the input shape is already retrieved it will just forward
        the model.
        """
        if self.input_shapes is None and not self.disable:
            try:
                self.input_shapes = self._get_input_shapes(*args, **kwargs)
            except Exception:
                log.warning(
                    "Failed to retrieve input shapes after forward! To export the model you'll need to "
                    "provide the input shapes manually setting the config.export.input_shapes parameter! "
                    "Alternatively you could try to use a forward with supported input types (and their compositions) "
                    "(list, tuple, dict, tensors)."
                )
                self.disable = True

        return self.instance.forward(*args, **kwargs)

    def to(self, *args, **kwargs):
        """Handle calls to to method returning the underlying model."""
        self.instance = self.instance.to(*args, **kwargs)

        return self

    def half(self, *args, **kwargs):
        """Handle calls to to method returning the underlying model."""
        self.instance = self.instance.half(*args, **kwargs)

        return self

    def cpu(self, *args, **kwargs):
        """Handle calls to to method returning the underlying model."""
        self.instance = self.instance.cpu(*args, **kwargs)

        return self

    def _get_input_shapes(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Retrieve the input shapes from the input. Inputs will be in the same order as the forward method
        signature.
        """
        input_shapes = []

        for arg in args:
            input_shapes.append(self._get_input_shape(arg))

        if isinstance(self.instance.forward, torch.ScriptMethod):
            # Handle torchscript backbones
            for i, argument in enumerate(self.instance.forward.schema.arguments):  # type: ignore[attr-defined]
                if i < (len(args) + 1):  # +1 for self
                    continue

                if argument.name == "self":
                    continue

                if argument.name in kwargs:
                    input_shapes.append(self._get_input_shape(kwargs[argument.name]))
                else:
                    # Retrieve the default value
                    input_shapes.append(self._get_input_shape(argument.default_value))
        else:
            signature = inspect.signature(self.instance.forward)

            for i, key in enumerate(signature.parameters.keys()):
                if i < len(args):
                    continue

                if key in kwargs:
                    input_shapes.append(self._get_input_shape(kwargs[key]))
                else:
                    # Retrieve the default value
                    input_shapes.append(self._get_input_shape(signature.parameters[key].default))

        return input_shapes

    def _get_input_shape(self, inp: Sequence | torch.Tensor) -> list[Any] | tuple[Any, ...] | dict[str, Any]:
        """Recursive function to retrieve the input shapes."""
        if isinstance(inp, list):
            return [self._get_input_shape(i) for i in inp]

        if isinstance(inp, tuple):
            return tuple(self._get_input_shape(i) for i in inp)

        if isinstance(inp, torch.Tensor):
            return tuple(inp.shape[1:])

        if isinstance(inp, dict):
            return {k: self._get_input_shape(v) for k, v in inp.items()}

        raise ValueError(f"Input type {type(inp)} not supported")

    def __getattr__(self, name: str) -> torch.Tensor | nn.Module:
        if name in ["instance", "input_shapes"]:
            return self.__dict__[name]

        return getattr(self.__dict__["instance"], name)

    def __setattr__(self, name: str, value: torch.Tensor | nn.Module) -> None:
        if name in ["instance", "input_shapes"]:
            self.__dict__[name] = value
        else:
            setattr(self.instance, name, value)

    def __getattribute__(self, __name: str) -> Any:
        if __name in [
            "instance",
            "input_shapes",
            "__dict__",
            "forward",
            "_get_input_shapes",
            "_get_input_shape",
            "to",
            "half",
            "cpu",
            "call_super_init",
            "_call_impl",
            "_compiled_call_impl",
        ]:
            return super().__getattribute__(__name)

        return getattr(self.instance, __name)
