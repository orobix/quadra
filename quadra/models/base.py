from __future__ import annotations

import inspect
from typing import Any, Sequence

import torch
from torch import nn


class ModelSignatureWrapper(nn.Module):
    """Model wrapper used to retrieve input shape. It can be used as a decorator of nn.Module, the first call to the
    forward method will retrieve the input shape and store it in the input_shapes attribute.
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
        """Retrieve the input shape and forward the model."""
        if self.input_shapes is None and not self.disable:
            try:
                self.input_shapes = self._get_input_shapes(*args, **kwargs)
            except Exception:
                # Avoid circular import
                # pylint: disable=import-outside-toplevel
                from quadra.utils.utils import get_logger  # noqa

                log = get_logger(__name__)
                log.warning(
                    "Failed to retrieve input shapes after forward! To export the model you'll need to "
                    "provide the input shapes manually setting the export_config.input_shapes parameter! "
                    "Alternatively you could try to use a forward with supported input types (and their compositions) "
                    "(list, tuple, dict, tensors)."
                )
                self.disable = True

        return self.instance.forward(*args, **kwargs)

    def to(self, *args, **kwargs):
        """Handle calls to to method returning the underlying model."""
        return ModelSignatureWrapper(self.instance.to(*args, **kwargs))

    def _get_input_shapes(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Retrieve the input shapes from the input. Inputs will be in the same order as the forward method
        signature.
        """
        input_shapes = []

        for arg in args:
            input_shapes.append(self._get_input_shape(arg))

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
        if __name in ["instance", "input_shapes", "__dict__", "forward", "_get_input_shapes", "_get_input_shape", "to"]:
            return super().__getattribute__(__name)

        return getattr(self.instance, __name)