from typing import Any, List, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.modules.module import Module


class ModelWrapper(nn.Module):
    """Model wrapper to retrieve input shape."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.instance = model
        self.input_shapes: Any = None

        if isinstance(self.instance, ModelWrapper):
            # Handle nested ModelWrapper
            self.input_shapes = self.instance.input_shapes
            self.instance = self.instance.instance

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Retrieve the input shape and forward the model."""
        if self.input_shapes is None:
            self.input_shapes = self._get_input_shapes(*args, **kwargs)

        return self.instance.forward(*args, **kwargs)

    def _get_input_shapes(self, *args, **kwargs) -> List[Any]:
        """Retrieve the input shapes from the input."""
        input_shapes = []
        for arg in args:
            input_shapes.append(self._get_input_shape(arg))

        # TODO: This is probably incorrect as we are not considering the order of the kwargs
        for kwarg in kwargs.values():
            input_shapes.append(self._get_input_shape(kwarg))

        return input_shapes

    def _get_input_shape(self, inp) -> Union[List[Any], Tuple[int, ...]]:
        """Recursive function to retrieve the input shapes."""
        # TODO: Do we need to support dicts?
        if isinstance(inp, Sequence):
            return [self._get_input_shape(i) for i in inp]

        if isinstance(inp, torch.Tensor):
            return tuple(inp.shape[1:])

        raise ValueError(f"Input type {type(inp)} not supported")

    def __getattr__(self, name: str) -> Union[Tensor, Module]:
        if name in ["instance", "input_shapes"]:
            return self.__dict__[name]

        return getattr(self.__dict__["instance"], name)

    def __setattr__(self, name: str, value: Union[Tensor, Module]) -> None:
        if name in ["instance", "input_shapes"]:
            self.__dict__[name] = value
        else:
            setattr(self.instance, name, value)

    def __getattribute__(self, __name: str) -> Any:
        if __name in ["instance", "input_shapes", "__dict__", "forward", "_get_input_shapes", "_get_input_shape"]:
            return super().__getattribute__(__name)

        return getattr(self.instance, __name)
