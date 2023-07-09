from __future__ import annotations

import torch
from torch import nn

from quadra.models.base import ModelSignatureWrapper


class SimpleModel(nn.Module):
    """Model taking a single input."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


class DoubleInputModel(nn.Module):
    """Model taking two inputs."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x, y


class DictInputModel(nn.Module):
    """Model taking a dict as input."""

    def __init__(self):
        super().__init__()

    def forward(self, x: dict[str, torch.Tensor]):
        return x


class TupleInputModel(nn.Module):
    """Model taking a tuple as input."""

    def __init__(self):
        super().__init__()

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]):
        return x


class ListOfTensorsInputModel(nn.Module):
    """Model taking a list of tensors as input."""

    def __init__(self):
        super().__init__()

    def forward(self, x: list[torch.Tensor]):
        return x


class ListOfDictsInputModel(nn.Module):
    """Model taking a list of dicts as input."""

    def __init__(self):
        super().__init__()

    def forward(self, x: list[dict[str, torch.Tensor]]):
        return x


class UnsupportedInputModel(nn.Module):
    """Model taking an unsupported input."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: str):
        return x


def test_simple_model():
    """Test the input shape retrieval for a simple model."""
    model = ModelSignatureWrapper(SimpleModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    model(x)
    assert model.input_shapes == [(3, 224, 224)]


def test_double_input_model():
    """Test the input shape retrieval for a model with multiple inputs."""
    model = ModelSignatureWrapper(DoubleInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    model(x, y)
    assert model.input_shapes == [(3, 224, 224), (3, 448, 448)]

    model.input_shapes = None
    model(y=y, x=x)
    # Check that the order of the inputs does not matter
    assert model.input_shapes == [(3, 224, 224), (3, 448, 448)]


def test_dict_input_model():
    """Test the input shape retrieval for a model with a dict input."""
    model = ModelSignatureWrapper(DictInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    model({"x": x})
    assert model.input_shapes == [{"x": (3, 224, 224)}]


def test_tuple_input_model():
    """Test the input shape retrieval for a model with a tuple input."""
    model = ModelSignatureWrapper(TupleInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    model((x, y))
    assert model.input_shapes == [((3, 224, 224), (3, 448, 448))]


def test_list_of_tensors_input_model():
    """Test the input shape retrieval for a model with a list of tensors input."""
    model = ModelSignatureWrapper(ListOfTensorsInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    model([x, y])
    assert model.input_shapes == [[(3, 224, 224), (3, 448, 448)]]


def test_list_of_dicts_input_model():
    """Test the input shape retrieval for a model with a list of dicts input."""
    model = ModelSignatureWrapper(ListOfDictsInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    model([{"x": x}, {"y": y}])
    assert model.input_shapes == [[{"x": (3, 224, 224)}, {"y": (3, 448, 448)}]]


def test_unsupported_input_model():
    """Test the input shape retrieval for a model with an unsupported input."""
    model = ModelSignatureWrapper(UnsupportedInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    y = "test"
    model(x, y)
    assert model.input_shapes is None
    assert model.disable
