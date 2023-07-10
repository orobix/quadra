from __future__ import annotations

from typing import Any

import torch
from torch import nn

from quadra.models.base import ModelSignatureWrapper
from quadra.utils.export import generate_torch_inputs


class SingleInputModel(nn.Module):
    """Model taking a single input."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Any):
        return x


class DoubleInputModel(nn.Module):
    """Model taking two inputs."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Any, y: Any):
        return x, y


class UnsupportedInputModel(nn.Module):
    """Model taking an unsupported input."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: str):
        return x


def test_simple_model():
    """Test the input shape retrieval for a simple model."""
    model = ModelSignatureWrapper(SingleInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    model(x)
    assert model.input_shapes == [(3, 224, 224)]

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")
    model(*inputs)


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

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")
    assert len(inputs) == 2
    assert inputs[0].shape == (1, 3, 224, 224)
    assert inputs[1].shape == (1, 3, 448, 448)
    model(*inputs)


def test_dict_input_model():
    """Test the input shape retrieval for a model with a dict input."""
    model = ModelSignatureWrapper(SingleInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    model({"x": x})
    assert model.input_shapes == [{"x": (3, 224, 224)}]

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")
    assert len(inputs) == 1
    assert isinstance(inputs[0], dict)
    assert inputs[0]["x"].shape == (1, 3, 224, 224)
    model(*inputs)


def test_tuple_input_model():
    """Test the input shape retrieval for a model with a tuple input."""
    model = ModelSignatureWrapper(SingleInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    model((x, y))
    assert model.input_shapes == [((3, 224, 224), (3, 448, 448))]

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")
    assert len(inputs) == 1
    assert len(inputs[0]) == 2
    assert isinstance(inputs[0], tuple)
    assert inputs[0][0].shape == (1, 3, 224, 224)
    assert inputs[0][1].shape == (1, 3, 448, 448)
    model(*inputs)


def test_list_of_tensors_input_model():
    """Test the input shape retrieval for a model with a list of tensors input."""
    model = ModelSignatureWrapper(SingleInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    model([x, y])
    assert model.input_shapes == [[(3, 224, 224), (3, 448, 448)]]

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")
    assert len(inputs) == 1
    assert isinstance(inputs[0], list)
    assert len(inputs[0]) == 2
    assert inputs[0][0].shape == (1, 3, 224, 224)
    assert inputs[0][1].shape == (1, 3, 448, 448)
    model(*inputs)


def test_list_of_dicts_input_model():
    """Test the input shape retrieval for a model with a list of dicts input."""
    model = ModelSignatureWrapper(SingleInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    model([{"x": x}, {"y": y}])
    assert model.input_shapes == [[{"x": (3, 224, 224)}, {"y": (3, 448, 448)}]]

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")
    assert len(inputs) == 1
    assert isinstance(inputs[0], list)
    assert len(inputs[0]) == 2
    assert inputs[0][0]["x"].shape == (1, 3, 224, 224)
    assert inputs[0][1]["y"].shape == (1, 3, 448, 448)
    model(*inputs)


def test_tuple_of_dicts_input_model():
    """Test the input shape retrieval for a model with a tuple of dicts input."""
    model = ModelSignatureWrapper(SingleInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    model(({"x": x}, {"y": y}))
    assert model.input_shapes == [({"x": (3, 224, 224)}, {"y": (3, 448, 448)})]

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")
    assert len(inputs) == 1
    assert isinstance(inputs[0], tuple)
    assert len(inputs[0]) == 2
    assert inputs[0][0]["x"].shape == (1, 3, 224, 224)
    assert inputs[0][1]["y"].shape == (1, 3, 448, 448)
    model(*inputs)


def test_unsupported_input_model():
    """Test the input shape retrieval for a model with an unsupported input."""
    model = ModelSignatureWrapper(UnsupportedInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    y = "test"
    model(x, y)
    assert model.input_shapes is None
    assert model.disable
