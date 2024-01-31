from __future__ import annotations

import importlib

import pytest

try:
    importlib.import_module("mlflow")
except ImportError:
    pytest.skip("Mlflow is not installed", allow_module_level=True)

from typing import Sequence

import torch
from mlflow.models import infer_signature
from mlflow.types.schema import TensorSpec
from pytest import raises

from quadra.models.base import ModelSignatureWrapper
from quadra.utils.export import generate_torch_inputs
from quadra.utils.mlflow import infer_signature_input, infer_signature_model
from quadra.utils.tests.models import DoubleInputModel, SingleInputModel


def check_signature_equality(
    input_signature: Sequence[TensorSpec] | dict[str, TensorSpec] | TensorSpec,
    expected_signature: Sequence[TensorSpec] | dict[str, TensorSpec] | TensorSpec,
):
    """Assert that the generated signature matches the expected signature."""
    is_equal = True

    if isinstance(input_signature, Sequence) and isinstance(expected_signature, Sequence):
        for c_input_signature, c_expected_signature in zip(input_signature, expected_signature):
            is_equal = is_equal and check_signature_equality(c_input_signature, c_expected_signature)
    elif isinstance(input_signature, dict) and isinstance(expected_signature, dict):
        for k, v in input_signature.items():
            is_equal = is_equal and check_signature_equality(v, expected_signature[k])
    elif isinstance(input_signature, TensorSpec) and isinstance(expected_signature, TensorSpec):
        is_equal = (
            input_signature.type == expected_signature.type
            and input_signature.shape == expected_signature.shape
            and input_signature.name == expected_signature.name
        )
    else:
        is_equal = False

    return is_equal


@torch.inference_mode()
def test_single_tensor_signature():
    """Test the input shape retrieval for a simple model."""
    model = ModelSignatureWrapper(SingleInputModel())

    x = torch.zeros(1, 3, 224, 224)
    _ = model(x)

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")

    signature = infer_signature_model(model, inputs)

    expected_input_signature = [TensorSpec(shape=(-1, *x.shape[1:]), type=x.numpy().dtype)]

    # The model output is the same as the input
    assert check_signature_equality(signature.inputs.inputs, expected_input_signature)
    assert check_signature_equality(signature.outputs.inputs, expected_input_signature)


@torch.inference_mode()
def test_multiple_tensor_signature():
    """Test the input shape retrieval for a model with multiple inputs."""
    model = ModelSignatureWrapper(DoubleInputModel())
    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    _ = model(x, y)
    assert model.input_shapes == [(3, 224, 224), (3, 448, 448)]
    inputs = generate_torch_inputs(model.input_shapes, device="cpu")

    signature = infer_signature_model(model, inputs)

    expected_input_signature = [
        TensorSpec(shape=(-1, *x.shape[1:]), type=x.numpy().dtype, name="output_0"),
        TensorSpec(shape=(-1, *y.shape[1:]), type=y.numpy().dtype, name="output_1"),
    ]

    # The model output is the same as the input
    assert check_signature_equality(signature.inputs.inputs, expected_input_signature)
    assert check_signature_equality(signature.outputs.inputs, expected_input_signature)


@torch.inference_mode()
def test_dict_signature():
    """Test the input shape retrieval for a model with a dict input."""
    model = ModelSignatureWrapper(SingleInputModel())

    x = torch.zeros(1, 3, 224, 224)
    _ = model({"x": x})

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")

    signature = infer_signature_model(model, inputs)

    expected_input_signature = [TensorSpec(shape=(-1, *x.shape[1:]), type=x.numpy().dtype, name="x")]

    # The model output is the same as the input
    assert check_signature_equality(signature.inputs.inputs, expected_input_signature)
    assert check_signature_equality(signature.outputs.inputs, expected_input_signature)


@torch.inference_mode()
def test_nested_tuple_signature():
    """Test the input shape retrieval for a model with a tuple input."""
    model = ModelSignatureWrapper(SingleInputModel())
    assert model.input_shapes is None

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    outputs = model((x, y))

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")

    expected_output_signature = [
        TensorSpec(shape=(-1, *x.shape[1:]), type=x.numpy().dtype, name="output_0"),
        TensorSpec(shape=(-1, *y.shape[1:]), type=y.numpy().dtype, name="output_1"),
    ]

    # Nested structures are not supported
    with raises(ValueError):
        infer_signature_input(inputs)

    signature = infer_signature(infer_signature_input(outputs), infer_signature_input(outputs))

    assert check_signature_equality(signature.outputs.inputs, expected_output_signature)


@torch.inference_mode()
def test_nested_list_signature():
    """Test the input shape retrieval for a model with a list of tensors input."""
    model = ModelSignatureWrapper(SingleInputModel())

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    outputs = model([x, y])

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")

    expected_output_signature = [
        TensorSpec(shape=(-1, *x.shape[1:]), type=x.numpy().dtype, name="output_0"),
        TensorSpec(shape=(-1, *y.shape[1:]), type=y.numpy().dtype, name="output_1"),
    ]

    # Nested structures are not supported
    with raises(ValueError):
        infer_signature_input(inputs)

    signature = infer_signature(infer_signature_input(outputs), infer_signature_input(outputs))

    assert check_signature_equality(signature.outputs.inputs, expected_output_signature)


@torch.inference_mode()
def test_nested_dicts_signature():
    """Test the input shape retrieval for a model with a list of dicts input."""
    model = ModelSignatureWrapper(SingleInputModel())

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    outputs = model([{"x": x}, {"y": y}])
    inputs = generate_torch_inputs(model.input_shapes, device="cpu")

    # Nested structures are not supported
    with raises(ValueError):
        infer_signature_input(inputs)

    with raises(ValueError):
        infer_signature_input(outputs)


@torch.inference_mode()
def test_tuple_of_dicts_signature():
    """Test the input shape retrieval for a model with a tuple of dicts input."""
    model = ModelSignatureWrapper(SingleInputModel())

    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 448, 448)
    outputs = model(({"x": x}, {"y": y}))

    inputs = generate_torch_inputs(model.input_shapes, device="cpu")

    # Nested structures are not supported
    with raises(ValueError):
        infer_signature_input(inputs)

    with raises(ValueError):
        infer_signature_input(outputs)
