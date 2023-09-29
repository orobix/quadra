import os
from pathlib import Path
from typing import Callable, Generator

import pytest
import torch
from pytest_mock import MockerFixture

from quadra.tasks.base import LightningTask


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu", help="device to run tests on")
    parser.addoption("--mock-training", action="store_true", default=False, help="mock training")


@pytest.fixture
def mock_training(pytestconfig: pytest.Config, mocker: Callable[..., Generator[MockerFixture, None, None]]):
    """Mock the training of the model."""
    if pytestconfig.getoption("mock_training"):
        mocker.patch.object(LightningTask, "train", return_value=None)


@pytest.fixture(autouse=True)
def change_test_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Change the current working directory to a test folder."""
    test_working_directory = tmp_path / "test_working_directory"
    os.makedirs(test_working_directory, exist_ok=True)

    monkeypatch.chdir(test_working_directory)


@pytest.fixture(scope="session")
def device(pytestconfig: pytest.Config):
    return pytestconfig.getoption("device")


@pytest.fixture(autouse=True)
def limit_cpu_resources():
    """Limit the number of threads used by numpy and pytorch to 4."""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMBA_NUM_THREADS"] = "4"


@pytest.fixture(autouse=True)
def setup_devices(device: str):
    """Set the device to run tests on."""
    torch_device = torch.device(device)
    os.environ["QUADRA_TEST_DEVICE"] = device
    if torch_device.type != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
