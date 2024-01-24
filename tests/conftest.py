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
def limit_torch_threads():
    """Limit the number of threads used by pytorch to 4."""
    torch.set_num_threads(4)


@pytest.fixture(autouse=True)
def setup_devices(device: str):
    """Set the device to run tests on."""
    # torch_device = torch.device(device)
    os.environ["QUADRA_TEST_DEVICE"] = device

    # TODO: If we use this lightning crashes because it sees gpus but no gpu are available!!
    # if torch_device.type != "cuda":
    #    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
