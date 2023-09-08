import os
from pathlib import Path

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu", help="device to run tests on")


@pytest.fixture(autouse=True)
def change_test_dir(tmp_path: Path, monkeypatch):
    """Change the current working directory to a test folder."""
    test_working_directory = tmp_path / "test_working_directory"
    os.makedirs(test_working_directory, exist_ok=True)

    monkeypatch.chdir(test_working_directory)


@pytest.fixture(scope="session")
def device(pytestconfig):
    return pytestconfig.getoption("device")


@pytest.fixture(autouse=True)
def setup_devices(device: str):
    """Set the device to run tests on."""
    torch_device = torch.device(device)
    os.environ["QUADRA_TEST_DEVICE"] = device
    if torch_device.type != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
