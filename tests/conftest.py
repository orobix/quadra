import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def change_test_dir(tmp_path: Path, monkeypatch):
    """Change the current working directory to a test folder."""
    test_working_directory = tmp_path / "test_working_directory"
    os.makedirs(test_working_directory, exist_ok=True)

    monkeypatch.chdir(test_working_directory)


@pytest.fixture(autouse=True)
def hyde_gpu():
    """Hyde GPU for tests."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
