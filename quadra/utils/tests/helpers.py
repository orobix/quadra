from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig

from quadra.main import main
from quadra.utils.export import get_export_extension


# taken from hydra unit tests
def _random_image(size: tuple[int, int] = (10, 10)) -> np.ndarray:
    """Generate random image."""
    return np.random.randint(0, 255, size=size, dtype=np.uint8)


def execute_quadra_experiment(overrides: list[str], experiment_path: Path) -> None:
    """Execute quadra experiment."""
    with initialize_config_module(config_module="quadra.configs", version_base="1.3.0"):
        if not experiment_path.exists():
            experiment_path.mkdir(parents=True)
        os.chdir(experiment_path)
        # cfg = compose(config_name="config", overrides=overrides)
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        # workaround without actual main function
        # check https://github.com/facebookresearch/hydra/issues/2017 for more details
        HydraConfig.instance().set_config(cfg)

        main(cfg)


def check_deployment_model(export_type: str):
    """Check that the runtime model is present and valid.

    Args:
        export_type: The type of the exported model.
    """
    extension = get_export_extension(export_type)

    assert os.path.exists(f"deployment_model/model.{extension}")
    assert os.path.exists("deployment_model/model.json")


def get_quadra_test_device():
    """Get the device to use for the tests. If the QUADRA_TEST_DEVICE environment variable is set, it is used."""
    return os.environ.get("QUADRA_TEST_DEVICE", "cpu")


def setup_trainer_for_lightning() -> list[str]:
    """Setup trainer for lightning depending on the device. If cuda is used, the device index is also set.
    If cpu is used, the trainer is set to lightning_cpu.

    Returns:
        A list of overrides for the trainer.
    """
    overrides = []
    device = get_quadra_test_device()
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        device_index = torch_device.index
        overrides.append("trainer=lightning_gpu")
        overrides.append(f"trainer.devices=[{device_index}]")
    else:
        overrides.append("trainer=lightning_cpu")

    return overrides
