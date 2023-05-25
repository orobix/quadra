import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig

from quadra.main import main


# taken from hydra unit tests
def _random_image(size: Tuple[int, int] = (10, 10)) -> np.ndarray:
    """Generate random image."""
    return np.random.randint(0, 255, size=size, dtype=np.uint8)


def execute_quadra_experiment(overrides: List[str], experiment_path: Path) -> None:
    """Execute quadra experiment."""
    with initialize_config_module(config_module="quadras", version_base="1.3.0"):
        if not experiment_path.exists():
            experiment_path.mkdir(parents=True)
        os.chdir(experiment_path)
        # cfg = compose(config_name="config", overrides=overrides)
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        # workaround without actual main function
        # check https://github.com/facebookresearch/hydra/issues/2017 for more details
        HydraConfig.instance().set_config(cfg)

        main(cfg)
