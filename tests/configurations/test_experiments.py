import glob
from pathlib import Path
from typing import List

import pytest
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig

from quadra.utils.utils import load_envs
from quadra.utils.validator import validate_config


def get_experiment_configs(experiment_folder: str) -> List[str]:
    path = Path(__file__).parent.parent.parent / Path(f"quadra/configs/experiment/{experiment_folder}/**/*.yaml")
    experiment_paths = glob.glob(str(path), recursive=True)
    experiments: List[str] = []
    for path in experiment_paths:
        experiment_tag = path.split("experiment/")[-1]
        experiments.append(experiment_tag.split(".yaml")[0])
    return experiments


GENERIC_EXP_CONFIGS = get_experiment_configs(experiment_folder="generic")


@pytest.mark.parametrize("generic_experiment", GENERIC_EXP_CONFIGS)
def test_generic_experiments(generic_experiment):
    load_envs()
    with initialize_config_module(config_module="quadra.configs", version_base="1.3.0"):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={generic_experiment}", "logger=csv"],
            return_hydra_config=True,
        )
        # workaround without actual main function
        # check https://github.com/facebookresearch/hydra/issues/2017 for more details
        HydraConfig.instance().set_config(cfg)
        validate_config(cfg)
