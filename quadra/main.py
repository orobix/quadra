from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig

from quadra.utils.resolver import register_resolvers
from quadra.utils.utils import get_logger, load_envs, setup_opencv

if TYPE_CHECKING:
    from quadra.tasks import Task

load_envs()
register_resolvers()

log = get_logger(__name__)


@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.3.0")
def main(config: DictConfig):
    # pylint: disable=import-outside-toplevel
    """Main entry function for any of the tasks."""
    import time

    from pytorch_lightning import seed_everything

    from quadra.utils import utils
    from quadra.utils.validator import validate_config

    if config.validate:
        start = time.time()
        validate_config(config)
        stop = time.time()
        log.info("Config validation took %f seconds", stop - start)

    utils.extras(config)

    # Prints the resolved configuration to the console
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Set seed for random number generators in pytorch, numpy and python.random
    seed_everything(config.core.seed, workers=True)
    setup_opencv()

    # Run specified task using the configuration composition
    task: Task = hydra.utils.instantiate(config.task, config, _recursive_=False)
    task.execute()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
