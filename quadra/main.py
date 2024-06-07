import time

import hydra
from omegaconf import DictConfig

from quadra.utils.resolver import register_resolvers

register_resolvers()


# pylint: disable=import-outside-toplevel
@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.3.0")
def main(config: DictConfig):
    """Main entry function for any of the tasks."""
    # Import everything here to allow setting environment variables before importing torch
    import torch
    from pytorch_lightning import seed_everything

    from quadra.tasks import Task
    from quadra.utils import utils
    from quadra.utils.utils import get_logger, load_envs, setup_opencv
    from quadra.utils.validator import validate_config

    log = get_logger(__name__)
    log.debug("Running with %d torch threads: ", torch.get_num_threads())

    load_envs()
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
