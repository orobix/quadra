from __future__ import annotations

from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def multirun_subdir_beautify(subdir: str) -> str:
    """Change the subdir name to be more readable and usable, this function will replace / with | to avoid creating
    undesired subdirectories and remove the left part of the equals sign to avoid having too long names.

    Args:
        subdir: The subdir name.

    Returns:
        The beautified subdir name.

    Examples:
        >>> multirun_subdir_beautify("experiment=pippo/anomaly/padim,trainer.batch_size=32")
        "pippo|anomaly|padim,32"
    """
    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode is None or hydra_cfg.mode.name == "RUN":
        return subdir
    # Remove slashes to avoid creating multiple subdirs
    # TODO: if right side of the equals sign has `,` this will not work.
    subdir_list = subdir.replace("/", "|").split(",")
    subdir = ",".join([x.split("=")[1].replace(" ", "") for x in subdir_list])

    return subdir


def as_tuple(*args: Any) -> tuple[Any, ...]:
    """Resolves a list of arguments to a tuple."""
    return tuple(args)


def register_resolvers() -> None:
    """Register custom resolver."""
    OmegaConf.register_new_resolver("multirun_subdir_beautify", multirun_subdir_beautify)
    OmegaConf.register_new_resolver("as_tuple", as_tuple)
