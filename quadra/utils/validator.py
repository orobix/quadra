from __future__ import annotations

import difflib
import importlib
import inspect
from collections.abc import Iterable
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from quadra.utils.utils import get_logger

OMEGACONF_FIELDS: tuple[str, ...] = ("_target_", "_convert_", "_recursive_", "_args_")
EXCLUDE_KEYS: tuple[str, ...] = ("hydra",)

logger = get_logger(__name__)


def get_callable_arguments(full_module_path: str) -> tuple[list[str], bool]:
    """Gets all arguments from module path.

    Args:
        full_module_path:  Full module path to the target class or function.

    Raises:
        ValueError: If the target is not a class or a function.

    Returns:
        All arguments from the target class or function.
    """
    module_path, callable_name = full_module_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    callable_ = getattr(module, callable_name)
    # check if it is a class
    accepts_kwargs = False
    if inspect.isclass(callable_):
        arg_names = []
        for cls in callable_.__mro__:
            if cls is object:
                break
            # We don' access the instance but mypy complains
            init_argspec = inspect.getfullargspec(cls.__init__)  # type: ignore
            cls_arg_names = init_argspec.args[1:]
            cls_kwonlyargs = init_argspec.kwonlyargs
            arg_names.extend(cls_arg_names)
            arg_names.extend(cls_kwonlyargs)
            # if the target class or function accepts kwargs, we cannot check arguments
            accepts_kwargs = init_argspec.varkw is not None or accepts_kwargs
        arg_names = list(set(arg_names))
    elif inspect.isfunction(callable_):
        init_argspec = inspect.getfullargspec(callable_)
        arg_names = []
        arg_names.extend(init_argspec.args)
        arg_names.extend(init_argspec.kwonlyargs)
        accepts_kwargs = init_argspec.varkw is not None or accepts_kwargs
    else:
        raise ValueError("The target must be a class or a function.")

    return arg_names, accepts_kwargs


def check_all_arguments(callable_variable: str, configuration_arguments: list[str], argument_names: list[str]) -> None:
    """Checks if all arguments passed from configuration are valid for the target class or function.

    Args:
        callable_variable : Full module path to the target class or function.
        configuration_arguments : All arguments passed from configuration.
        argument_names: All arguments from the target class or function.

    Raises:
        ValueError: If the argument is not valid for the target class or function.
    """
    for argument in configuration_arguments:
        if argument not in argument_names:
            error_string = f"`{argument}` is not a valid argument passed from configuration to `{callable_variable}`."
            closest_match = difflib.get_close_matches(argument, argument_names, n=1, cutoff=0.5)
            if len(closest_match) > 0:
                error_string += f" Did you mean `{closest_match[0]}`?"
            raise ValueError(error_string)


def validate_config(_cfg: DictConfig | ListConfig, package_name: str = "quadra") -> None:
    """Recursively traverse OmegaConf object and check if arguments are valid for the target class or function.
    If not, raise a ValueError with a suggestion for the closest match of the argument name.

    Args:
        _cfg: OmegaConf object
        package_name: package name to check for instantiation.
    """
    # The below lines of code for looping over a DictConfig/ListConfig are
    # borrowed from OmegaConf PR #719.
    itr: Iterable[Any]
    if isinstance(_cfg, ListConfig):
        itr = range(len(_cfg))
    else:
        itr = _cfg
    for key in itr:
        if OmegaConf.is_missing(_cfg, key):
            continue
        if isinstance(key, str) and any(x in key for x in EXCLUDE_KEYS):
            continue
        if OmegaConf.is_config(_cfg[key]):
            validate_config(_cfg[key])
        elif isinstance(_cfg[key], str):
            if key == "_target_":
                callable_variable = str(_cfg[key])
                if callable_variable.startswith(package_name):
                    configuration_arguments = [str(x) for x in _cfg if x not in OMEGACONF_FIELDS]
                    argument_names, accepts_kwargs = get_callable_arguments(callable_variable)
                    if not accepts_kwargs:
                        check_all_arguments(callable_variable, configuration_arguments, argument_names)
        else:
            logger.debug("Skipping %s from config. It is not supported.", key)
