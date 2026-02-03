"""Common utility functions.
Some of them are mostly based on https://github.com/ashleve/lightning-hydra-template.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import cv2
import dotenv
import numpy as np
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

import quadra

IMAGE_EXTENSIONS: list[str] = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm"]


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode.
    Modifies DictConfig in place.

    Args:
        config: Configuration composed by Hydra.
    """
    logging.basicConfig()
    logging.getLogger().setLevel(config.core.log_level.upper())

    log = get_logger(__name__)
    config.core.command += " ".join(sys.argv)
    config.core.experiment_path = os.getcwd()

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.get("trainer") and config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.devices = 1
            config.trainer.accelerator = "cpu"
            config.trainer.gpus = None
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "task",
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "core",
        "backbone",
        "transforms",
        "optimizer",
        "scheduler",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config: Configuration composed by Hydra.
        fields: Determines which main fields from config will
            be printed and in what order.
        resolve: Whether to resolve reference fields of DictConfig.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """
    log = get_logger(__name__)

    if not HydraConfig.initialized() or trainer.logger is None:
        return

    log.info("Logging hyperparameters!")
    hydra_cfg = HydraConfig.get()
    hydra_choices = OmegaConf.to_container(hydra_cfg.runtime.choices)
    if isinstance(hydra_choices, dict):
        # For multirun override the choices that are not automatically updated
        for item in hydra_cfg.overrides.task:
            if "." in item:
                continue

            override, value = item.split("=")
            hydra_choices[override] = value

        hparams = {}
        hydra_choices_final = {}
        for k, v in hydra_choices.items():
            if isinstance(k, str):
                k_replaced = k.replace("@", "_at_")
                hydra_choices_final[k_replaced] = v
                if v is not None and isinstance(v, str) and "@" in v:
                    hydra_choices_final[k_replaced] = v.replace("@", "_at_")

        hparams.update(hydra_choices_final)
    else:
        logging.warning("Hydra choices is not a dictionary, skip adding them to the logger")
    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    hparams["experiment_path"] = config.core.experiment_path
    hparams["command"] = config.core.command
    hparams["library/version"] = str(quadra.__version__)

    with open(os.devnull, "w") as fnull:
        if subprocess.call(["git", "-C", get_original_cwd(), "status"], stderr=subprocess.STDOUT, stdout=fnull) == 0:
            try:
                hparams["git/commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
                hparams["git/branch"] = (
                    subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
                )
                hparams["git/remote"] = (
                    subprocess.check_output(["git", "remote", "get-url", "origin"]).decode("ascii").strip()
                )
            except subprocess.CalledProcessError:
                log.warning(
                    "Could not get git commit, branch or remote information, the repository might not have any commits "
                    " yet or it might have been initialized wrongly."
                )
        else:
            log.warning("Could not find git repository, skipping git commit and branch info")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def upload_file_tensorboard(file_path: str, tensorboard_logger: TensorBoardLogger) -> None:
    """Upload a file to tensorboard handling different extensions.

    Args:
        file_path: Path to the file to upload.
        tensorboard_logger: Tensorboard logger instance.
    """
    tag = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".json":
        with open(file_path) as f:
            json_content = json.load(f)

            json_content = f"```json\n{json.dumps(json_content, indent=4)}\n```"
            tensorboard_logger.experiment.add_text(tag=tag, text_string=json_content, global_step=0)
    elif ext in [".yaml", ".yml"]:
        with open(file_path) as f:
            yaml_content = f.read()
            yaml_content = f"```yaml\n{yaml_content}\n```"
            tensorboard_logger.experiment.add_text(tag=tag, text_string=yaml_content, global_step=0)
    else:
        with open(file_path, encoding="utf-8") as f:
            tensorboard_logger.experiment.add_text(tag=tag, text_string=f.read().replace("\n", "  \n"), global_step=0)

    tensorboard_logger.experiment.flush()


def load_envs(env_file: str | None = None) -> None:
    """Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    Args:
        env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


def model_type_from_path(model_path: str) -> str | None:
    """Determine the type of the machine learning model based on its file extension.

    Parameters:
    - model_path (str): The file path of the machine learning model.

    Returns:
    - str: The type of the model, which can be one of the following:
      - "torchscript" if the model has a '.pt' extension (TorchScript).
      - "pytorch" if the model has a '.pth' extension (PyTorch).
      - "simplified_onnx" if the model file ends with 'simplified.onnx' (Simplified ONNX).
      - "onnx" if the model has a '.onnx' extension (ONNX).
      - "json" if the model has a '.json' extension (JSON).
      - None if model extension is not supported.

    Example:
    ```python
    model_path = "path/to/your/model.onnx"
    model_type = model_type_from_path(model_path)
    print(f"The model type is: {model_type}")
    ```
    """
    if model_path.endswith(".pt"):
        return "torchscript"
    if model_path.endswith(".pth"):
        return "pytorch"
    if model_path.endswith("simplified.onnx"):
        return "simplified_onnx"
    if model_path.endswith(".onnx"):
        return "onnx"
    if model_path.endswith(".json"):
        return "json"
    return None


def setup_opencv() -> None:
    """Setup OpenCV to use only one thread and not use OpenCL."""
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)


def get_device(cuda: bool = True) -> str:
    """Returns the device to use for training.

    Args:
        cuda: whether to use cuda or not

    Returns:
        The device to use
    """
    if torch.cuda.is_available() and cuda:
        return "cuda:0"

    return "cpu"


def nested_set(dic: dict, keys: list[str], value: str) -> None:
    """Assign the value of a dictionary using nested keys."""
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})

    dic[keys[-1]] = value


def flatten_list(input_list: Iterable[Any]) -> Iterator[Any]:
    """Return an iterator over the flattened list.

    Args:
        input_list: the list to be flattened

    Yields:
        The iterator over the flattend list
    """
    for v in input_list:
        if isinstance(v, Iterable) and not isinstance(v, str | bytes):
            yield from flatten_list(v)
        else:
            yield v


class HydraEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle OmegaConf objects."""

    def default(self, o):
        """Convert OmegaConf objects to base python objects."""
        if o is not None and OmegaConf.is_config(o):
            return OmegaConf.to_container(o)
        return json.JSONEncoder.default(self, o)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy objects."""

    def default(self, o):
        """Custom JSON encoder to handle numpy objects."""
        if o is not None:
            if isinstance(o, np.ndarray):
                if o.size == 1:
                    return o.item()
                return o.tolist()
            if isinstance(o, np.number):
                return o.item()
        return json.JSONEncoder.default(self, o)


class AllGatherSyncFunction(torch.autograd.Function):
    """Function to gather gradients from multiple GPUs."""

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_tensorboard_logger(trainer: pl.Trainer) -> TensorBoardLogger | None:
    """Safely get tensorboard logger from Lightning Trainer loggers.

    Args:
        trainer: Pytorch Lightning Trainer.

    Returns:
        An mlflow logger if available, else None.
    """
    if isinstance(trainer.logger, TensorBoardLogger):
        return trainer.logger

    if isinstance(trainer.logger, list):
        for logger in trainer.logger:
            if isinstance(logger, TensorBoardLogger):
                return logger

    return None
