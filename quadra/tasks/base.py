from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Generic, TypeVar

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from lightning_fabric.utilities.device_parser import _parse_gpu_ids
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import Logger, MLFlowLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from quadra import get_version
from quadra.callbacks.mlflow import validate_artifact_storage
from quadra.datamodules.base import BaseDataModule
from quadra.models.evaluation import BaseEvaluationModel
from quadra.utils import utils
from quadra.utils.export import import_deployment_model

log = utils.get_logger(__name__)
DataModuleT = TypeVar("DataModuleT", bound=BaseDataModule)


class Task(Generic[DataModuleT]):
    """Base Experiment Task.

    Args:
        config: The experiment configuration.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.export_folder: str = "deployment_model"
        self._datamodule: DataModuleT
        self.metadata: dict[str, Any]
        self.save_config()

    def save_config(self) -> None:
        """Save the experiment configuration when running an Hydra experiment."""
        if HydraConfig.initialized():
            with open("config_resolved.yaml", "w") as fp:
                OmegaConf.save(config=OmegaConf.to_container(self.config, resolve=True), f=fp.name)

    def prepare(self) -> None:
        """Prepare the experiment."""
        self.datamodule = self.config.datamodule

    @property
    def datamodule(self) -> DataModuleT:
        """T_DATAMODULE: The datamodule."""
        return self._datamodule

    @datamodule.setter
    def datamodule(self, datamodule_config: DictConfig) -> None:
        """DataModuleT: The datamodule. Instantiated from the datamodule config."""
        log.info("Instantiating datamodule <%s>", {datamodule_config["_target_"]})
        datamodule: DataModuleT = hydra.utils.instantiate(datamodule_config)
        self._datamodule = datamodule

    def train(self) -> Any:
        """Train the model."""
        log.info("Training not implemented for this task!")

    def test(self) -> Any:
        """Test the model."""
        log.info("Testing not implemented for this task!")

    def export(self) -> None:
        """Export model for production."""
        log.info("Export model for production not implemented for this task!")

    def generate_report(self) -> None:
        """Generate a report."""
        log.info("Report generation not implemented for this task!")

    def finalize(self) -> None:
        """Finalize the experiment."""
        log.info("Results are saved in %s", os.getcwd())

    def execute(self) -> None:
        """Execute the experiment and all the steps."""
        self.prepare()
        self.train()
        self.test()
        if self.config.export is not None and len(self.config.export.types) > 0:
            self.export()
        self.generate_report()
        self.finalize()


class LightningTask(Generic[DataModuleT], Task[DataModuleT]):
    """Base Experiment Task.

    Args:
        config: The experiment configuration
        checkpoint_path: The path to the checkpoint to load the model from. Defaults to None.
        run_test: Whether to run the test after training. Defaults to False.
        report: Whether to generate a report. Defaults to False.
    """

    def __init__(
        self,
        config: DictConfig,
        checkpoint_path: str | None = None,
        run_test: bool = False,
        report: bool = False,
    ):
        super().__init__(config=config)
        self.checkpoint_path = checkpoint_path
        self.run_test = run_test
        self.report = report
        self._module: LightningModule
        self._devices: int | list[int]
        self._callbacks: list[Callback]
        self._logger: list[Logger]
        self._trainer: Trainer

    def prepare(self) -> None:
        """Prepare the experiment."""
        super().prepare()

        # First setup loggers since some callbacks might need logger setup correctly.
        if "logger" in self.config:
            self.logger = self.config.logger

        if "callbacks" in self.config:
            self.callbacks = self.config.callbacks

        self.devices = self.config.trainer.devices
        self.trainer = self.config.trainer

    @property
    def module(self) -> LightningModule:
        """LightningModule: The model."""
        return self._module

    @module.setter
    def module(self, module_config) -> None:
        """LightningModule: The model."""
        raise NotImplementedError("module must be set in subclass")

    @property
    def trainer(self) -> Trainer:
        """Trainer: The trainer."""
        return self._trainer

    @trainer.setter
    def trainer(self, trainer_config: DictConfig) -> None:
        """Trainer: The trainer."""
        log.info("Instantiating trainer <%s>", trainer_config["_target_"])
        trainer_config.devices = self.devices
        trainer: Trainer = hydra.utils.instantiate(
            trainer_config,
            callbacks=self.callbacks,
            logger=self.logger,
            _convert_="partial",
        )
        self._trainer = trainer

    @property
    def callbacks(self) -> list[Callback]:
        """List[Callback]: The callbacks."""
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks_config) -> None:
        """List[Callback]: The callbacks."""
        if self.config.core.get("unit_test"):
            log.info("Unit Testing, skipping callbacks")
            return
        instatiated_callbacks = []
        for _, cb_conf in callbacks_config.items():
            if "_target_" in cb_conf:
                # Disable is a reserved keyword for callbacks, hopefully no callback will use it
                if "disable" in cb_conf:
                    if cb_conf["disable"]:
                        log.info("Skipping callback <%s> as it is disabled", cb_conf["_target_"])
                        continue

                    with open_dict(cb_conf):
                        del cb_conf.disable

                # Skip the gpu stats logger callback if no gpu is available to avoid errors
                if not torch.cuda.is_available() and cb_conf["_target_"] == "nvitop.callbacks.lightning.GpuStatsLogger":
                    continue

                log.info("Instantiating callback <%s>", cb_conf["_target_"])
                instatiated_callbacks.append(hydra.utils.instantiate(cb_conf))
        self._callbacks = instatiated_callbacks
        if len(instatiated_callbacks) <= 0:
            log.warning("No callback found in configuration.")

    @property
    def logger(self) -> list[Logger]:
        """List[Logger]: The loggers."""
        return self._logger

    @logger.setter
    def logger(self, logger_config) -> None:
        """List[Logger]: The loggers."""
        if self.config.core.get("unit_test"):
            log.info("Unit Testing, skipping loggers")
            return
        instantiated_loggers = []
        for _, lg_conf in logger_config.items():
            if "_target_" in lg_conf:
                log.info("Instantiating logger <%s>", lg_conf["_target_"])
                logger = hydra.utils.instantiate(lg_conf)
                if isinstance(logger, MLFlowLogger):
                    validate_artifact_storage(logger)
                instantiated_loggers.append(logger)

        self._logger = instantiated_loggers

        if len(instantiated_loggers) <= 0:
            log.warning("No logger found in configuration.")

    @property
    def devices(self) -> int | list[int]:
        """List[int]: The devices ids."""
        return self._devices

    @devices.setter
    def devices(self, devices) -> None:
        """List[int]: The devices ids."""
        if self.config.trainer.get("accelerator") == "cpu":
            self._devices = self.config.trainer.devices
            return

        try:
            self._devices = _parse_gpu_ids(devices, include_cuda=True)
        except MisconfigurationException:
            self._devices = 1
            self.config.trainer["accelerator"] = "cpu"
            log.warning("Trying to instantiate GPUs but no GPUs are available, training will be done on CPU")

    def train(self) -> None:
        """Train the model."""
        log.info("Starting training!")
        utils.log_hyperparameters(
            config=self.config,
            model=self.module,
            trainer=self.trainer,
        )

        self.trainer.fit(model=self.module, datamodule=self.datamodule)

    def test(self) -> Any:
        """Test the model."""
        log.info("Starting testing!")

        best_model = None
        if (
            self.trainer.checkpoint_callback is not None
            and hasattr(self.trainer.checkpoint_callback, "best_model_path")
            and self.trainer.checkpoint_callback.best_model_path is not None
            and len(self.trainer.checkpoint_callback.best_model_path) > 0
        ):
            best_model = self.trainer.checkpoint_callback.best_model_path

        if best_model is None:
            log.warning(
                "No best checkpoint model found, using last weights for test, this might lead to worse results, "
                "consider using a checkpoint callback."
            )

        return self.trainer.test(model=self.module, datamodule=self.datamodule, ckpt_path=best_model)

    def finalize(self) -> None:
        """Finalize the experiment."""
        super().finalize()
        utils.finish(
            config=self.config,
            module=self.module,
            datamodule=self.datamodule,
            trainer=self.trainer,
            callbacks=self.callbacks,
            logger=self.logger,
            export_folder=self.export_folder,
        )

        if (
            not self.config.trainer.get("fast_dev_run")
            and self.trainer.checkpoint_callback is not None
            and hasattr(self.trainer.checkpoint_callback, "best_model_path")
        ):
            log.info("Best model ckpt: %s", self.trainer.checkpoint_callback.best_model_path)

    def add_callback(self, callback: Callback):
        """Add a callback to the trainer.

        Args:
            callback: The callback to add
        """
        if hasattr(self.trainer, "callbacks") and isinstance(self.trainer.callbacks, list):
            self.trainer.callbacks.append(callback)

    def execute(self) -> None:
        """Execute the experiment and all the steps."""
        self.prepare()
        self.train()
        if self.run_test:
            self.test()
        if self.config.export is not None and len(self.config.export.types) > 0:
            self.export()
        if self.report:
            self.generate_report()
        self.finalize()


class PlaceholderTask(Task):
    """Placeholder task."""

    def execute(self) -> None:
        """Execute the task and all the steps."""
        log.info("Running Placeholder Task.")
        log.info("Quadra Version: %s", str(get_version()))
        log.info("If you are reading this, it means that library is installed correctly!")


class Evaluation(Generic[DataModuleT], Task[DataModuleT]):
    """Base Evaluation Task with deployment models.

    Args:
        config: The experiment configuration
        model_path: The model path.
        device: Device to use for evaluation. If None, the device is automatically determined.

    """

    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        device: str | None = None,
    ):
        super().__init__(config=config)

        if device is None:
            self.device = utils.get_device()
        else:
            self.device = device

        self.config = config
        self.model_data: dict[str, Any]
        self.model_path = model_path
        self._deployment_model: BaseEvaluationModel
        self.deployment_model_type: str
        self.model_info_filename = "model.json"
        self.report_path = ""
        self.metadata = {"report_files": []}

    @property
    def deployment_model(self) -> BaseEvaluationModel:
        """Deployment model."""
        return self._deployment_model

    @deployment_model.setter
    def deployment_model(self, model_path: str):
        """Set the deployment model."""
        self._deployment_model = import_deployment_model(
            model_path=model_path, device=self.device, inference_config=self.config.inference
        )

    def prepare(self) -> None:
        """Prepare the evaluation."""
        with open(os.path.join(Path(self.model_path).parent, self.model_info_filename)) as f:
            self.model_data = json.load(f)

        if not isinstance(self.model_data, dict):
            raise ValueError("Model info file is not a valid json")

        for input_size in self.model_data["input_size"]:
            if len(input_size) != 3:
                continue

            # Adjust the transform for 2D models (CxHxW)
            # We assume that each input size has the same height and width
            if input_size[1] != self.config.transforms.input_height:
                log.warning(
                    "Input height of the model (%s) is different from the one specified "
                    + "in the config (%s). Fixing the config.",
                    input_size[1],
                    self.config.transforms.input_height,
                )
                self.config.transforms.input_height = input_size[1]

            if input_size[2] != self.config.transforms.input_width:
                log.warning(
                    "Input width of the model (%s) is different from the one specified "
                    + "in the config (%s). Fixing the config.",
                    input_size[2],
                    self.config.transforms.input_width,
                )
                self.config.transforms.input_width = input_size[2]

        self.deployment_model = self.model_path  # type: ignore[assignment]
