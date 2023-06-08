import os
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import Logger, MLFlowLogger
from pytorch_lightning.utilities.device_parser import parse_gpu_ids
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.jit._script import RecursiveScriptModule
from torch.nn import Module

from quadra import get_version
from quadra.callbacks.mlflow import get_mlflow_logger
from quadra.datamodules.base import BaseDataModule
from quadra.utils import utils
from quadra.utils.export import import_deployment_model

try:
    import mlflow  # noqa

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

log = utils.get_logger(__name__)
DataModuleT = TypeVar("DataModuleT", bound=BaseDataModule)


class Task(Generic[DataModuleT]):
    """Base Experiment Task.

    Args:
        config: The experiment configuration.
        export_type: List of export method for the model, e.g. [torchscript]. Defaults to None.
    """

    def __init__(self, config: DictConfig, export_type: Optional[List[str]] = None):
        self.config = config
        self.export_type = export_type
        self.export_folder: str = "deployment_model"
        self._datamodule: DataModuleT
        self.metadata: Dict[str, Any]
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
        if self.export_type is not None and len(self.export_type) > 0:
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
        export_type: List of export method for the model, e.g. [torchscript]. Defaults to None.
    """

    def __init__(
        self,
        config: DictConfig,
        checkpoint_path: Optional[str] = None,
        run_test: bool = False,
        report: bool = False,
        export_type: Optional[List[str]] = None,
    ):
        super().__init__(config, export_type=export_type)
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.run_test = run_test
        self.report = report
        self._module: LightningModule
        self._devices: Union[int, List[int]]
        self._callbacks: List[Callback]
        self._logger: List[Logger]
        self._trainer: Trainer

    def prepare(self) -> None:
        """Prepare the experiment."""
        super().prepare()

        if "callbacks" in self.config:
            self.callbacks = self.config.callbacks

        if "logger" in self.config:
            self.logger = self.config.logger

            for logger in self.logger:
                if (
                    isinstance(logger, MLFlowLogger)
                    and MLFLOW_AVAILABLE
                    and os.environ.get("MLFLOW_TRACKING_URI") is not None
                ):
                    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
                    mlflow.pytorch.autolog()

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
    def callbacks(self) -> List[Callback]:
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
                if not torch.cuda.is_available():
                    # Skip the gpu stats logger callback if no gpu is available to avoid errors
                    if cb_conf["_target_"] == "nvitop.callbacks.lightning.GpuStatsLogger":
                        continue

                log.info("Instantiating callback <%s>", cb_conf["_target_"])
                instatiated_callbacks.append(hydra.utils.instantiate(cb_conf))
        self._callbacks = instatiated_callbacks
        if len(instatiated_callbacks) <= 0:
            log.warning("No callback found in configuration.")

    @property
    def logger(self) -> List[Logger]:
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
                instantiated_loggers.append(logger)

        self._logger = instantiated_loggers

        if len(instantiated_loggers) <= 0:
            log.warning("No logger found in configuration.")

    @property
    def devices(self) -> Union[int, List[int]]:
        """List[int]: The devices ids."""
        return self._devices

    @devices.setter
    def devices(self, devices) -> None:
        """List[int]: The devices ids."""
        if self.config.trainer.get("accelerator") == "cpu":
            self._devices = self.config.trainer.devices
            return

        try:
            self._devices = parse_gpu_ids(devices, include_cuda=True)
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

        mlflow_logger = get_mlflow_logger(self.trainer)

        if mlflow_logger is not None:
            with mlflow.start_run(run_id=mlflow_logger.run_id) as _:
                self.trainer.fit(model=self.module, datamodule=self.datamodule)
        else:
            self.trainer.fit(model=self.module, datamodule=self.datamodule)

    def test(self) -> Any:
        """Test the model."""
        log.info("Starting testing!")
        return self.trainer.test(model=self.module, datamodule=self.datamodule)

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

        if not self.config.trainer.get("fast_dev_run"):
            if self.trainer.checkpoint_callback is not None and hasattr(
                self.trainer.checkpoint_callback, "best_model_path"
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
        if self.export_type is not None and len(self.export_type) > 0:
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


class Evaluation(Task):
    """Base Evaluation Task with deployment models.

    Args:
        config: The experiment configuration
        model_path: The model path.
        report_folder: The report folder. Defaults to None.

    Raises:
        ValueError: If the experiment path is not provided
    """

    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        report_folder: Optional[str] = None,
    ):
        super().__init__(config=config)
        self.config = config
        self.metadata = {"report_files": []}
        self.model_path = model_path
        self.device = utils.get_device()
        self.report_folder = report_folder
        self._deployment_model: Union[RecursiveScriptModule, Module]
        self.deployment_model_type: str
        if self.report_folder is None:
            log.warning("Report folder is not provided, using default report folder")
            self.report_folder = "report"

    @property
    def deployment_model(self) -> Union[RecursiveScriptModule, Module]:
        """RecursiveScriptModule: The deployment model."""
        return self._deployment_model

    @deployment_model.setter
    def deployment_model(self, model: Union[RecursiveScriptModule, Module]) -> None:
        """RecursiveScriptModule: The deployment model."""
        self._deployment_model = model

    def prepare(self) -> None:
        """Prepare the evaluation."""
        self.deployment_model, self.deployment_model_type = import_deployment_model(self.model_path, self.device)
