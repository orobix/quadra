import json
import os
import typing
from typing import Any, Dict, Generic, List, Optional

import cv2
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from quadra.callbacks.mlflow import get_mlflow_logger
from quadra.datamodules import SegmentationDataModule, SegmentationMulticlassDataModule
from quadra.models.base import ModelWrapper
from quadra.modules.base import SegmentationModel
from quadra.tasks.base import Evaluation, LightningTask
from quadra.utils import utils
from quadra.utils.evaluation import create_mask_report
from quadra.utils.export import export_torchscript_model

log = utils.get_logger(__name__)

SegmentationDataModuleT = typing.TypeVar(
    "SegmentationDataModuleT", SegmentationDataModule, SegmentationMulticlassDataModule
)


class Segmentation(Generic[SegmentationDataModuleT], LightningTask[SegmentationDataModuleT]):
    """Task for segmentation.

    Args:
        config: Config object
        num_viz_samples: Number of samples to visualize. Defaults to 5.
        checkpoint_path: Path to the checkpoint to load the model from. Defaults to None.
        run_test: If True, run test after training. Defaults to False.
        evaluate: Dict with evaluation parameters. Defaults to None.
        report: If True, create report after training. Defaults to False.
        export_type: List of export method for the model, e.g. [torchscript]. Defaults to None.
    """

    def __init__(
        self,
        config: DictConfig,
        num_viz_samples: int = 5,
        checkpoint_path: Optional[str] = None,
        run_test: bool = False,
        evaluate: Optional[DictConfig] = None,
        report: bool = False,
        export_type: Optional[List[str]] = None,
    ):
        super().__init__(
            config=config,
            checkpoint_path=checkpoint_path,
            run_test=run_test,
            report=report,
            export_type=export_type,
        )
        self.evaluate = evaluate
        self.num_viz_samples = num_viz_samples
        self.export_folder: str = "deployment_model"
        self.exported_model_path: Optional[str] = None
        if self.evaluate and any(self.evaluate.values()):
            if self.export_type is None or len(self.export_type) == 0 or "torchscript" not in self.export_type:
                log.info(
                    "Evaluation is enabled, but training does not export a deployment model. Automatically export the "
                    "model as torchscript."
                )
                if self.export_type is None:
                    self.export_type = ["torchscript"]
                else:
                    self.export_type.append("torchscript")
            if not self.report:
                log.info("Evaluation is enabled, but reporting is disabled. Enabling reporting automatically.")
                self.report = True

    @property
    def module(self) -> SegmentationModel:
        """Get the module."""
        return self._module

    @module.setter
    def module(self, module_config) -> None:
        """Set the module."""
        log.info("Instantiating model <%s>", module_config.model["_target_"])

        if isinstance(self.datamodule, SegmentationMulticlassDataModule):
            if module_config.model.num_classes != (len(self.datamodule.idx_to_class) + 1):
                log.warning(
                    f"Number of classes in the model ({module_config.model.num_classes}) does not match the number of "
                    + f"classes in the datamodule ({len(self.datamodule.idx_to_class)}). Updating the model..."
                )
                module_config.model.num_classes = len(self.datamodule.idx_to_class) + 1

        model = hydra.utils.instantiate(module_config.model)
        model = ModelWrapper(model)
        log.info("Instantiating optimizer <%s>", self.config.optimizer["_target_"])
        param_list = []
        for param in model.parameters():
            if param.requires_grad:
                param_list.append(param)
        optimizer = hydra.utils.instantiate(self.config.optimizer, param_list)
        log.info("Instantiating scheduler <%s>", self.config.scheduler["_target_"])
        scheduler = hydra.utils.instantiate(self.config.scheduler, optimizer=optimizer)
        log.info("Instantiating module <%s>", module_config.module["_target_"])
        module = hydra.utils.instantiate(module_config.module, model=model, optimizer=optimizer, lr_scheduler=scheduler)
        if self.checkpoint_path is not None:
            module.load_from_checkpoint(self.checkpoint_path, model=model, optimizer=optimizer, lr_scheduler=scheduler)
        self._module = module

    def prepare(self) -> None:
        """Prepare the task."""
        super().prepare()
        self.module = self.config.model

    def export(self) -> None:
        """Generate a deployment model for the task."""
        log.info("Exporting model ready for deployment")
        self.config.transforms.get("input_width")
        self.config.transforms.get("input_height")

        # Get best model!
        if self.trainer.checkpoint_callback is None:
            raise ValueError("No checkpoint callback found in the trainer")
        best_model_path = self.trainer.checkpoint_callback.best_model_path  # type: ignore[attr-defined]
        log.info("Loaded best model from %s", best_model_path)

        module = self.module.load_from_checkpoint(
            best_model_path,
            model=self.module.model,
            loss_fun=None,
            optimizer=self.module.optimizer,
            lr_scheduler=self.module.schedulers,
        )

        if "idx_to_class" not in self.config.datamodule:
            log.info("No idx_to_class key")
            classes = {0: "good", 1: "bad"}
        else:
            log.info("idx_to_class is present")
            classes = self.config.datamodule.idx_to_class

        if self.export_type is None:
            raise ValueError(
                "No export type specified. This should not happen, please check if you have set "
                "the export_type or assign it to a default value."
            )

        # TODO: Take it from the config
        input_shape = None

        half_precision = self.trainer.precision == 16

        for export_type in self.export_type:
            if export_type == "torchscript":
                out = export_torchscript_model(
                    model=module.model,
                    input_shapes=input_shape,
                    output_path=self.export_folder,
                    half_precision=half_precision,
                )

                if out is None:
                    log.warning("Skipping torchscript export since the model is not supported")
                    continue

                self.exported_model_path, input_shape = out

        if input_shape is None:
            log.warning("Not able to export the model in any format")

        model_json = {
            "input_size": input_shape,
            "classes": classes,
            "mean": self.config.transforms.mean,
            "std": self.config.transforms.std,
        }

        with open(os.path.join(self.export_folder, "model.json"), "w") as f:
            json.dump(model_json, f, cls=utils.HydraEncoder)

    def generate_report(self) -> None:
        """Generate a report for the task."""
        if self.evaluate is not None:
            log.info("Generating evaluation report!")
            eval_tasks: List[SegmentationEvaluation] = []
            if self.evaluate.analysis:
                if self.exported_model_path is None:
                    raise ValueError(
                        "Exported model path is not set yet but the task tries to do an analysis evaluation"
                    )
                eval_task = SegmentationAnalysisEvaluation(
                    config=self.config,
                    model_path=self.exported_model_path,
                )
                eval_tasks.append(eval_task)
            for task in eval_tasks:
                task.execute()

            if len(self.logger) > 0:
                mflow_logger = get_mlflow_logger(trainer=self.trainer)
                tensorboard_logger = utils.get_tensorboard_logger(trainer=self.trainer)

                if mflow_logger is not None and self.config.core.get("upload_artifacts"):
                    for task in eval_tasks:
                        for file in task.metadata["report_files"]:
                            mflow_logger.experiment.log_artifact(
                                run_id=mflow_logger.run_id, local_path=file, artifact_path=task.report_folder
                            )

                if tensorboard_logger is not None and self.config.core.get("upload_artifacts"):
                    for task in eval_tasks:
                        for file in task.metadata["report_files"]:
                            ext = os.path.splitext(file)[1].lower()

                            if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"]:
                                try:
                                    img = cv2.imread(file)
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                except cv2.error:
                                    log.info("Could not upload artifact image %s", file)
                                    continue

                                tensorboard_logger.experiment.add_image(
                                    os.path.basename(file), img, 0, dataformats="HWC"
                                )
                            else:
                                utils.upload_file_tensorboard(file, tensorboard_logger)


class SegmentationEvaluation(Evaluation):
    """Segmentation Evaluation Task with deployment models.

    Args:
        config: The experiment configuration
        model_path: The experiment path.
        report_folder: The report folder. Defaults to None.

    Raises:
        ValueError: If the model path is not provided
    """

    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        report_folder: Optional[str] = None,
    ):
        super().__init__(config=config, model_path=model_path, report_folder=report_folder)
        self.config = config
        self.metadata = {"report_files": []}

        # TODO: It's not possible to specify the device from outside!!
        self.device = utils.get_device(config.trainer.accelerator != "cpu")

    def save_config(self) -> None:
        """Skip saving the config."""

    @torch.no_grad()
    def inference(
        self, dataloader: DataLoader, deployment_model: torch.nn.Module, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Run inference on the dataloader and return the output.

        Args:
            dataloader: The dataloader to run inference on
            deployment_model: The deployment model to use
            device: The device to run inference on
        """
        image_list, mask_list, mask_pred_list, label_list = [], [], [], []
        for batch in dataloader:
            images, masks, labels = batch
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            image_list.append(images.cpu())
            mask_list.append(masks.cpu())
            mask_pred_list.append(deployment_model(images.to(device)).cpu())
            label_list.append(labels.cpu())
        output = {
            "image": torch.cat(image_list, dim=0),
            "mask": torch.cat(mask_list, dim=0),
            "label": torch.cat(label_list, dim=0),
            "mask_pred": torch.cat(mask_pred_list, dim=0),
        }
        return output


class SegmentationAnalysisEvaluation(SegmentationEvaluation):
    """Segmentation Analysis Evaluation Task
    Args:
        config: The experiment configuration
        model_path: The model path.
        report_folder: The report folder. Defaults to "analysis_report".
    """

    def __init__(self, config: DictConfig, model_path: str, report_folder: str = "analysis_report"):
        super().__init__(config=config, model_path=model_path, report_folder=report_folder)
        self.test_output: Dict[str, Any] = {}

    def prepare(self) -> None:
        """Prepare the experiment."""
        super().prepare()
        self.datamodule = self.config.datamodule

    def train(self) -> None:
        """Skip training."""

    def test(self) -> None:
        """Run testing."""
        log.info("Starting testing")

        stages: List[str] = []
        dataloaders: List[torch.utils.data.DataLoader] = []
        self.datamodule.setup(stage="fit")
        self.datamodule.setup(stage="test")
        if self.datamodule.train_dataset_available:
            stages.append("train")
            dataloaders.append(self.datamodule.train_dataloader())
            if self.datamodule.val_dataset_available:
                stages.append("val")
                dataloaders.append(self.datamodule.val_dataloader())
        if self.datamodule.test_dataset_available:
            stages.append("test")
            dataloaders.append(self.datamodule.test_dataloader())
        for stage, dataloader in zip(stages, dataloaders):
            image_list, mask_list, mask_pred_list, label_list = [], [], [], []
            for batch in dataloader:
                images, masks, labels = batch
                images = images.to(self.device)
                if len(masks.shape) == 3:  # BxHxW -> Bx1xHxW
                    masks = masks.unsqueeze(1)
                with torch.no_grad():
                    image_list.append(images.cpu())
                    mask_list.append(masks.cpu())
                    mask_pred_list.append(self.deployment_model(images.to(self.device)).cpu())
                    label_list.append(labels.cpu())

            output = {
                "image": torch.cat(image_list, dim=0),
                "mask": torch.cat(mask_list, dim=0),
                "label": torch.cat(label_list, dim=0),
                "mask_pred": torch.cat(mask_pred_list, dim=0),
            }
            self.test_output[stage] = output

    def generate_report(self) -> None:
        """Generate a report."""
        log.info("Generating analysis report")

        for stage, output in self.test_output.items():
            image_mean = OmegaConf.to_container(self.config.transforms.mean)
            if not isinstance(image_mean, list) or any(not isinstance(x, (int, float)) for x in image_mean):
                raise ValueError("Image mean is not a list of float or integer values, please check your config")
            image_std = OmegaConf.to_container(self.config.transforms.std)
            if not isinstance(image_std, list) or any(not isinstance(x, (int, float)) for x in image_std):
                raise ValueError("Image std is not a list of float or integer values, please check your config")
            reports = create_mask_report(
                stage=stage,
                output=output,
                report_path="analysis_report",
                mean=image_mean,
                std=image_std,
                analysis=True,
                nb_samples=10,
                apply_sigmoid=True,
                show_orj_predictions=True,
            )
            self.metadata["report_files"].extend(reports)
            log.info("%s analysis report completed.", stage)
