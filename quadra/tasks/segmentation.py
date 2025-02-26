from __future__ import annotations

import json
import os
import typing
from typing import Any, Generic

import cv2
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from quadra.callbacks.mlflow import get_mlflow_logger
from quadra.datamodules import SegmentationDataModule, SegmentationMulticlassDataModule
from quadra.models.base import ModelSignatureWrapper
from quadra.models.evaluation import BaseEvaluationModel
from quadra.modules.base import SegmentationModel
from quadra.tasks.base import Evaluation, LightningTask
from quadra.utils import utils
from quadra.utils.evaluation import automatic_datamodule_batch_size, create_mask_report
from quadra.utils.export import export_model

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
    """

    def __init__(
        self,
        config: DictConfig,
        num_viz_samples: int = 5,
        checkpoint_path: str | None = None,
        run_test: bool = False,
        evaluate: DictConfig | None = None,
        report: bool = False,
    ):
        super().__init__(
            config=config,
            checkpoint_path=checkpoint_path,
            run_test=run_test,
            report=report,
        )
        self.evaluate = evaluate
        self.num_viz_samples = num_viz_samples
        self.export_folder: str = "deployment_model"
        self.exported_model_path: str | None = None
        if self.evaluate and any(self.evaluate.values()):
            if (
                self.config.export is None
                or len(self.config.export.types) == 0
                or "torchscript" not in self.config.export.types
            ):
                log.info(
                    "Evaluation is enabled, but training does not export a deployment model. Automatically export the "
                    "model as torchscript."
                )
                if self.config.export is None:
                    self.config.export = DictConfig({"types": ["torchscript"]})
                else:
                    self.config.export.types.append("torchscript")

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

        if isinstance(self.datamodule, SegmentationMulticlassDataModule) and module_config.model.num_classes != (
            len(self.datamodule.idx_to_class) + 1
        ):
            log.warning(
                "Number of classes in the model (%s) does not match the number of "
                + "classes in the datamodule (%d). Updating the model...",
                module_config.model.num_classes,
                len(self.datamodule.idx_to_class),
            )
            module_config.model.num_classes = len(self.datamodule.idx_to_class) + 1

        model = hydra.utils.instantiate(module_config.model)
        model = ModelSignatureWrapper(model)
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
            module.__class__.load_from_checkpoint(
                self.checkpoint_path, model=model, optimizer=optimizer, lr_scheduler=scheduler
            )
        self._module = module

    def prepare(self) -> None:
        """Prepare the task."""
        super().prepare()
        self.module = self.config.model

    def export(self) -> None:
        """Generate a deployment model for the task."""
        log.info("Exporting model ready for deployment")

        # Get best model!
        if (
            self.trainer.checkpoint_callback is not None
            and hasattr(self.trainer.checkpoint_callback, "best_model_path")
            and self.trainer.checkpoint_callback.best_model_path is not None
            and len(self.trainer.checkpoint_callback.best_model_path) > 0
        ):
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            log.info("Loaded best model from %s", best_model_path)

            module = self.module.__class__.load_from_checkpoint(
                best_model_path,
                model=self.module.model,
                loss_fun=None,
                optimizer=self.module.optimizer,
                lr_scheduler=self.module.schedulers,
            )
        else:
            log.warning("No checkpoint callback found in the trainer, exporting the last model weights")
            module = self.module

        if "idx_to_class" not in self.config.datamodule:
            log.info("No idx_to_class key")
            idx_to_class = {0: "good", 1: "bad"}  # TODO: Why is this the default value?
        else:
            log.info("idx_to_class is present")
            idx_to_class = self.config.datamodule.idx_to_class

        if self.config.export is None:
            raise ValueError(
                "No export type specified. This should not happen, please check if you have set "
                "the export_type or assign it to a default value."
            )

        half_precision = "16" in self.trainer.precision

        input_shapes = self.config.export.input_shapes

        model_json, export_paths = export_model(
            config=self.config,
            model=module.model,
            export_folder=self.export_folder,
            half_precision=half_precision,
            input_shapes=input_shapes,
            idx_to_class=idx_to_class,
        )

        if len(export_paths) == 0:
            return

        # Pick one model for evaluation, it should be independent of the export type as the model is wrapped
        self.exported_model_path = next(iter(export_paths.values()))

        with open(os.path.join(self.export_folder, "model.json"), "w") as f:
            json.dump(model_json, f, cls=utils.HydraEncoder)

    def generate_report(self) -> None:
        """Generate a report for the task."""
        if self.evaluate is not None:
            log.info("Generating evaluation report!")
            eval_tasks: list[SegmentationEvaluation] = []
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
                                run_id=mflow_logger.run_id, local_path=file, artifact_path=task.report_path
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


class SegmentationEvaluation(Evaluation[SegmentationDataModuleT]):
    """Segmentation Evaluation Task with deployment models.

    Args:
        config: The experiment configuration
        model_path: The experiment path.
        device: Device to use for evaluation. If None, the device is automatically determined.

    Raises:
        ValueError: If the model path is not provided
    """

    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        device: str | None = "cpu",
    ):
        super().__init__(config=config, model_path=model_path, device=device)
        self.config = config

    def save_config(self) -> None:
        """Skip saving the config."""

    def prepare(self) -> None:
        """Prepare the evaluation."""
        super().prepare()
        # TODO: Why we propagate mean and std only in Segmentation?
        self.config.transforms.mean = self.model_data["mean"]
        self.config.transforms.std = self.model_data["std"]
        # Setup datamodule
        if hasattr(self.config.datamodule, "idx_to_class"):
            idx_to_class = self.model_data["classes"]  # dict {index: class}
            self.config.datamodule.idx_to_class = idx_to_class
        self.datamodule = self.config.datamodule
        # prepare_data() must be explicitly called because there is no lightning training
        self.datamodule.prepare_data()

    @torch.no_grad()
    def inference(
        self, dataloader: DataLoader, deployment_model: BaseEvaluationModel, device: torch.device
    ) -> dict[str, torch.Tensor]:
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
        device: Device to use for evaluation. If None, the device is automatically determined.
    """

    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        device: str | None = None,
    ):
        super().__init__(config=config, model_path=model_path, device=device)
        self.test_output: dict[str, Any] = {}

    def train(self) -> None:
        """Skip training."""

    def prepare(self) -> None:
        """Prepare the evaluation task."""
        super().prepare()
        self.datamodule.setup(stage="fit")
        self.datamodule.setup(stage="test")

    @automatic_datamodule_batch_size(batch_size_attribute_name="batch_size")
    def test(self) -> None:
        """Run testing."""
        log.info("Starting inference for analysis.")

        stages: list[str] = []
        dataloaders: list[torch.utils.data.DataLoader] = []

        # if self.datamodule.train_dataset_available:
        #     stages.append("train")
        #     dataloaders.append(self.datamodule.train_dataloader())
        #     if self.datamodule.val_dataset_available:
        #         stages.append("val")
        #         dataloaders.append(self.datamodule.val_dataloader())

        if self.datamodule.test_dataset_available:
            stages.append("test")
            dataloaders.append(self.datamodule.test_dataloader())
        for stage, dataloader in zip(stages, dataloaders, strict=False):
            log.info("Running inference on %s set with batch size: %d", stage, dataloader.batch_size)
            image_list, mask_list, mask_pred_list, label_list = [], [], [], []
            for batch in dataloader:
                images, masks, labels = batch
                images = images.to(device=self.device, dtype=self.deployment_model.model_dtype)
                if len(masks.shape) == 3:  # BxHxW -> Bx1xHxW
                    masks = masks.unsqueeze(1)
                with torch.no_grad():
                    image_list.append(images)
                    mask_list.append(masks)
                    mask_pred_list.append(self.deployment_model(images.to(self.device)))
                    label_list.append(labels)

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
            if not isinstance(image_mean, list) or any(not isinstance(x, int | float) for x in image_mean):
                raise ValueError("Image mean is not a list of float or integer values, please check your config")
            image_std = OmegaConf.to_container(self.config.transforms.std)
            if not isinstance(image_std, list) or any(not isinstance(x, int | float) for x in image_std):
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
