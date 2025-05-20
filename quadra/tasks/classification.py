from __future__ import annotations

import glob
import json
import os
import typing
from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, cast

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from joblib import dump, load
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_grad_cam import GradCAM
from scipy import ndimage
from sklearn.base import ClassifierMixin
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torchinfo import summary
from tqdm import tqdm

from quadra.callbacks.mlflow import get_mlflow_logger
from quadra.callbacks.scheduler import WarmupInit
from quadra.datamodules import (
    ClassificationDataModule,
    MultilabelClassificationDataModule,
    SklearnClassificationDataModule,
)
from quadra.datasets.classification import ImageClassificationListDataset
from quadra.models.base import ModelSignatureWrapper
from quadra.models.classification import BaseNetworkBuilder
from quadra.models.evaluation import BaseEvaluationModel, TorchEvaluationModel, TorchscriptEvaluationModel
from quadra.modules.classification import ClassificationModule
from quadra.tasks.base import Evaluation, LightningTask, Task
from quadra.trainers.classification import SklearnClassificationTrainer
from quadra.utils import utils
from quadra.utils.classification import (
    get_results,
    save_classification_result,
)
from quadra.utils.evaluation import automatic_datamodule_batch_size
from quadra.utils.export import export_model, import_deployment_model
from quadra.utils.models import get_feature, is_vision_transformer
from quadra.utils.vit_explainability import VitAttentionGradRollout

log = utils.get_logger(__name__)

SklearnClassificationDataModuleT = typing.TypeVar(
    "SklearnClassificationDataModuleT", bound=SklearnClassificationDataModule
)
ClassificationDataModuleT = typing.TypeVar("ClassificationDataModuleT", bound=ClassificationDataModule)


# TODO: Maybe we should have a BaseClassificationTask that is extended by Classification and MultilabelClassification
# at the current time, multilabel experiments use this Classification task class and they can not generate report
# (it is written specifically for a vanilla classification). Moreover, this class takes generic
# ClassificationDataModuleT, but multilabel experim. uses MultilabelClassificationDataModule, which is not a child of
# ClassificationDataModule
class Classification(Generic[ClassificationDataModuleT], LightningTask[ClassificationDataModuleT]):
    """Classification Task.

    Args:
        config: The experiment configuration
        output: The otuput configuration.
        gradcam: Whether to compute gradcams
        checkpoint_path: The path to the checkpoint to load the model from. Defaults to None.
        lr_multiplier: The multiplier for the backbone learning rate. Defaults to None.
        output: The ouput configuration (under task config). It contains the bool "example" to generate
            figs of discordant/concordant predictions.
        report: Whether to generate a report containing the results after test phase
        run_test: Whether to run the test phase.
    """

    def __init__(
        self,
        config: DictConfig,
        output: DictConfig,
        checkpoint_path: str | None = None,
        lr_multiplier: float | None = None,
        gradcam: bool = False,
        report: bool = False,
        run_test: bool = False,
    ):
        super().__init__(
            config=config,
            checkpoint_path=checkpoint_path,
            run_test=run_test,
            report=report,
        )
        self.output = output
        self.gradcam = gradcam
        self._lr_multiplier = lr_multiplier
        self._pre_classifier: nn.Module
        self._classifier: nn.Module
        self._model: nn.Module
        self._optimizer: torch.optim.Optimizer
        self._scheduler: torch.optim.lr_scheduler._LRScheduler
        self.model_json: dict[str, Any] | None = None
        self.export_folder: str = "deployment_model"
        self.deploy_info_file: str = "model.json"
        self.report_confmat: pd.DataFrame
        self.best_model_path: str | None = None

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_config: DictConfig) -> None:
        """Set the optimizer."""
        if (
            isinstance(self.model.features_extractor, nn.Module)
            and isinstance(self.model.pre_classifier, nn.Module)
            and isinstance(self.model.classifier, nn.Module)
        ):
            log.info("Instantiating optimizer <%s>", self.config.optimizer["_target_"])
            if self._lr_multiplier is not None and self._lr_multiplier > 0:
                params = [
                    {
                        "params": self.model.features_extractor.parameters(),
                        "lr": optimizer_config.lr * self._lr_multiplier,
                    }
                ]
            else:
                params = [{"params": self.model.features_extractor.parameters(), "lr": optimizer_config.lr}]
            params.append({"params": self.model.pre_classifier.parameters(), "lr": optimizer_config.lr})
            params.append({"params": self.model.classifier.parameters(), "lr": optimizer_config.lr})
            self._optimizer = hydra.utils.instantiate(optimizer_config, params)

    @property
    def len_train_dataloader(self) -> int:
        """Get the length of the train dataloader."""
        len_train_dataloader = len(self.datamodule.train_dataloader())
        if self.devices is not None:
            num_gpus = len(self.devices) if isinstance(self.devices, list) else 1
            len_train_dataloader = len_train_dataloader // num_gpus
            if not self.datamodule.train_dataloader().drop_last:
                len_train_dataloader += int(len(self.datamodule.train_dataloader()) % num_gpus != 0)
        return len_train_dataloader

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Get the scheduler."""
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler_config: DictConfig) -> None:
        log.info("Instantiating scheduler <%s>", scheduler_config["_target_"])
        if "CosineAnnealingWithLinearWarmUp" in self.config.scheduler["_target_"]:
            # This scheduler will be overwritten by the SSLCallback
            self._scheduler = hydra.utils.instantiate(
                scheduler_config,
                optimizer=self.optimizer,
                batch_size=1,
                len_loader=1,
            )
            self.add_callback(WarmupInit(scheduler_config=scheduler_config))
        else:
            self._scheduler = hydra.utils.instantiate(scheduler_config, optimizer=self.optimizer)

    @property
    def module(self) -> ClassificationModule:
        """Get the module of the model."""
        return self._module

    @LightningTask.module.setter
    def module(self, module_config):  # noqa: F811
        """Set the module of the model."""
        module = hydra.utils.instantiate(
            module_config,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            gradcam=self.gradcam,
        )
        if self.checkpoint_path is not None:
            log.info("Loading model from lightning checkpoint: %s", self.checkpoint_path)
            module = module.__class__.load_from_checkpoint(
                self.checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
                criterion=module.criterion,
                gradcam=self.gradcam,
            )
        self._module = module

    @property
    def pre_classifier(self) -> nn.Module:
        return self._pre_classifier

    @pre_classifier.setter
    def pre_classifier(self, model_config: DictConfig) -> None:
        if "pre_classifier" in model_config and model_config.pre_classifier is not None:
            log.info("Instantiating pre_classifier <%s>", model_config.pre_classifier["_target_"])
            self._pre_classifier = hydra.utils.instantiate(model_config.pre_classifier, _convert_="partial")
        else:
            log.info("No pre-classifier found in config: instantiate a torch.nn.Identity instead")
            self._pre_classifier = nn.Identity()

    @property
    def classifier(self) -> nn.Module:
        return self._classifier

    @classifier.setter
    def classifier(self, model_config: DictConfig) -> None:
        if "classifier" in model_config:
            log.info("Instantiating classifier <%s>", model_config.classifier["_target_"])
            if self.datamodule.num_classes is None or self.datamodule.num_classes < 2:
                raise ValueError(f"Non compliant datamodule.num_classes : {self.datamodule.num_classes}")
            self._classifier = hydra.utils.instantiate(
                model_config.classifier, out_features=self.datamodule.num_classes, _convert_="partial"
            )
        else:
            raise ValueError("A `classifier` definition must be specified in the config")

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, model_config: DictConfig) -> None:
        self.pre_classifier = model_config  # type: ignore[assignment]
        self.classifier = model_config  # type: ignore[assignment]
        log.info("Instantiating backbone <%s>", model_config.model["_target_"])
        self._model = hydra.utils.instantiate(
            model_config.model, classifier=self.classifier, pre_classifier=self.pre_classifier, _convert_="partial"
        )
        if getattr(self.config.backbone, "freeze_parameters_name", None) is not None:
            self.freeze_layers_by_name(self.config.backbone.freeze_parameters_name)

        if getattr(self.config.backbone, "freeze_parameters_index", None) is not None:
            frozen_parameters_indices: list[int]
            if isinstance(self.config.backbone.freeze_parameters_index, int):
                # Freeze all layers up to the specified index
                frozen_parameters_indices = list(range(self.config.backbone.freeze_parameters_index + 1))
            elif isinstance(self.config.backbone.freeze_parameters_index, ListConfig):
                frozen_parameters_indices = cast(
                    list[int], OmegaConf.to_container(self.config.backbone.freeze_parameters_index, resolve=True)
                )
            else:
                raise ValueError("freeze_parameters_index must be an int or a list of int")

            self.freeze_parameters_by_index(frozen_parameters_indices)

    def prepare(self) -> None:
        """Prepare the experiment."""
        super().prepare()
        self.model = self.config.model
        self.optimizer = self.config.optimizer
        self.scheduler = self.config.scheduler
        self.module = self.config.model.module

    def train(self):
        """Train the model."""
        super().train()
        if (
            self.trainer.checkpoint_callback is not None
            and hasattr(self.trainer.checkpoint_callback, "best_model_path")
            and self.trainer.checkpoint_callback.best_model_path is not None
            and len(self.trainer.checkpoint_callback.best_model_path) > 0
        ):
            self.best_model_path = self.trainer.checkpoint_callback.best_model_path
            log.info("Loading best epoch weights...")

    def test(self) -> None:
        """Test the model."""
        if not self.config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            self.trainer.test(datamodule=self.datamodule, model=self.module, ckpt_path=self.best_model_path)

    def export(self) -> None:
        """Generate deployment models for the task."""
        if self.datamodule.class_to_idx is None:
            log.warning(
                "No `class_to_idx` found in the datamodule, class information will not be saved in the model.json"
            )
            idx_to_class = {}
        else:
            idx_to_class = {v: k for k, v in self.datamodule.class_to_idx.items()}

        # Get best model!
        if self.best_model_path is not None:
            log.info("Saving deployment model for %s checkpoint", self.best_model_path)

            module = self.module.__class__.load_from_checkpoint(
                self.best_model_path,
                model=self.module.model,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
                criterion=self.module.criterion,
                gradcam=False,
            )
        else:
            log.warning("No checkpoint callback found in the trainer, exporting the last model weights")
            module = self.module

        input_shapes = self.config.export.input_shapes

        # TODO: What happens if we have 64 precision?
        half_precision = "16" in self.trainer.precision

        example_input: torch.Tensor | None = None

        if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "val_dataset"):
            # Retrieve a better input to evaluate fp16 performance or efficientnetb0 does not sometimes export properly
            example_input = self.trainer.datamodule.val_dataset[0][0]

        # Selected rtol and atol are quite high, this is mostly done for efficientnetb0 that seems to be
        # quite unstable in fp16
        self.model_json, export_paths = export_model(
            config=self.config,
            model=module.model,
            export_folder=self.export_folder,
            half_precision=half_precision,
            input_shapes=input_shapes,
            idx_to_class=idx_to_class,
            example_inputs=example_input,
            rtol=0.05,
            atol=0.01,
        )

        if len(export_paths) == 0:
            return

        with open(os.path.join(self.export_folder, self.deploy_info_file), "w") as f:
            json.dump(self.model_json, f)

    def generate_report(self) -> None:
        """Generate a report for the task."""
        if self.datamodule.class_to_idx is None:
            log.warning("No `class_to_idx` found in the datamodule, report will not be generated")
            return

        if isinstance(self.datamodule, MultilabelClassificationDataModule):
            log.warning("Report generation is not supported for multilabel classification tasks at the moment.")
            return

        log.info("Generating report!")
        if not self.run_test or self.config.trainer.get("fast_dev_run"):
            self.datamodule.setup(stage="test")

        # Deepcopy to remove the inference mode from gradients causing issues when loading checkpoints
        # TODO: Why deepcopy of module model removes ModelSignatureWrapper?
        self.module.model.instance = deepcopy(self.module.model.instance)
        if "16" in self.trainer.precision:
            log.warning("Gradcam is currently not supported with half precision, it will be disabled")
            self.module.gradcam = False
            self.gradcam = False

        predictions_outputs = self.trainer.predict(
            model=self.module, datamodule=self.datamodule, ckpt_path=self.best_model_path
        )
        if not predictions_outputs:
            log.warning("There is no prediction to generate the report. Skipping report generation.")
            return
        all_outputs = [x[0] for x in predictions_outputs]
        all_probs = [x[2] for x in predictions_outputs]
        if not all_outputs or not all_probs:
            log.warning("There is no prediction to generate the report. Skipping report generation.")
            return
        all_outputs = [item for sublist in all_outputs for item in sublist]
        all_probs = [item for sublist in all_probs for item in sublist]
        all_targets = [target.tolist() for im, target in self.datamodule.test_dataloader()]
        all_targets = [item for sublist in all_targets for item in sublist]

        if self.module.gradcam:
            grayscale_cams = [x[1] for x in predictions_outputs]
            grayscale_cams = [item for sublist in grayscale_cams for item in sublist]
            grayscale_cams = np.stack(grayscale_cams)  # N x H x W
        else:
            grayscale_cams = None

        # creating confusion matrix
        idx_to_class = {v: k for k, v in self.datamodule.class_to_idx.items()}
        _, self.report_confmat, accuracy = get_results(
            test_labels=all_targets,
            pred_labels=all_outputs,
            idx_to_labels=idx_to_class,
        )
        output_folder_test = "test"
        test_dataloader = self.datamodule.test_dataloader()
        test_dataset = cast(ImageClassificationListDataset, test_dataloader.dataset)
        self.res = pd.DataFrame(
            {
                "sample": list(test_dataset.x),
                "real_label": all_targets,
                "pred_label": all_outputs,
                "probability": all_probs,
            }
        )
        os.makedirs(output_folder_test, exist_ok=True)
        save_classification_result(
            results=self.res,
            output_folder=output_folder_test,
            confmat=self.report_confmat,
            accuracy=accuracy,
            test_dataloader=self.datamodule.test_dataloader(),
            config=self.config,
            output=self.output,
            grayscale_cams=grayscale_cams,
        )

        if len(self.logger) > 0:
            mflow_logger = get_mlflow_logger(trainer=self.trainer)
            tensorboard_logger = utils.get_tensorboard_logger(trainer=self.trainer)
            artifacts = glob.glob(os.path.join(output_folder_test, "**/*"), recursive=True)
            if self.config.core.get("upload_artifacts") and len(artifacts) > 0:
                if mflow_logger is not None:
                    log.info("Uploading artifacts to MLFlow")
                    for a in artifacts:
                        if os.path.isdir(a):
                            continue

                        dirname = Path(a).parent.name
                        mflow_logger.experiment.log_artifact(
                            run_id=mflow_logger.run_id,
                            local_path=a,
                            artifact_path=os.path.join("classification_output", dirname),
                        )
                if tensorboard_logger is not None:
                    log.info("Uploading artifacts to Tensorboard")
                    for a in artifacts:
                        if os.path.isdir(a):
                            continue

                        ext = os.path.splitext(a)[1].lower()

                        if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"]:
                            try:
                                img = cv2.imread(a)
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            except cv2.error:
                                log.info("Could not upload artifact image %s", a)
                                continue
                            output_path = os.path.sep.join(a.split(os.path.sep)[-2:])
                            tensorboard_logger.experiment.add_image(output_path, img, 0, dataformats="HWC")
                        else:
                            utils.upload_file_tensorboard(a, tensorboard_logger)

    def freeze_layers_by_name(self, freeze_parameters_name: list[str]):
        """Freeze layers specified in freeze_parameters_name.

        Args:
            freeze_parameters_name: Layers that will be frozen during training.

        """
        count_frozen = 0
        for name, param in self.model.named_parameters():
            if any(x in name.split(".")[1] for x in freeze_parameters_name):
                log.debug("Freezing layer %s", name)
                param.requires_grad = False

            if not param.requires_grad:
                count_frozen += 1

        log.info("Frozen %d parameters", count_frozen)

    def freeze_parameters_by_index(self, freeze_parameters_index: list[int]):
        """Freeze parameters specified in freeze_parameters_name.

        Args:
            freeze_parameters_index: Indices of parameters that will be frozen during training.

        """
        if getattr(self.config.backbone, "freeze_parameters_name", None) is not None:
            log.warning(
                "Please be aware that some of the model's parameters have already been frozen using \
                the specified freeze_parameters_name. You are combining these two actions."
            )
        count_frozen = 0
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in freeze_parameters_index:
                log.debug("Freezing layer %s", name)
                param.requires_grad = False

            if not param.requires_grad:
                count_frozen += 1

        log.info("Frozen %d parameters", count_frozen)


class SklearnClassification(Generic[SklearnClassificationDataModuleT], Task[SklearnClassificationDataModuleT]):
    """Sklearn classification task.

    Args:
        config: The experiment configuration
        device: The device to use. Defaults to None.
        output: Dictionary defining which kind of outputs to generate. Defaults to None.
        automatic_batch_size: Whether to automatically find the largest batch size that fits in memory.
        save_model_summary: Whether to save a model_summary.txt file containing the model summary.
        half_precision: Whether to use half precision during training.
        gradcam: Whether to compute gradcams for test results.
    """

    def __init__(
        self,
        config: DictConfig,
        output: DictConfig,
        device: str,
        automatic_batch_size: DictConfig,
        save_model_summary: bool = False,
        half_precision: bool = False,
        gradcam: bool = False,
    ):
        super().__init__(config=config)

        self._device = device
        self.output = output
        self._backbone: ModelSignatureWrapper
        self._trainer: SklearnClassificationTrainer
        self._model: ClassifierMixin
        self.metadata: dict[str, Any] = {
            "test_confusion_matrix": [],
            "test_accuracy": [],
            "test_results": [],
            "test_labels": [],
            "cams": [],
        }
        self.export_folder = "deployment_model"
        self.deploy_info_file = "model.json"
        self.train_dataloader_list: list[torch.utils.data.DataLoader] = []
        self.test_dataloader_list: list[torch.utils.data.DataLoader] = []
        self.automatic_batch_size = automatic_batch_size
        self.save_model_summary = save_model_summary
        self.half_precision = half_precision
        self.gradcam = gradcam

    @property
    def device(self) -> str:
        return self._device

    def prepare(self) -> None:
        """Prepare the experiment."""
        self.datamodule = self.config.datamodule

        self.backbone = self.config.backbone

        self.model = self.config.model

        # prepare_data() must be explicitly called if the task does not include a lightining training
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="fit")

        self.trainer = self.config.trainer

    @property
    def model(self) -> ClassifierMixin:
        """sklearn.base.ClassifierMixin: The model."""
        return self._model

    @model.setter
    def model(self, model_config: DictConfig):
        """sklearn.base.ClassifierMixin: The model."""
        log.info("Instantiating model <%s>", model_config["_target_"])
        self._model = hydra.utils.instantiate(model_config)

    @property
    def backbone(self) -> ModelSignatureWrapper:
        """Backbone: The backbone."""
        return self._backbone

    @backbone.setter
    def backbone(self, backbone_config):
        """Load backbone."""
        if backbone_config.metadata.get("checkpoint"):
            log.info("Loading backbone from <%s>", backbone_config.metadata.checkpoint)
            self._backbone = torch.load(backbone_config.metadata.checkpoint)
        else:
            log.info("Loading backbone from <%s>", backbone_config.model["_target_"])
            self._backbone = hydra.utils.instantiate(backbone_config.model)

        self._backbone = ModelSignatureWrapper(self._backbone)
        self._backbone.eval()
        if self.half_precision:
            if self.device == "cpu":
                raise ValueError("Half precision is not supported on CPU")
            self._backbone.half()

            if self.gradcam:
                log.warning("Gradcam is currently not supported with half precision, it will be disabled")
                self.gradcam = False
        self._backbone.to(self.device)

    @property
    def trainer(self) -> SklearnClassificationTrainer:
        """Trainer: The trainer."""
        return self._trainer

    @trainer.setter
    def trainer(self, trainer_config: DictConfig) -> None:
        """Trainer: The trainer."""
        log.info("Instantiating trainer <%s>", trainer_config["_target_"])
        trainer = hydra.utils.instantiate(trainer_config, backbone=self.backbone, classifier=self.model)
        self._trainer = trainer

    @typing.no_type_check
    @automatic_datamodule_batch_size(batch_size_attribute_name="batch_size")
    def train(self) -> None:
        """Train the model."""
        log.info("Starting training...!")
        all_features = None
        all_labels = None

        class_to_keep = None

        self.train_dataloader_list = list(self.datamodule.train_dataloader())
        self.test_dataloader_list = list(self.datamodule.val_dataloader())

        if hasattr(self.datamodule, "class_to_keep_training") and self.datamodule.class_to_keep_training is not None:
            class_to_keep = self.datamodule.class_to_keep_training

        if self.save_model_summary:
            self.extract_model_summary(feature_extractor=self.backbone, dl=self.datamodule.full_dataloader())

        if hasattr(self.datamodule, "cache") and self.datamodule.cache:
            if self.config.trainer.iteration_over_training != 1:
                raise AttributeError("Cache is only supported when iteration over training is set to 1")

            full_dataloader = self.datamodule.full_dataloader()
            all_features, all_labels, _ = get_feature(
                feature_extractor=self.backbone, dl=full_dataloader, iteration_over_training=1
            )

            sorted_indices = np.argsort(full_dataloader.dataset.x)
            all_features = all_features[sorted_indices]
            all_labels = all_labels[sorted_indices]

        # cycle over all train/test split
        for train_dataloader, test_dataloader in zip(
            self.train_dataloader_list, self.test_dataloader_list, strict=False
        ):
            # Reinit classifier
            self.model = self.config.model
            self.trainer.change_classifier(self.model)

            # Train on current training set
            if all_features is not None and all_labels is not None:
                # Find which are the indices used to pass from the sorted list of string to the disordered one
                sorted_indices = np.argsort(np.concatenate([train_dataloader.dataset.x, test_dataloader.dataset.x]))
                revese_sorted_indices = np.argsort(sorted_indices)

                # Use these indices to correctly match the extracted features with the new file order
                all_features_sorted = all_features[revese_sorted_indices]
                all_labels_sorted = all_labels[revese_sorted_indices]

                train_len = len(train_dataloader.dataset.x)

                self.trainer.fit(
                    train_features=all_features_sorted[0:train_len], train_labels=all_labels_sorted[0:train_len]
                )

                _, pd_cm, accuracy, res, cams = self.trainer.test(
                    test_dataloader=test_dataloader,
                    test_features=all_features_sorted[train_len:],
                    test_labels=all_labels_sorted[train_len:],
                    class_to_keep=class_to_keep,
                    idx_to_class=train_dataloader.dataset.idx_to_class,
                    predict_proba=True,
                    gradcam=self.gradcam,
                )
            else:
                self.trainer.fit(train_dataloader=train_dataloader)
                _, pd_cm, accuracy, res, cams = self.trainer.test(
                    test_dataloader=test_dataloader,
                    class_to_keep=class_to_keep,
                    idx_to_class=train_dataloader.dataset.idx_to_class,
                    predict_proba=True,
                    gradcam=self.gradcam,
                )

            # save results
            self.metadata["test_confusion_matrix"].append(pd_cm)
            self.metadata["test_accuracy"].append(accuracy)
            self.metadata["test_results"].append(res)
            self.metadata["test_labels"].append(
                [
                    train_dataloader.dataset.idx_to_class[i] if i != -1 else "N/A"
                    for i in res["real_label"].unique().tolist()
                ]
            )
            self.metadata["cams"].append(cams)

    def extract_model_summary(
        self, feature_extractor: torch.nn.Module | BaseEvaluationModel, dl: torch.utils.data.DataLoader
    ) -> None:
        """Given a dataloader and a PyTorch model, use torchinfo to extract a summary of the model and save it
        to a file.

        Args:
            dl: PyTorch dataloader
            feature_extractor: PyTorch backbone
        """
        if isinstance(feature_extractor, TorchEvaluationModel | TorchscriptEvaluationModel):
            # TODO: I'm not sure torchinfo supports torchscript models
            # If we are working with torch based evaluation models we need to extract the model
            feature_extractor = feature_extractor.model

        for b in tqdm(dl):
            x1, _ = b

            if hasattr(feature_extractor, "parameters"):
                # Move input to the correct device
                parameter = next(feature_extractor.parameters())
                x1 = x1.to(parameter.device).to(parameter.dtype)
                x1 = x1[0].unsqueeze(0)  # Remove batch dimension

                model_info = None

                try:
                    try:
                        # TODO: Do we want to print the summary to the console as well?
                        model_info = summary(feature_extractor, input_data=(x1), verbose=0)  # type: ignore[arg-type]
                    except Exception:
                        log.warning(
                            "Failed to retrieve model summary using input data information, retrieving only "
                            "parameters information"
                        )
                        model_info = summary(feature_extractor, verbose=0)  # type: ignore[arg-type]
                except Exception as e:
                    # If for some reason the summary fails we don't want to stop the training
                    log.warning("Failed to retrieve model summary: %s", e)

                if model_info is not None:
                    with open("model_summary.txt", "w") as f:
                        f.write(str(model_info))
            else:
                log.warning("Failed to retrieve model summary, current model has no parameters")

            break

    @automatic_datamodule_batch_size(batch_size_attribute_name="batch_size")
    def train_full_data(self):
        """Train the model on train + validation."""
        # Reinit classifier
        self.model = self.config.model
        self.trainer.change_classifier(self.model)

        self.trainer.fit(train_dataloader=self.datamodule.full_dataloader())

    def test(self) -> None:
        """Skip test phase."""
        # we don't need test phase since sklearn trainer is already running test inside
        # train module to handle cross validation

    @typing.no_type_check
    @automatic_datamodule_batch_size(batch_size_attribute_name="batch_size")
    def test_full_data(self) -> None:
        """Test model trained on full dataset."""
        self.config.datamodule.class_to_idx = self.datamodule.full_dataset.class_to_idx
        self.config.datamodule.phase = "test"
        idx_to_class = self.datamodule.full_dataset.idx_to_class
        self.datamodule.setup("test")
        test_dataloader = self.datamodule.test_dataloader()

        if len(self.datamodule.data["samples"]) == 0:
            log.info("No test data, skipping test")
            return

        # Put backbone on the correct device as it may be moved after export
        self.backbone.to(self.device)
        _, pd_cm, accuracy, res, cams = self.trainer.test(
            test_dataloader=test_dataloader, idx_to_class=idx_to_class, predict_proba=True, gradcam=self.gradcam
        )

        output_folder_test = "test"

        os.makedirs(output_folder_test, exist_ok=True)

        save_classification_result(
            results=res,
            output_folder=output_folder_test,
            confmat=pd_cm,
            accuracy=accuracy,
            test_dataloader=test_dataloader,
            config=self.config,
            output=self.output,
            grayscale_cams=cams,
        )

    def export(self) -> None:
        """Generate deployment model for the task."""
        if self.config.export is None or len(self.config.export.types) == 0:
            log.info("No export type specified skipping export")
            return

        input_shapes = self.config.export.input_shapes

        idx_to_class = {v: k for k, v in self.datamodule.full_dataset.class_to_idx.items()}

        model_json, export_paths = export_model(
            config=self.config,
            model=self.backbone,
            export_folder=self.export_folder,
            half_precision=self.half_precision,
            input_shapes=input_shapes,
            idx_to_class=idx_to_class,
            pytorch_model_type="backbone",
        )

        dump(self.model, os.path.join(self.export_folder, "classifier.joblib"))

        if len(export_paths) > 0:
            with open(os.path.join(self.export_folder, self.deploy_info_file), "w") as f:
                json.dump(model_json, f)

    def generate_report(self) -> None:
        """Generate report for the task."""
        log.info("Generating report!")

        cm_list = []

        for count in range(len(self.metadata["test_accuracy"])):
            current_output_folder = f"{self.output.folder}_{count}"
            os.makedirs(current_output_folder, exist_ok=True)

            c_matrix = self.metadata["test_confusion_matrix"][count]
            cm_list.append(c_matrix)
            save_classification_result(
                results=self.metadata["test_results"][count],
                output_folder=current_output_folder,
                confmat=c_matrix,
                accuracy=self.metadata["test_accuracy"][count],
                test_dataloader=self.test_dataloader_list[count],
                config=self.config,
                output=self.output,
                grayscale_cams=self.metadata["cams"][count],
            )
        final_confusion_matrix = sum(cm_list)

        self.metadata["final_confusion_matrix"] = final_confusion_matrix
        # Save final conf matrix
        final_folder = f"{self.output.folder}"
        os.makedirs(final_folder, exist_ok=True)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=np.array(final_confusion_matrix),
            display_labels=[x.replace("pred:", "") for x in final_confusion_matrix.columns.to_list()],
        )
        disp.plot(include_values=True, cmap=plt.cm.Greens, ax=None, colorbar=False, xticks_rotation=90)
        plt.title(f"Confusion Matrix (Accuracy: {(self.metadata['test_accuracy'][count] * 100):.2f}%)")
        plt.savefig(os.path.join(final_folder, "test_confusion_matrix.png"), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

    def execute(self) -> None:
        """Execute the experiment and all the steps."""
        self.prepare()
        self.train()
        if self.output.report:
            self.generate_report()
        self.train_full_data()
        if self.config.export is not None and len(self.config.export.types) > 0:
            self.export()
        if self.output.test_full_data:
            self.test_full_data()
        self.finalize()


class SklearnTestClassification(Evaluation[SklearnClassificationDataModuleT]):
    """Perform a test using an imported SklearnClassification pytorch model.

    Args:
        config: The experiment configuration
        output: where to save results
        model_path: path to trained model generated from SklearnClassification task.
        device: the device where to run the model (cuda or cpu)
        gradcam: Whether to compute gradcams
        **kwargs: Additional arguments to pass to the task
    """

    def __init__(
        self,
        config: DictConfig,
        output: DictConfig,
        model_path: str,
        device: str,
        gradcam: bool = False,
        **kwargs: Any,
    ):
        super().__init__(config=config, model_path=model_path, device=device, **kwargs)
        self.gradcam = gradcam
        self.output = output
        self._backbone: BaseEvaluationModel
        self._classifier: ClassifierMixin
        self.class_to_idx: dict[str, int]
        self.idx_to_class: dict[int, str]
        self.test_dataloader: torch.utils.data.DataLoader
        self.metadata: dict[str, Any] = {
            "test_confusion_matrix": None,
            "test_accuracy": None,
            "test_results": None,
            "test_labels": None,
            "cams": None,
        }

    def prepare(self) -> None:
        """Prepare the experiment."""
        super().prepare()

        idx_to_class = {}
        class_to_idx = {}
        for k, v in self.model_data["classes"].items():
            idx_to_class[int(k)] = v
            class_to_idx[v] = int(k)

        self.idx_to_class = idx_to_class
        self.class_to_idx = class_to_idx

        self.config.datamodule.class_to_idx = class_to_idx

        self.datamodule = self.config.datamodule
        # prepare_data() must be explicitly called because there is no lightning training
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="test")

        # Configure trainer
        self.trainer = self.config.trainer

    @property
    def deployment_model(self):
        """Deployment model."""
        return None

    @deployment_model.setter
    def deployment_model(self, model_path: str):
        """Set backbone and classifier."""
        self.backbone = model_path  # type: ignore[assignment]
        # Load classifier
        self.classifier = os.path.join(Path(model_path).parent, "classifier.joblib")

    @property
    def classifier(self) -> ClassifierMixin:
        """Classifier: The classifier."""
        return self._classifier

    @classifier.setter
    def classifier(self, classifier_path: str) -> None:
        """Load classifier."""
        self._classifier = load(classifier_path)

    @property
    def backbone(self) -> BaseEvaluationModel:
        """Backbone: The backbone."""
        return self._backbone

    @backbone.setter
    def backbone(self, model_path: str) -> None:
        """Load backbone."""
        file_extension = os.path.splitext(model_path)[1]

        model_architecture = None
        if file_extension == ".pth":
            backbone_config_path = os.path.join(Path(model_path).parent, "model_config.yaml")
            log.info("Loading backbone from config")
            backbone_config = OmegaConf.load(backbone_config_path)

            if backbone_config.metadata.get("checkpoint"):
                log.info("Loading backbone from <%s>", backbone_config.metadata.checkpoint)
                model_architecture = torch.load(backbone_config.metadata.checkpoint)
            else:
                log.info("Loading backbone from <%s>", backbone_config.model["_target_"])
                model_architecture = hydra.utils.instantiate(backbone_config.model)

        self._backbone = import_deployment_model(
            model_path=model_path,
            device=self.device,
            inference_config=self.config.inference,
            model_architecture=model_architecture,
        )

        if self.gradcam and not isinstance(self._backbone, TorchEvaluationModel):
            log.warning("Gradcam is supported only for pytorch models. Skipping gradcam")
            self.gradcam = False

    @property
    def trainer(self) -> SklearnClassificationTrainer:
        """Trainer: The trainer."""
        return self._trainer

    @trainer.setter
    def trainer(self, trainer_config: DictConfig) -> None:
        """Trainer: The trainer."""
        log.info("Instantiating trainer <%s>", trainer_config["_target_"])

        if self.backbone.training:
            self.backbone.eval()

        trainer = hydra.utils.instantiate(trainer_config, backbone=self.backbone, classifier=self.classifier)
        self._trainer = trainer

    @automatic_datamodule_batch_size(batch_size_attribute_name="batch_size")
    def test(self) -> None:
        """Run the test."""
        self.test_dataloader = self.datamodule.test_dataloader()

        _, pd_cm, accuracy, res, cams = self.trainer.test(
            test_dataloader=self.test_dataloader,
            idx_to_class=self.idx_to_class,
            predict_proba=True,
            gradcam=self.gradcam,
        )

        # save results
        self.metadata["test_confusion_matrix"] = pd_cm
        self.metadata["test_accuracy"] = accuracy
        self.metadata["test_results"] = res
        self.metadata["test_labels"] = [
            self.idx_to_class[i] if i != -1 else "N/A" for i in res["real_label"].unique().tolist()
        ]
        self.metadata["cams"] = cams

    def generate_report(self) -> None:
        """Generate a report for the task."""
        log.info("Generating report!")
        os.makedirs(self.output.folder, exist_ok=True)
        save_classification_result(
            results=self.metadata["test_results"],
            output_folder=self.output.folder,
            confmat=self.metadata["test_confusion_matrix"],
            accuracy=self.metadata["test_accuracy"],
            test_dataloader=self.test_dataloader,
            config=self.config,
            output=self.output,
            grayscale_cams=self.metadata["cams"],
        )

    def execute(self) -> None:
        """Execute the experiment and all the steps."""
        self.prepare()
        self.test()
        if self.output.report:
            self.generate_report()
        self.finalize()


class ClassificationEvaluation(Evaluation[ClassificationDataModuleT]):
    """Perform a test on an imported Classification pytorch model.

    Args:
        config: Task configuration
        output: Configuration for the output
        model_path: Path to pytorch .pt model file
        report: Whether to generate the report of the predictions
        gradcam: Whether to compute gradcams
        device: Device to use for evaluation. If None, the device is automatically determined

    """

    def __init__(
        self,
        config: DictConfig,
        output: DictConfig,
        model_path: str,
        report: bool = True,
        gradcam: bool = False,
        device: str | None = None,
    ):
        super().__init__(config=config, model_path=model_path, device=device)
        self.report_path = "test_output"
        self.output = output
        self.report = report
        self.gradcam = gradcam
        self.cam: GradCAM

    def get_torch_model(self, model_config: DictConfig) -> nn.Module:
        """Instantiate the torch model from the config."""
        pre_classifier = self.get_pre_classifier(model_config)
        classifier = self.get_classifier(model_config)
        log.info("Instantiating backbone <%s>", model_config.model["_target_"])

        return hydra.utils.instantiate(
            model_config.model, classifier=classifier, pre_classifier=pre_classifier, _convert_="partial"
        )

    def get_pre_classifier(self, model_config: DictConfig) -> nn.Module:
        """Instantiate the pre-classifier from the config."""
        if "pre_classifier" in model_config and model_config.pre_classifier is not None:
            log.info("Instantiating pre_classifier <%s>", model_config.pre_classifier["_target_"])
            pre_classifier = hydra.utils.instantiate(model_config.pre_classifier, _convert_="partial")
        else:
            log.info("No pre-classifier found in config: instantiate a torch.nn.Identity instead")
            pre_classifier = nn.Identity()

        return pre_classifier

    def get_classifier(self, model_config: DictConfig) -> nn.Module:
        """Instantiate the classifier from the config."""
        if "classifier" in model_config:
            log.info("Instantiating classifier <%s>", model_config.classifier["_target_"])
            return hydra.utils.instantiate(
                model_config.classifier, out_features=len(self.model_data["classes"]), _convert_="partial"
            )

        raise ValueError("A `classifier` definition must be specified in the config")

    @property
    def deployment_model(self) -> BaseEvaluationModel:
        """Deployment model."""
        return self._deployment_model

    @deployment_model.setter
    def deployment_model(self, model_path: str):
        """Set the deployment model."""
        file_extension = os.path.splitext(model_path)[1]
        model_architecture = None
        if file_extension == ".pth":
            model_config = OmegaConf.load(os.path.join(Path(model_path).parent, "model_config.yaml"))

            if not isinstance(model_config, DictConfig):
                raise ValueError(f"The model config must be a DictConfig, got {type(model_config)}")

            model_architecture = self.get_torch_model(model_config)

        self._deployment_model = import_deployment_model(
            model_path=model_path,
            device=self.device,
            inference_config=self.config.inference,
            model_architecture=model_architecture,
        )

        if self.gradcam and not isinstance(self.deployment_model, TorchEvaluationModel):
            log.warning("To compute gradcams you need to provide the path to an exported .pth state_dict file")
            self.gradcam = False

    def prepare(self) -> None:
        """Prepare the evaluation."""
        super().prepare()
        self.datamodule = self.config.datamodule
        self.datamodule.class_to_idx = {v: int(k) for k, v in self.model_data["classes"].items()}
        self.datamodule.num_classes = len(self.datamodule.class_to_idx)

        # prepare_data() must be explicitly called because there is no training
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="test")

    def prepare_gradcam(self) -> None:
        """Initializing gradcam for the predictions."""
        if not hasattr(self.deployment_model.model, "features_extractor"):
            log.warning("Gradcam not implemented for this backbone, it will not be computed")
            self.gradcam = False
            return

        if isinstance(self.deployment_model.model.features_extractor, timm.models.resnet.ResNet):
            target_layers = [cast(BaseNetworkBuilder, self.deployment_model.model).features_extractor.layer4[-1]]  # type: ignore[index]
            self.cam = GradCAM(
                model=self.deployment_model.model,
                target_layers=target_layers,
            )
            for p in self.deployment_model.model.features_extractor.layer4[-1].parameters():
                p.requires_grad = True
        elif is_vision_transformer(cast(BaseNetworkBuilder, self.deployment_model.model).features_extractor):
            self.grad_rollout = VitAttentionGradRollout(cast(nn.Module, self.deployment_model.model))
        else:
            log.warning("Gradcam not implemented for this backbone, it will not be computed")
            self.gradcam = False

    @automatic_datamodule_batch_size(batch_size_attribute_name="batch_size")
    def test(self) -> None:
        """Perform test."""
        log.info("Running test")
        test_dataloader = self.datamodule.test_dataloader()

        image_labels = []
        probabilities = []
        predicted_classes = []
        grayscale_cams_list = []

        if self.gradcam:
            self.prepare_gradcam()

        with torch.set_grad_enabled(self.gradcam):
            for batch_item in tqdm(test_dataloader):
                im, target = batch_item
                im = im.to(device=self.device, dtype=self.deployment_model.model_dtype).detach()

                if self.gradcam:
                    # When gradcam is used we need to remove gradients
                    outputs = self.deployment_model(im).detach()
                else:
                    outputs = self.deployment_model(im)

                probs = torch.softmax(outputs, dim=1)
                preds = torch.max(probs, dim=1).indices

                probabilities.append(probs.tolist())
                predicted_classes.append(preds.tolist())
                image_labels.extend(target.tolist())
                if self.gradcam and hasattr(self.deployment_model.model, "features_extractor"):
                    with torch.inference_mode(False):
                        im = im.clone()
                        if isinstance(self.deployment_model.model.features_extractor, timm.models.resnet.ResNet):
                            grayscale_cam = self.cam(input_tensor=im, targets=None)
                            grayscale_cams_list.append(torch.from_numpy(grayscale_cam))
                        elif is_vision_transformer(
                            cast(BaseNetworkBuilder, self.deployment_model.model).features_extractor
                        ):
                            grayscale_cam_low_res = self.grad_rollout(input_tensor=im, targets_list=preds.tolist())
                            orig_shape = grayscale_cam_low_res.shape
                            new_shape = (orig_shape[0], im.shape[2], im.shape[3])
                            zoom_factors = tuple(np.array(new_shape) / np.array(orig_shape))
                            grayscale_cam = ndimage.zoom(grayscale_cam_low_res, zoom_factors, order=1)
                            grayscale_cams_list.append(torch.from_numpy(grayscale_cam))

        grayscale_cams: torch.Tensor | None = None
        if self.gradcam:
            grayscale_cams = torch.cat(grayscale_cams_list, dim=0)

        predicted_classes = [item for sublist in predicted_classes for item in sublist]
        probabilities = [max(item) for sublist in probabilities for item in sublist]
        if self.datamodule.class_to_idx is not None:
            idx_to_class = {v: k for k, v in self.datamodule.class_to_idx.items()}
        else:
            idx_to_class = None

        _, pd_cm, test_accuracy = get_results(
            test_labels=image_labels,
            pred_labels=predicted_classes,
            idx_to_labels=idx_to_class,
        )

        res = pd.DataFrame(
            {
                "sample": list(test_dataloader.dataset.x),  # type: ignore[attr-defined]
                "real_label": image_labels,
                "pred_label": predicted_classes,
                "probability": probabilities,
            }
        )

        log.info("Avg classification accuracy: %s", test_accuracy)

        self.res = pd.DataFrame(
            {
                "sample": list(test_dataloader.dataset.x),  # type: ignore[attr-defined]
                "real_label": image_labels,
                "pred_label": predicted_classes,
                "probability": probabilities,
            }
        )

        # save results
        self.metadata["test_confusion_matrix"] = pd_cm
        self.metadata["test_accuracy"] = test_accuracy
        self.metadata["predictions"] = predicted_classes
        self.metadata["test_results"] = res
        self.metadata["probabilities"] = probabilities
        self.metadata["test_labels"] = image_labels
        self.metadata["grayscale_cams"] = grayscale_cams

    def generate_report(self) -> None:
        """Generate a report for the task."""
        log.info("Generating report!")
        os.makedirs(self.report_path, exist_ok=True)

        save_classification_result(
            results=self.metadata["test_results"],
            output_folder=self.report_path,
            confmat=self.metadata["test_confusion_matrix"],
            accuracy=self.metadata["test_accuracy"],
            test_dataloader=self.datamodule.test_dataloader(),
            config=self.config,
            output=self.output,
            grayscale_cams=self.metadata["grayscale_cams"],
        )

    def execute(self) -> None:
        """Execute the evaluation."""
        self.prepare()
        self.test()
        if self.report:
            self.generate_report()
        self.finalize()
