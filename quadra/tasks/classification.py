import glob
import json
import os
import typing
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, cast

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from omegaconf import DictConfig
from sklearn.base import ClassifierMixin
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

from quadra.callbacks.mlflow import get_mlflow_logger
from quadra.callbacks.scheduler import WarmupInit
from quadra.datamodules import (
    ClassificationDataModule,
    MultilabelClassificationDataModule,
    SklearnClassificationDataModule,
)
from quadra.datasets.classification import ImageClassificationListDataset
from quadra.modules.classification import ClassificationModule
from quadra.tasks.base import LightningTask, Task
from quadra.trainers.classification import SklearnClassificationTrainer
from quadra.utils import utils
from quadra.utils.classification import get_results, save_classification_result
from quadra.utils.export import export_torchscript_model, import_deployment_model
from quadra.utils.models import get_feature

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
        export_type: List of export method for the model, e.g. [torchscript]. Defaults to None.
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
        checkpoint_path: Optional[str] = None,
        lr_multiplier: Optional[float] = None,
        export_type: Optional[List[str]] = None,
        report: bool = False,
        run_test: bool = False,
    ):
        super().__init__(
            config=config, checkpoint_path=checkpoint_path, run_test=run_test, report=report, export_type=export_type
        )
        self.output = output
        self._lr_multiplier = lr_multiplier
        self._pre_classifier: torch.nn.Module
        self._classifier: torch.nn.Module
        self._model: torch.nn.Module
        self._optimizer: torch.optim.Optimizer
        self._scheduler: torch.optim.lr_scheduler._LRScheduler
        self.export_folder: str = "deployment_model"
        self.report_confmat: pd.DataFrame

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_config: DictConfig) -> None:
        """Set the optimizer."""
        if (
            isinstance(self.model.features_extractor, torch.nn.Module)
            and isinstance(self.model.pre_classifier, torch.nn.Module)
            and isinstance(self.model.classifier, torch.nn.Module)
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
    def module(self, module_config):
        """Set the module of the model."""
        module = hydra.utils.instantiate(
            module_config,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
        )
        if self.checkpoint_path is not None:
            log.info("Loading model from lightning checkpoint: %s", self.checkpoint_path)
            module = module.load_from_checkpoint(
                self.checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
                criterion=module.criterion,
                gradcam=self.config.model.module.gradcam,
            )
        self._module = module

    @property
    def pre_classifier(self) -> torch.nn.Module:
        return self._pre_classifier

    @pre_classifier.setter
    def pre_classifier(self, model_config: DictConfig) -> None:
        if "pre_classifier" in model_config and model_config.pre_classifier is not None:
            log.info("Instantiating pre_classifier <%s>", model_config.pre_classifier["_target_"])
            self._pre_classifier = hydra.utils.instantiate(model_config.pre_classifier, _convert_="partial")
        else:
            log.info("No pre-classifier found in the config: instantiate a torch.nn.Identity instead")
            self._pre_classifier = torch.nn.Identity()

    @property
    def classifier(self) -> torch.nn.Module:
        return self._classifier

    @classifier.setter
    def classifier(self, model_config: DictConfig) -> None:
        if "classifier" in model_config:
            log.info("Instantiating classifier <%s>", model_config.classifier["_target_"])
            self._classifier = hydra.utils.instantiate(model_config.classifier, _convert_="partial")
        else:
            raise ValueError("A `classifier` definition must be specified in the config")

    @property
    def model(self) -> torch.nn.Module:
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
            self.freeze_layers(self.config.backbone.freeze_parameters_name)

    def prepare(self) -> None:
        """Prepare the experiment."""
        super().prepare()
        self.model = self.config.model
        self.optimizer = self.config.optimizer
        self.scheduler = self.config.scheduler
        self.module = self.config.model.module

    def test(self) -> None:
        """Test the model."""
        if not self.config.trainer.get("fast_dev_run"):
            if self.trainer.checkpoint_callback is None:
                raise ValueError("Checkpoint callback is not defined!")
            log.info("Starting testing!")
            self.datamodule.setup(stage="test")
            log.info("Using best epoch's weights for testing.")
            self.trainer.test(datamodule=self.datamodule, model=self.module, ckpt_path="best")

    def export(self) -> None:
        """Generate deployment model for the task."""
        if self.export_type is None:
            raise ValueError("`Export_type` must be specified in the config to export the model")

        input_width = self.config.transforms.get("input_width")
        input_height = self.config.transforms.get("input_height")

        if self.trainer.checkpoint_callback is None:
            raise ValueError("No checkpoint callback found in the trainer")
        best_model_path = self.trainer.checkpoint_callback.best_model_path  # type: ignore[attr-defined]
        log.info("Saving runtime model for %s checkpoint", best_model_path)

        module = self.module.load_from_checkpoint(
            best_model_path,
            model=self.module.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            criterion=self.module.criterion,
            gradcam=self.module.gradcam,
        )

        for export_type in self.export_type:
            if export_type == "torchscript":
                export_torchscript_model(
                    module.model,
                    (1, 3, input_height, input_width),
                    self.export_folder,
                    half_precision=int(self.trainer.precision) == 16,
                )
        if self.datamodule.class_to_idx is None:
            log.warning(
                "No `class_to_idx` found in the datamodule, class information will not be saved in the model.json"
            )
            idx_to_class = {}
        else:
            idx_to_class = {v: k for k, v in self.datamodule.class_to_idx.items()}
        model_json = {
            "input_size": [input_width, input_height, 3],
            "classes": idx_to_class,
            "mean": list(self.config.transforms.mean),
            "std": list(self.config.transforms.std),
        }

        with open(os.path.join(self.export_folder, "model.json"), "w") as f:
            json.dump(model_json, f)

    def generate_report(self) -> None:
        """Generate a report for the task."""
        if self.datamodule.class_to_idx is None:
            log.warning("No `class_to_idx` found in the datamodule, report will not be generated")
            return

        if isinstance(self.datamodule, MultilabelClassificationDataModule):
            log.warning("Report generation is not supported for multilabel classification tasks at the moment.")
            return

        log.info("Generating report!")

        predictions_outputs = self.trainer.predict(model=self.module, datamodule=self.datamodule, ckpt_path="best")
        if not predictions_outputs:
            log.warning("There is no prediction to generate the report. Skipping report generation.")
            return
        all_outputs = [x[0] for x in predictions_outputs]
        if not all_outputs:
            log.warning("There is no prediction to generate the report. Skipping report generation.")
            return
        all_outputs = [item for sublist in all_outputs for item in sublist]
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
        res = pd.DataFrame(
            {
                "sample": list(test_dataset.x),
                "real_label": all_targets,
                "pred_label": all_outputs,
            }
        )
        os.makedirs(output_folder_test, exist_ok=True)
        save_classification_result(
            results=res,
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

    def freeze_layers(self, freeze_parameters_name: List[str]):
        """Freeze layers specified in freeze_parameters_name
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


class SklearnClassification(Generic[SklearnClassificationDataModuleT], Task[SklearnClassificationDataModuleT]):
    """Sklearn classification task.

    Args:
        config: The experiment configuration
        device: The device to use. Defaults to None.
        output: Dictionary defining which kind of outputs to generate. Defaults to None.
        export_type: List of export method for the model, e.g. [torchscript]. Defaults to None.
    """

    def __init__(
        self,
        config: DictConfig,
        output: DictConfig,
        device: str,
        export_type: Optional[List[str]] = None,
    ):
        super().__init__(config=config, export_type=export_type)

        self._device = device
        self.output = output
        self._backbone: torch.nn.Module
        self._trainer: SklearnClassificationTrainer
        self._model: ClassifierMixin
        self.metadata: Dict[str, Any] = {
            "test_confusion_matrix": [],
            "test_accuracy": [],
            "test_results": [],
            "test_labels": [],
        }
        self.export_folder: str = "deployment_model"
        self.train_dataloader_list: List[torch.utils.data.DataLoader] = []
        self.test_dataloader_list: List[torch.utils.data.DataLoader] = []

    @property
    def device(self) -> str:
        return self._device

    def prepare(self) -> None:
        """Prepare the experiment."""
        self.datamodule = self.config.datamodule

        self.backbone = self.config.backbone

        self.model = self.config.model
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="fit")

        self.train_dataloader_list = list(self.datamodule.train_dataloader())
        self.test_dataloader_list = list(self.datamodule.val_dataloader())
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
    def backbone(self) -> torch.nn.Module:
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
        self._backbone.eval()
        self._backbone = self._backbone.to(self.device)

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
    def train(self) -> None:
        """Train the model."""
        log.info("Starting training...!")
        all_features = None
        all_labels = None

        class_to_keep = None

        if hasattr(self.datamodule, "class_to_keep_training") and self.datamodule.class_to_keep_training is not None:
            class_to_keep = self.datamodule.class_to_keep_training

        if hasattr(self.datamodule, "cache") and self.datamodule.cache:
            if self.config.trainer.iteration_over_training != 1:
                raise AttributeError("Cache is only supported when iteration over training is set to 1")

            full_dataloader = self.datamodule.full_dataloader()
            all_features, all_labels = get_feature(
                feature_extractor=self.backbone, dl=full_dataloader, iteration_over_training=1
            )

            sorted_indices = np.argsort(full_dataloader.dataset.x)
            all_features = all_features[sorted_indices]
            all_labels = all_labels[sorted_indices]

        # cycle over all train/test split
        for train_dataloader, test_dataloader in zip(self.train_dataloader_list, self.test_dataloader_list):
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

                _, pd_cm, accuracy, res = self.trainer.test(
                    test_dataloader=test_dataloader,
                    test_features=all_features_sorted[train_len:],
                    test_labels=all_labels_sorted[train_len:],
                    class_to_keep=class_to_keep,
                    idx_to_class=train_dataloader.dataset.idx_to_class,
                    predict_proba=True,
                )
            else:
                self.trainer.fit(train_dataloader=train_dataloader)
                _, pd_cm, accuracy, res = self.trainer.test(
                    test_dataloader=test_dataloader,
                    class_to_keep=class_to_keep,
                    idx_to_class=train_dataloader.dataset.idx_to_class,
                    predict_proba=True,
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

        _, pd_cm, accuracy, res = self.trainer.test(
            test_dataloader=test_dataloader, idx_to_class=idx_to_class, predict_proba=True
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
        )

    def export(self) -> None:
        """Generate deployment model for the task."""
        input_width = self.config.transforms.get("input_width")
        input_height = self.config.transforms.get("input_height")

        if self.export_type is None:
            log.info("No export type specified, skipping export")
            return

        for export_type in self.export_type:
            if export_type == "torchscript":
                export_torchscript_model(
                    self.backbone, (1, 3, input_height, input_width), self.export_folder, half_precision=False
                )

        dump(self.model, os.path.join(self.export_folder, "classifier.joblib"))

        idx_to_class = {v: k for k, v in self.datamodule.full_dataset.class_to_idx.items()}
        model_json = {
            "input_size": [input_width, input_height, 3],
            "classes": idx_to_class,
            "mean": list(self.config.transforms.mean),
            "std": list(self.config.transforms.std),
        }

        with open(os.path.join(self.export_folder, "model.json"), "w") as f:
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
        if self.export_type is not None and len(self.export_type) > 0:
            self.export()
        if self.output.test_full_data:
            self.test_full_data()
        self.finalize()


class SklearnTestClassification(Task[SklearnClassificationDataModuleT]):
    """Perform a test of an already trained classification model.

    Args:
        config: The experiment configuration
        output: where to save results
        experiment_path: path to training experiment generated from SklearnClassification task.
        device: the device where to run the model (cuda or cpu)
    """

    def __init__(
        self,
        config: DictConfig,
        output: DictConfig,
        experiment_path: str,
        device: str,
    ):
        super().__init__(config=config)
        self._device = device
        self.output = output
        self.experiment_path = experiment_path
        self._backbone: torch.nn.Module
        self._classifier: ClassifierMixin
        self.class_to_idx: Dict[str, int]
        self.idx_to_class: Dict[int, str]
        self.runtime_info_file = "model.json"
        self.test_dataloader: torch.utils.data.DataLoader
        self.metadata: Dict[str, Any] = {
            "test_confusion_matrix": None,
            "test_accuracy": None,
            "test_results": None,
            "test_labels": None,
        }
        self.model_info: Any

    def prepare(self) -> None:
        """Prepare the experiment."""
        # Read the information of the already trained model
        with open(os.path.join(self.experiment_path, "deployment_model", self.runtime_info_file), "r") as f:
            self.model_info = json.load(f)

        idx_to_class = {}
        class_to_idx = {}
        for k, v in self.model_info["classes"].items():
            idx_to_class[int(k)] = v
            class_to_idx[v] = int(k)

        self.idx_to_class = idx_to_class
        self.class_to_idx = class_to_idx

        self.config.datamodule.class_to_idx = class_to_idx

        # Setup datamodule
        self.datamodule = self.config.datamodule
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="test")

        self.test_dataloader = self.datamodule.test_dataloader()

        # TODO: It is not currently possible to load a deployed backbone!!
        self.backbone = self.config.backbone

        # Load classifier
        self.classifier = os.path.join(self.experiment_path, "deployment_model", "classifier.joblib")

        # Configure trainer
        self.trainer = self.config.trainer

    @property
    def classifier(self) -> ClassifierMixin:
        """Classifier: The classifier."""
        return self._classifier

    @classifier.setter
    def classifier(self, classifier_path: str) -> None:
        """Load classifier."""
        self._classifier = load(classifier_path)

    @property
    def backbone(self) -> torch.nn.Module:
        """Backbone: The backbone."""
        return self._backbone

    @backbone.setter
    def backbone(self, backbone_config: DictConfig) -> None:
        """Load backbone."""
        if backbone_config.metadata.get("checkpoint"):
            log.info("Loading backbone from <%s>", backbone_config.metadata.checkpoint)
            self._backbone = torch.load(backbone_config.metadata.checkpoint)
        else:
            log.info("Loading backbone from <%s>", backbone_config.model["_target_"])
            self._backbone = hydra.utils.instantiate(backbone_config.model)
        self._backbone.eval()
        self._backbone = self._backbone.to(self.device)

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

    def test(self) -> None:
        """Run the test."""
        _, pd_cm, accuracy, res = self.trainer.test(
            test_dataloader=self.test_dataloader, idx_to_class=self.idx_to_class, predict_proba=True
        )

        # save results
        self.metadata["test_confusion_matrix"] = pd_cm
        self.metadata["test_accuracy"] = accuracy
        self.metadata["test_results"] = res
        self.metadata["test_labels"] = [
            self.idx_to_class[i] if i != -1 else "N/A" for i in res["real_label"].unique().tolist()
        ]

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
        )

    def execute(self) -> None:
        """Execute the experiment and all the steps."""
        self.prepare()
        self.test()
        if self.output.report:
            self.generate_report()
        self.finalize()

    @property
    def device(self) -> str:
        return self._device


class ClassificationEvaluation(Task[ClassificationDataModule]):
    """Evaluation task for Classification.

    Args:
        config: Task configuration
        output: Configuration for the output
        model_path: Path to .pt model file
        device: Device to use for evaluation. If None, the device is automatically determined

    """

    def __init__(
        self,
        config: DictConfig,
        output: DictConfig,
        model_path: str,
        device: Optional[str] = None,
    ):
        super().__init__(config=config)
        self.model_data: Dict[str, Any]
        self._deployment_model: Any
        self.deployment_model_type: str
        self.model_path = model_path
        self.output_path = "test_output"
        self.output = output
        self.config = config
        self.report_path = ""
        self.metadata = {"report_files": []}
        self.model_info_filename = "model.json"
        if device is None:
            self.device = utils.get_device()
        else:
            self.device = device

    @property
    def deployment_model(self):
        """Deployment model."""
        return self._deployment_model

    @deployment_model.setter
    def deployment_model(self, model_path: str):
        """Set the deployment model."""
        self._deployment_model, self.deployment_model_type = import_deployment_model(model_path, self.device)

    def prepare(self) -> None:
        """Prepare the evaluation."""
        with open(os.path.join(Path(self.model_path).parent, self.model_info_filename)) as f:
            self.model_data = json.load(f)

        if not isinstance(self.model_data, dict):
            raise ValueError("Model info file is not a valid json")

        if self.model_data["input_size"][0] != self.config.transforms.input_height:
            log.warning(
                f"Input height of the model ({self.model_data['input_size'][0]}) is different from the one specified "
                + f"in the config ({self.config.transforms.input_height}). Fixing the config."
            )
            self.config.transforms.input_height = self.model_data["input_size"][0]

        if self.model_data["input_size"][1] != self.config.transforms.input_width:
            log.warning(
                f"Input width of the model ({self.model_data['input_size'][1]}) is different from the one specified "
                + f"in the config ({self.config.transforms.input_width}). Fixing the config."
            )
            self.config.transforms.input_width = self.model_data["input_size"][1]

        self.deployment_model = self.model_path

        super().prepare()
        self.datamodule.class_to_idx = {v: int(k) for k, v in self.model_data["classes"].items()}

    def test(self) -> None:
        """Perform test."""
        log.info("Running test")
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="test")
        test_dataloader = self.datamodule.test_dataloader()

        image_labels = []
        predicted_classes = []

        with torch.no_grad():
            for batch_item in tqdm(test_dataloader):
                im, target = batch_item

                outputs = self.deployment_model(im.to(self.device))
                probs = torch.softmax(outputs, dim=1)
                preds = torch.max(probs, dim=1).indices

                predicted_classes.append(preds.tolist())
                image_labels.extend(target.tolist())

        predicted_classes = [item for sublist in predicted_classes for item in sublist]
        if self.datamodule.class_to_idx is not None:
            idx_to_class = {v: k for k, v in self.datamodule.class_to_idx.items()}

        _, pd_cm, test_accuracy = get_results(
            test_labels=image_labels,
            pred_labels=predicted_classes,
            idx_to_labels=idx_to_class,
        )
        log.info("Avg classification accuracy: %s", test_accuracy)

        # save results
        self.metadata["test_confusion_matrix"] = pd_cm
        self.metadata["test_accuracy"] = test_accuracy
        self.metadata["test_results"] = predicted_classes
        self.metadata["test_labels"] = image_labels

    def generate_report(self) -> None:
        """Generate a report for the task."""
        log.info("Generating report!")
        os.makedirs(self.output_path, exist_ok=True)

        test_dataset = cast(ImageClassificationListDataset, self.datamodule.test_dataloader().dataset)
        res = pd.DataFrame(
            {
                "sample": list(test_dataset.x),
                "real_label": self.metadata["test_labels"],
                "pred_label": self.metadata["test_results"],
            }
        )
        os.makedirs(self.output_path, exist_ok=True)
        save_classification_result(
            results=res,
            output_folder=self.output_path,
            confmat=self.metadata["test_confusion_matrix"],
            accuracy=self.metadata["test_accuracy"],
            test_dataloader=self.datamodule.test_dataloader(),
            config=self.config,
            output=self.output,
            grayscale_cams=None,
        )

    def execute(self) -> None:
        """Execute the evaluation."""
        self.prepare()
        self.test()
        self.generate_report()
        log.info("Evaluation finished")
        self.finalize()
