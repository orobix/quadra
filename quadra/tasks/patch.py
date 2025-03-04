from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from joblib import dump, load
from omegaconf import DictConfig, OmegaConf
from sklearn.base import ClassifierMixin

from quadra.datamodules import PatchSklearnClassificationDataModule
from quadra.datasets.patch import PatchSklearnClassificationTrainDataset
from quadra.models.base import ModelSignatureWrapper
from quadra.models.evaluation import BaseEvaluationModel
from quadra.tasks.base import Evaluation, Task
from quadra.trainers.classification import SklearnClassificationTrainer
from quadra.utils import utils
from quadra.utils.classification import automatic_batch_size_computation
from quadra.utils.evaluation import automatic_datamodule_batch_size
from quadra.utils.export import export_model, import_deployment_model
from quadra.utils.patch import RleEncoder, compute_patch_metrics, save_classification_result
from quadra.utils.patch.dataset import PatchDatasetFileFormat

log = utils.get_logger(__name__)


class PatchSklearnClassification(Task[PatchSklearnClassificationDataModule]):
    """Patch classification using torch backbone for feature extraction and sklearn to learn a linear classifier.

    Args:
        config: The experiment configuration
        device: The device to use
        output: Dictionary defining which kind of outputs to generate. Defaults to None.
        automatic_batch_size: Whether to automatically find the largest batch size that fits in memory.
        half_precision: Whether to use half precision.
    """

    def __init__(
        self,
        config: DictConfig,
        output: DictConfig,
        device: str,
        automatic_batch_size: DictConfig,
        half_precision: bool = False,
    ):
        super().__init__(config=config)
        self.device: str = device
        self.output: DictConfig = output
        self.return_polygon: bool = True
        self.reconstruction_results: dict[str, Any]
        self._backbone: ModelSignatureWrapper
        self._trainer: SklearnClassificationTrainer
        self._model: ClassifierMixin
        self.metadata: dict[str, Any] = {
            "test_confusion_matrix": [],
            "test_accuracy": [],
            "test_results": [],
            "test_labels": [],
        }
        self.export_folder: str = "deployment_model"
        self.automatic_batch_size = automatic_batch_size
        self.half_precision = half_precision

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
        self._backbone = self._backbone.to(self.device)

    def prepare(self) -> None:
        """Prepare the experiment."""
        self.datamodule = self.config.datamodule
        self.backbone = self.config.backbone
        self.model = self.config.model

        if not self.automatic_batch_size.disable and self.device != "cpu":
            self.datamodule.batch_size = automatic_batch_size_computation(
                datamodule=self.datamodule,
                backbone=self.backbone,
                starting_batch_size=self.automatic_batch_size.starting_batch_size,
            )

        self.trainer = self.config.trainer

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

    def train(self) -> None:
        """Train the model."""
        log.info("Starting training...!")
        # prepare_data() must be explicitly called if the task does not include a lightining training
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="fit")
        class_to_keep = None
        if hasattr(self.datamodule, "class_to_skip_training") and self.datamodule.class_to_skip_training is not None:
            class_to_keep = [x for x in self.datamodule.class_to_idx if x not in self.datamodule.class_to_skip_training]

        self.model = self.config.model
        self.trainer.change_classifier(self.model)
        train_dataloader = self.datamodule.train_dataloader()
        val_dataloader = self.datamodule.val_dataloader()
        train_dataset = cast(PatchSklearnClassificationTrainDataset, train_dataloader.dataset)
        self.trainer.fit(train_dataloader=train_dataloader)
        _, pd_cm, accuracy, res, _ = self.trainer.test(
            test_dataloader=val_dataloader,
            class_to_keep=class_to_keep,
            idx_to_class=train_dataset.idx_to_class,
            predict_proba=True,
        )

        # save results
        self.metadata["test_confusion_matrix"] = pd_cm
        self.metadata["test_accuracy"] = accuracy
        self.metadata["test_results"] = res
        self.metadata["test_labels"] = [
            train_dataset.idx_to_class[i] if i != -1 else "N/A" for i in res["real_label"].unique().tolist()
        ]

    def generate_report(self) -> None:
        """Generate the report for the task."""
        log.info("Generating report!")
        os.makedirs(self.output.folder, exist_ok=True)

        c_matrix = self.metadata["test_confusion_matrix"]
        idx_to_class = {v: k for k, v in self.datamodule.class_to_idx.items()}

        datamodule: PatchSklearnClassificationDataModule = self.datamodule
        val_img_info: list[PatchDatasetFileFormat] = datamodule.info.val_files
        for img_info in val_img_info:
            if not os.path.isabs(img_info.image_path):
                img_info.image_path = os.path.join(datamodule.data_path, img_info.image_path)
            if img_info.mask_path is not None and not os.path.isabs(img_info.mask_path):
                img_info.mask_path = os.path.join(datamodule.data_path, img_info.mask_path)

        false_region_bad, false_region_good, true_region_bad, reconstructions = compute_patch_metrics(
            test_img_info=val_img_info,
            test_results=self.metadata["test_results"],
            patch_num_h=datamodule.info.patch_number[0] if datamodule.info.patch_number is not None else None,
            patch_num_w=datamodule.info.patch_number[1] if datamodule.info.patch_number is not None else None,
            patch_h=datamodule.info.patch_size[0] if datamodule.info.patch_size is not None else None,
            patch_w=datamodule.info.patch_size[1] if datamodule.info.patch_size is not None else None,
            overlap=datamodule.info.overlap,
            idx_to_class=idx_to_class,
            return_polygon=self.return_polygon,
            patch_reconstruction_method=self.output.reconstruction_method,
            annotated_good=datamodule.info.annotated_good,
        )

        self.reconstruction_results = {
            "false_region_bad": false_region_bad,
            "false_region_good": false_region_good,
            "true_region_bad": true_region_bad,
            "reconstructions": reconstructions,
            "reconstructions_type": "polygon" if self.return_polygon else "rle",
            "patch_reconstruction_method": self.output.reconstruction_method,
        }

        with open("reconstruction_results.json", "w") as f:
            json.dump(
                self.reconstruction_results,
                f,
                cls=RleEncoder,
            )

        if hasattr(self.datamodule, "class_to_skip_training") and self.datamodule.class_to_skip_training is not None:
            ignore_classes = [self.datamodule.class_to_idx[x] for x in self.datamodule.class_to_skip_training]
        else:
            ignore_classes = None
        val_dataloader = self.datamodule.val_dataloader()
        save_classification_result(
            results=self.metadata["test_results"],
            output_folder=self.output.folder,
            confusion_matrix=c_matrix,
            accuracy=self.metadata["test_accuracy"],
            test_dataloader=val_dataloader,
            config=self.config,
            output=self.output,
            reconstructions=reconstructions,
            ignore_classes=ignore_classes,
        )

    def export(self) -> None:
        """Generate deployment model for the task."""
        input_shapes = self.config.export.input_shapes

        idx_to_class = {v: k for k, v in self.datamodule.class_to_idx.items()}

        model_json, export_paths = export_model(
            config=self.config,
            model=self.backbone,
            export_folder=self.export_folder,
            half_precision=self.half_precision,
            input_shapes=input_shapes,
            idx_to_class=idx_to_class,
            pytorch_model_type="backbone",
        )

        if len(export_paths) > 0:
            dataset_info = self.datamodule.info

            horizontal_patches = dataset_info.patch_number[1] if dataset_info.patch_number is not None else None
            vertical_patches = dataset_info.patch_number[0] if dataset_info.patch_number is not None else None
            patch_height = dataset_info.patch_size[0] if dataset_info.patch_size is not None else None
            patch_width = dataset_info.patch_size[1] if dataset_info.patch_size is not None else None
            overlap = dataset_info.overlap

            model_json.update(
                {
                    "horizontal_patches": horizontal_patches,
                    "vertical_patches": vertical_patches,
                    "patch_height": patch_height,
                    "patch_width": patch_width,
                    "overlap": overlap,
                    "reconstruction_method": self.output.reconstruction_method,
                    "class_to_skip": self.datamodule.class_to_skip_training,
                }
            )

            with open(os.path.join(self.export_folder, "model.json"), "w") as f:
                json.dump(model_json, f, cls=utils.HydraEncoder)

        dump(self.model, os.path.join(self.export_folder, "classifier.joblib"))

    def execute(self) -> None:
        """Execute the experiment and all the steps."""
        self.prepare()
        self.train()
        if self.output.report:
            self.generate_report()
        if self.config.export is not None and len(self.config.export.types) > 0:
            self.export()
        self.finalize()


class PatchSklearnTestClassification(Evaluation[PatchSklearnClassificationDataModule]):
    """Perform a test of an already trained classification model.

    Args:
        config: The experiment configuration
        output: where to save resultss
        model_path: path to trained model from PatchSklearnClassification task.
        device: the device where to run the model (cuda or cpu). Defaults to 'cpu'.
    """

    def __init__(
        self,
        config: DictConfig,
        output: DictConfig,
        model_path: str,
        device: str = "cpu",
    ):
        super().__init__(config=config, model_path=model_path, device=device)
        self.output = output
        self._backbone: BaseEvaluationModel
        self._classifier: ClassifierMixin
        self.class_to_idx: dict[str, int]
        self.idx_to_class: dict[int, str]
        self.metadata: dict[str, Any] = {
            "test_confusion_matrix": None,
            "test_accuracy": None,
            "test_results": None,
            "test_labels": None,
        }
        self.class_to_skip: list[str] | None = []
        self.reconstruction_results: dict[str, Any]
        self.return_polygon: bool = True

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
        # Configure trainer
        self.trainer = self.config.trainer

        # prepare_data() must be explicitly called because there is no lightning training
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="test")

    @automatic_datamodule_batch_size(batch_size_attribute_name="batch_size")
    def test(self) -> None:
        """Run the test."""
        test_dataloader = self.datamodule.test_dataloader()

        self.class_to_skip = self.model_data["class_to_skip"] if hasattr(self.model_data, "class_to_skip") else None
        class_to_keep = None

        if self.class_to_skip is not None:
            class_to_keep = [x for x in self.datamodule.class_to_idx if x not in self.class_to_skip]
        _, pd_cm, accuracy, res, _ = self.trainer.test(
            test_dataloader=test_dataloader,
            idx_to_class=self.idx_to_class,
            predict_proba=True,
            class_to_keep=class_to_keep,
        )

        # save results
        self.metadata["test_confusion_matrix"] = pd_cm
        self.metadata["test_accuracy"] = accuracy
        self.metadata["test_results"] = res
        self.metadata["test_labels"] = [
            self.idx_to_class[i] if i != -1 else "N/A" for i in res["real_label"].unique().tolist()
        ]

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

    def generate_report(self) -> None:
        """Generate a report for the task."""
        log.info("Generating report!")
        os.makedirs(self.output.folder, exist_ok=True)

        c_matrix = self.metadata["test_confusion_matrix"]
        idx_to_class = {v: k for k, v in self.datamodule.class_to_idx.items()}

        datamodule: PatchSklearnClassificationDataModule = self.datamodule
        test_img_info = datamodule.info.test_files
        for img_info in test_img_info:
            if not os.path.isabs(img_info.image_path):
                img_info.image_path = os.path.join(datamodule.data_path, img_info.image_path)
            if img_info.mask_path is not None and not os.path.isabs(img_info.mask_path):
                img_info.mask_path = os.path.join(datamodule.data_path, img_info.mask_path)

        false_region_bad, false_region_good, true_region_bad, reconstructions = compute_patch_metrics(
            test_img_info=test_img_info,
            test_results=self.metadata["test_results"],
            patch_num_h=datamodule.info.patch_number[0] if datamodule.info.patch_number is not None else None,
            patch_num_w=datamodule.info.patch_number[1] if datamodule.info.patch_number is not None else None,
            patch_h=datamodule.info.patch_size[0] if datamodule.info.patch_size is not None else None,
            patch_w=datamodule.info.patch_size[1] if datamodule.info.patch_size is not None else None,
            overlap=datamodule.info.overlap,
            idx_to_class=idx_to_class,
            return_polygon=self.return_polygon,
            patch_reconstruction_method=self.output.reconstruction_method,
            annotated_good=datamodule.info.annotated_good,
        )

        self.reconstruction_results = {
            "false_region_bad": false_region_bad,
            "false_region_good": false_region_good,
            "true_region_bad": true_region_bad,
            "reconstructions": reconstructions,
            "reconstructions_type": "polygon" if self.return_polygon else "rle",
            "patch_reconstruction_method": self.output.reconstruction_method,
        }

        with open("reconstruction_results.json", "w") as f:
            json.dump(
                self.reconstruction_results,
                f,
                cls=RleEncoder,
            )

        if self.class_to_skip is not None:
            ignore_classes = [datamodule.class_to_idx[x] for x in self.class_to_skip]
        else:
            ignore_classes = None
        test_dataloader = self.datamodule.test_dataloader()
        save_classification_result(
            results=self.metadata["test_results"],
            output_folder=self.output.folder,
            confusion_matrix=c_matrix,
            accuracy=self.metadata["test_accuracy"],
            test_dataloader=test_dataloader,
            config=self.config,
            output=self.output,
            reconstructions=reconstructions,
            ignore_classes=ignore_classes,
        )

    def execute(self) -> None:
        """Execute the experiment and all the steps."""
        self.prepare()
        self.test()
        if self.output.report:
            self.generate_report()
        self.finalize()
