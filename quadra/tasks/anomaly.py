from __future__ import annotations

import csv
import glob
import json
import os
from collections import Counter
from typing import Any, Generic, Literal, TypeVar, cast

import cv2
import hydra
import numpy as np
import torch
from anomalib.models.components.base import AnomalyModule
from anomalib.post_processing import anomaly_map_to_color_map
from anomalib.utils import plot_cumulative_histogram
from anomalib.utils.callbacks.min_max_normalization import MinMaxNormalizationCallback
from anomalib.utils.metrics.optimal_f1 import OptimalF1
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from tqdm import tqdm

from quadra.callbacks.mlflow import get_mlflow_logger
from quadra.datamodules import AnomalyDataModule
from quadra.modules.base import ModelSignatureWrapper
from quadra.tasks.base import Evaluation, LightningTask
from quadra.utils import utils
from quadra.utils.anomaly import MapOrValue, ThresholdNormalizationCallback, normalize_anomaly_score
from quadra.utils.classification import get_results
from quadra.utils.evaluation import automatic_datamodule_batch_size
from quadra.utils.export import export_model

log = utils.get_logger(__name__)

AnomalyDataModuleT = TypeVar("AnomalyDataModuleT", bound=AnomalyDataModule)


class AnomalibDetection(Generic[AnomalyDataModuleT], LightningTask[AnomalyDataModuleT]):
    """Anomaly Detection Task.

    Args:
        config: The experiment configuration
        module_function: The function that instantiates the module and model
        checkpoint_path: The path to the checkpoint to load the model from.
            Defaults to None.
        run_test: Whether to run the test after training. Defaults to False.
        report: Whether to report the results. Defaults to False.
    """

    def __init__(
        self,
        config: DictConfig,
        module_function: DictConfig,
        checkpoint_path: str | None = None,
        run_test: bool = True,
        report: bool = True,
    ):
        super().__init__(
            config=config,
            checkpoint_path=checkpoint_path,
            run_test=run_test,
            report=report,
        )
        self._module: AnomalyModule
        self.module_function = module_function
        self.export_folder = "deployment_model"
        self.report_path = ""
        self.test_results: list[dict] | None = None

    @property
    def module(self) -> AnomalyModule:
        """Get the module."""
        return self._module

    @module.setter
    def module(self, module_config):
        """Set the module."""
        if hasattr(self.config.model.model, "input_size"):
            transform_height = self.config.transforms.input_height
            transform_width = self.config.transforms.input_width
            original_model_height, original_model_width = self.config.model.model.input_size

            if transform_height != original_model_height or transform_width != original_model_width:
                log.warning(
                    "Model input size %dx%d "
                    "does not match the transform size %dx%d. "
                    "The model input size will be updated to match the transform size.",
                    original_model_height,
                    original_model_width,
                    transform_height,
                    transform_width,
                )
            self.config.model.model.input_size = [transform_height, transform_width]

        _module = cast(
            AnomalyModule,
            hydra.utils.instantiate(
                self.module_function,
                module_config,
            ),
        )

        self._module = _module

    def prepare(self) -> None:
        """Prepare the task."""
        super().prepare()
        self.module = self.config.model
        self.module.model = ModelSignatureWrapper(self.module.model)

    def export(self) -> None:
        """Export model for production."""
        if self.config.trainer.get("fast_dev_run"):
            log.warning("Skipping export since fast_dev_run is enabled")
            return

        model = self.module.model

        input_shapes = self.config.export.input_shapes

        half_precision = "16" in self.trainer.precision

        model_json, export_paths = export_model(
            config=self.config,
            model=model,
            export_folder=self.export_folder,
            half_precision=half_precision,
            input_shapes=input_shapes,
            idx_to_class={0: "good", 1: "defect"},
        )

        if len(export_paths) == 0:
            return

        model_json["image_threshold"] = np.round(self.module.image_threshold.value.item(), 3)
        model_json["pixel_threshold"] = np.round(self.module.pixel_threshold.value.item(), 3)
        model_json["anomaly_method"] = self.config.model.model.name

        with open(os.path.join(self.export_folder, "model.json"), "w") as f:
            json.dump(model_json, f, cls=utils.HydraEncoder)

    def test(self) -> Any:
        """Lightning test."""
        self.test_results = super().test()
        return self.test_results

    def _generate_report(self) -> None:
        """Generate a report for the task."""
        if len(self.report_path) > 0:
            os.makedirs(self.report_path, exist_ok=True)

        # Save json with test results
        if self.test_results is not None:
            with open(os.path.join(self.report_path, "test_results.json"), "w") as f:
                json.dump(self.test_results[0], f)

        all_output = cast(
            list[dict], self.trainer.predict(model=self.module, dataloaders=self.datamodule.test_dataloader())
        )
        all_output_flatten: dict[str, torch.Tensor | list] = {}

        for key in all_output[0]:
            if isinstance(all_output[0][key], torch.Tensor):
                tensor_gatherer = torch.cat([x[key] for x in all_output])
                all_output_flatten[key] = tensor_gatherer
            else:
                list_gatherer = []
                for x in all_output:
                    list_gatherer.extend(x[key])
                all_output_flatten[key] = list_gatherer

        image_paths = all_output_flatten["image_path"]
        named_labels = [x.split("/")[-2] for x in all_output_flatten["image_path"]]

        class_to_idx = {"good": 0}
        idx = 1
        for cls in set(named_labels):
            if cls == "good":
                continue

            class_to_idx[cls] = idx
            idx += 1

        class_to_idx["false_defect"] = idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        gt_labels = [class_to_idx[x] for x in named_labels]
        pred_labels = []
        for i, _ in enumerate(named_labels):
            pred_label = all_output_flatten["pred_labels"][i].item()

            if pred_label == 0:
                pred_labels.append(0)
            elif pred_label == 1 and gt_labels[i] == 0:
                if idx > 2:
                    pred_labels.append(class_to_idx["false_defect"])
                else:
                    pred_labels.append(1)
            else:
                pred_labels.append(class_to_idx[named_labels[i]])

        if class_to_idx["false_defect"] not in pred_labels:
            # If there are no false defects remove the label from the confusion matrix
            class_to_idx.pop("false_defect")

        anomaly_scores = all_output_flatten["pred_scores"]

        exportable_anomaly_scores: list[Any] | np.ndarray
        if isinstance(anomaly_scores, torch.Tensor):
            exportable_anomaly_scores = anomaly_scores.cpu().numpy()
        else:
            exportable_anomaly_scores = anomaly_scores

        # Zip the lists together to create rows for the CSV file
        rows = zip(image_paths, pred_labels, gt_labels, exportable_anomaly_scores, strict=False)
        # Specify the CSV file name
        csv_file = "test_predictions.csv"
        # Write the data to the CSV file
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write the header if needed
            writer.writerow(["image_path", "predicted_label", "ground_truth_label", "predicted_score"])
            # Write the rows
            writer.writerows(rows)

        log.info("CSV file %s has been created.", csv_file)

        if not isinstance(anomaly_scores, torch.Tensor):
            raise ValueError("Anomaly scores must be a tensor")

        good_scores = anomaly_scores[np.where(all_output_flatten["label"] == 0)]
        defect_scores = anomaly_scores[np.where(all_output_flatten["label"] == 1)]

        # Lightning has a callback attribute but is not inside the __init__ so mypy complains
        if any(
            isinstance(x, MinMaxNormalizationCallback)
            for x in self.trainer.callbacks  # type: ignore[attr-defined]
        ):
            threshold = torch.tensor(0.5)
        elif any(
            isinstance(x, ThresholdNormalizationCallback)
            for x in self.trainer.callbacks  # type: ignore[attr-defined]
        ):
            threshold = torch.tensor(100.0)
        else:
            threshold = self.module.image_metrics.F1Score.threshold  # type: ignore[union-attr,assignment]

        # The output of the prediction is a normalized score so the cumulative histogram is displayed with the
        # normalized scores
        plot_cumulative_histogram(
            good_scores.cpu().numpy(), defect_scores.cpu().numpy(), threshold.item(), self.report_path
        )

        _, pd_cm, _ = get_results(np.array(gt_labels), np.array(pred_labels), idx_to_class)
        np_cm = np.array(pd_cm)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=np_cm,
            display_labels=class_to_idx.keys(),
        )
        disp.plot(include_values=True, cmap=plt.cm.Greens, ax=None, colorbar=False, xticks_rotation=90)
        plt.title("Confusion Matrix")
        plt.savefig(
            os.path.join(self.report_path, "test_confusion_matrix.png"), bbox_inches="tight", pad_inches=0, dpi=300
        )
        plt.close()

        avg_score_dict = {k: 0.0 for k in set(named_labels)}

        for i, item in enumerate(named_labels):
            avg_score_dict[item] += all_output_flatten["pred_scores"][i].item()

        counter = Counter(named_labels)
        avg_score_dict = {k: v / counter[k] for k, v in avg_score_dict.items()}
        avg_score_dict = dict(sorted(avg_score_dict.items(), key=lambda q: q[1]))

        with open(os.path.join(self.report_path, "avg_score_by_label.csv"), "w") as f:
            f.write("label,avg_anomaly_score\n")
            for k, v in avg_score_dict.items():
                f.write(f"{k},{v:.3f}\n")

    def generate_report(self):
        """Generate a report for the task and try to upload artifacts."""
        self._generate_report()
        self._upload_artifacts()

    def _upload_artifacts(self):
        """If MLflow is available upload artifacts to the artifact repository."""
        mflow_logger = get_mlflow_logger(trainer=self.trainer)
        tensorboard_logger = utils.get_tensorboard_logger(trainer=self.trainer)

        if mflow_logger is not None and self.config.core.get("upload_artifacts"):
            mflow_logger.experiment.log_artifact(run_id=mflow_logger.run_id, local_path="test_confusion_matrix.png")
            mflow_logger.experiment.log_artifact(run_id=mflow_logger.run_id, local_path="avg_score_by_label.csv")

            if "visualizer" in self.config.callbacks:
                artifacts = glob.glob(os.path.join(self.config.callbacks.visualizer.output_path, "**", "*"))
                for a in artifacts:
                    mflow_logger.experiment.log_artifact(
                        run_id=mflow_logger.run_id, local_path=a, artifact_path="anomaly_output"
                    )

        if tensorboard_logger is not None and self.config.core.get("upload_artifacts"):
            artifacts = []
            artifacts.append("test_confusion_matrix.png")
            artifacts.append("avg_score_by_label.csv")

            if "visualizer" in self.config.callbacks:
                artifacts.extend(
                    glob.glob(os.path.join(self.config.callbacks.visualizer.output_path, "**/*"), recursive=True)
                )

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

    def execute(self):
        """Execute the experiment and all the steps."""
        self.prepare()
        self.train()
        # When training in fp16 mixed precision, export function casts model weights from fp32 to fp16,
        # for this reason, predictions logits could slightly change and predictions could be inconsistent between
        # test and generated report.
        # Performing export before test allows to have consistent results in test metrics and generated report.
        if self.config.export is not None and len(self.config.export.types) > 0:
            self.export()
        if self.run_test:
            self.test()
        if self.report:
            self.generate_report()
        self.finalize()


class AnomalibEvaluation(Evaluation[AnomalyDataModule]):
    """Evaluation task for Anomalib.

    Args:
        config: Task configuration
        model_path: Path to the model folder that contains an exported model
        use_training_threshold: Whether to use the training threshold for the evaluation or use the one that
            maximizes the F1 score on the test set.
        device: Device to use for evaluation. If None, the device is automatically determined.
    """

    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        use_training_threshold: bool = False,
        device: str | None = None,
        training_threshold_type: Literal["image", "pixel"] | None = None,
    ):
        super().__init__(config=config, model_path=model_path, device=device)

        self.use_training_threshold = use_training_threshold

        if training_threshold_type is not None and training_threshold_type not in ["image", "pixel"]:
            raise ValueError("Training threshold type must be either image or pixel")

        if training_threshold_type is None and use_training_threshold:
            log.warning("Using training threshold but no training threshold type is provided, defaulting to image")
            training_threshold_type = "image"

        self.training_threshold_type = training_threshold_type

    def prepare(self) -> None:
        """Prepare the evaluation."""
        super().prepare()
        self.datamodule = self.config.datamodule
        # prepare_data() must be explicitly called because there is no lightning training
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="test")

    @automatic_datamodule_batch_size(batch_size_attribute_name="test_batch_size")
    def test(self) -> None:
        """Perform test."""
        log.info("Running test")
        test_dataloader = self.datamodule.test_dataloader()

        optimal_f1 = OptimalF1(num_classes=None, pos_label=1)  # type: ignore[arg-type]

        anomaly_scores = []
        anomaly_maps = []
        image_labels = []
        image_paths = []

        with torch.no_grad():
            for batch_item in tqdm(test_dataloader):
                batch_images = batch_item["image"]
                batch_labels = batch_item["label"]
                image_labels.extend(batch_labels.tolist())
                image_paths.extend(batch_item["image_path"])
                batch_images = batch_images.to(device=self.device, dtype=self.deployment_model.model_dtype)
                if self.model_data.get("anomaly_method") == "efficientad":
                    model_output = self.deployment_model(batch_images, None)
                else:
                    model_output = self.deployment_model(batch_images)
                anomaly_map, anomaly_score = model_output[0], model_output[1]
                anomaly_map = anomaly_map.cpu()
                anomaly_score = anomaly_score.cpu()
                known_labels = torch.where(batch_labels != -1)[0]
                if len(known_labels) > 0:
                    # Skip computing F1 score for images without gt
                    optimal_f1.update(anomaly_score[known_labels], batch_labels[known_labels])
                anomaly_scores.append(anomaly_score)
                anomaly_maps.append(anomaly_map)

        anomaly_scores = torch.cat(anomaly_scores)
        anomaly_maps = torch.cat(anomaly_maps)

        if any(x != -1 for x in image_labels):
            if self.use_training_threshold:
                _image_labels = torch.tensor(image_labels)
                threshold = torch.tensor(float(self.model_data[f"{self.training_threshold_type}_threshold"]))
                known_labels = torch.where(_image_labels != -1)[0]

                _image_labels = _image_labels[known_labels]
                _anomaly_scores = anomaly_scores[known_labels]

                pred_labels = (_anomaly_scores >= threshold).long()

                optimal_f1_score = torch.tensor(f1_score(_image_labels, pred_labels))
            else:
                optimal_f1_score = optimal_f1.compute()
                threshold = optimal_f1.threshold
        else:
            log.warning("No ground truth available during evaluation, use training image threshold for reporting")
            optimal_f1_score = torch.tensor(0)
            threshold = torch.tensor(float(self.model_data["image_threshold"]))

        log.info("Computed F1 score: %s", optimal_f1_score.item())
        self.metadata["anomaly_scores"] = anomaly_scores
        self.metadata["anomaly_maps"] = anomaly_maps
        self.metadata["image_labels"] = image_labels
        self.metadata["image_paths"] = image_paths
        self.metadata["threshold"] = threshold.item()
        self.metadata["optimal_f1"] = optimal_f1_score.item()

    def generate_report(self) -> None:
        """Generate report."""
        log.info("Generating report")
        if len(self.report_path) > 0:
            os.makedirs(self.report_path, exist_ok=True)

        # TODO: We currently don't use anomaly for segmentation, so the pixel threshold handling is not properly
        # implemented and we produce as output only a single threshold.
        training_threshold = self.model_data[f"{self.training_threshold_type}_threshold"]
        optimal_threshold = self.metadata["threshold"]

        os.makedirs(os.path.join(self.report_path, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(self.report_path, "heatmaps"), exist_ok=True)

        anomaly_scores = self.metadata["anomaly_scores"].cpu().numpy()

        # The reason I have to expand dims and cast the optimal threshold to anomaly_scores dtype is because
        # of internal roundings performed differently by numpy and python
        # Particularly the normalized_optimal_threshold computed directly using float values might be higher than the
        # actual value obtained by the anomaly_scores
        normalized_optimal_threshold = cast(
            np.ndarray,
            normalize_anomaly_score(
                np.expand_dims(np.array(optimal_threshold, dtype=anomaly_scores.dtype), -1), training_threshold
            ),
        ).item()

        anomaly_scores = normalize_anomaly_score(anomaly_scores, training_threshold)

        if not isinstance(anomaly_scores, np.ndarray):
            raise ValueError("Anomaly scores must be a numpy array")

        good_scores = anomaly_scores[np.where(np.array(self.metadata["image_labels"]) == 0)]
        defect_scores = anomaly_scores[np.where(np.array(self.metadata["image_labels"]) == 1)]

        count_overlapping_scores = 0

        if len(good_scores) != 0 and len(defect_scores) != 0 and defect_scores.min() <= good_scores.max():
            count_overlapping_scores = len(
                np.where((anomaly_scores >= defect_scores.min()) & (anomaly_scores <= good_scores.max()))[0]
            )

        plot_cumulative_histogram(good_scores, defect_scores, normalized_optimal_threshold, self.report_path)

        json_output = {
            "observations": [],
            "threshold": np.round(normalized_optimal_threshold, 3),
            "unnormalized_threshold": np.round(optimal_threshold, 3),
            "f1_score": np.round(self.metadata["optimal_f1"], 3),
            "metrics": {
                "overlapping_scores": count_overlapping_scores,
            },
        }

        tg, fb, fg, tb = 0, 0, 0, 0

        mask_area = None
        crop_area = None

        if hasattr(self.datamodule, "valid_area_mask") and self.datamodule.valid_area_mask is not None:
            mask_area = cv2.imread(self.datamodule.valid_area_mask, 0)
            mask_area = (mask_area > 0).astype(np.uint8)

        if hasattr(self.datamodule, "crop_area") and self.datamodule.crop_area is not None:
            crop_area = self.datamodule.crop_area

        anomaly_maps = normalize_anomaly_score(self.metadata["anomaly_maps"], training_threshold)

        if not isinstance(anomaly_maps, torch.Tensor):
            raise ValueError("Anomaly maps must be a tensor")

        for img_path, gt_label, anomaly_score, anomaly_map in tqdm(
            zip(
                self.metadata["image_paths"],
                self.metadata["image_labels"],
                anomaly_scores,
                anomaly_maps,
                strict=False,
            ),
            total=len(self.metadata["image_paths"]),
        ):
            img = cv2.imread(img_path, 0)
            if mask_area is not None:
                img = img * mask_area

            if crop_area is not None:
                img = img[crop_area[1] : crop_area[3], crop_area[0] : crop_area[2]]

            output_mask = (anomaly_map >= normalized_optimal_threshold).cpu().numpy().squeeze().astype(np.uint8)
            output_mask_label = os.path.basename(os.path.dirname(img_path))
            output_mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
            pred_label = int(anomaly_score >= normalized_optimal_threshold)

            json_output["observations"].append(
                {
                    "image_path": os.path.dirname(img_path),
                    "file_name": os.path.basename(img_path),
                    "expectation": gt_label if gt_label != -1 else "",
                    "prediction": pred_label,
                    "prediction_mask": os.path.join("predictions", output_mask_label, output_mask_name),
                    "prediction_heatmap": os.path.join("heatmaps", output_mask_label, output_mask_name),
                    "is_correct": pred_label == gt_label if gt_label != -1 else True,
                    "anomaly_score": f"{anomaly_score.item():.3f}",
                }
            )

            if gt_label == 0 and pred_label == 0:
                tg += 1
            elif gt_label == 0 and pred_label == 1:
                fb += 1
            elif gt_label == 1 and pred_label == 0:
                fg += 1
            elif gt_label == 1 and pred_label == 1:
                tb += 1

            output_mask = output_mask * 255
            output_mask = cv2.resize(output_mask, (img.shape[1], img.shape[0]))
            output_prediction_folder = os.path.join(self.report_path, "predictions", output_mask_label)
            os.makedirs(output_prediction_folder, exist_ok=True)
            cv2.imwrite(os.path.join(output_prediction_folder, output_mask_name), output_mask)

            # Normalize the map and rescale it to 0-1 range
            # In this case we are saying that the anomaly map is in the range [normalized_th - 50, normalized_th + 50]
            # This allow to have a stronger color for the anomalies and a lighter one for really normal regions
            # It's also independent from the max or min anomaly score!
            normalized_map: MapOrValue = (anomaly_map - (normalized_optimal_threshold - 50)) / 100

            if isinstance(normalized_map, torch.Tensor):
                normalized_map = normalized_map.cpu().numpy().squeeze()

            normalized_map = np.clip(normalized_map, 0, 1)
            output_heatmap = anomaly_map_to_color_map(normalized_map, normalize=False)
            output_heatmap = cv2.resize(output_heatmap, (img.shape[1], img.shape[0]))

            output_heatmap_folder = os.path.join(self.report_path, "heatmaps", output_mask_label)
            os.makedirs(output_heatmap_folder, exist_ok=True)

            cv2.imwrite(
                os.path.join(output_heatmap_folder, output_mask_name),
                cv2.cvtColor(output_heatmap, cv2.COLOR_RGB2BGR),
            )

        json_output["metrics"]["confusion_matrix"] = {
            "class_labels": ["normal", "anomaly"],
            "matrix": [
                [tg, fb],
                [fg, tb],
            ],
        }

        with open(os.path.join(self.report_path, "anomaly_test_output.json"), "w") as f:
            json.dump(json_output, f)

    def execute(self) -> None:
        """Execute the evaluation."""
        self.prepare()
        self.test()
        self.generate_report()
        self.finalize()
