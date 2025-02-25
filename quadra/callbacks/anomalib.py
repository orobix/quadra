from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from anomalib.models.components.base import AnomalyModule
from anomalib.post_processing import (
    add_anomalous_label,
    add_normal_label,
    compute_mask,
    superimpose_anomaly_map,
)
from anomalib.pre_processing.transforms import Denormalize
from anomalib.utils.loggers import AnomalibWandbLogger
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from skimage.segmentation import mark_boundaries
from tqdm import tqdm

from quadra.utils.anomaly import MapOrValue


class Visualizer:
    """Anomaly Visualization.

    The visualizer object is responsible for collating all the images passed to it into a single image. This can then
    either be logged by accessing the `figure` attribute or can be saved directly by calling `save()` method.

    Example:
        >>> visualizer = Visualizer()
        >>> visualizer.add_image(image=image, title="Image")
        >>> visualizer.close()
    """

    def __init__(self) -> None:
        self.images: list[dict] = []

        self.figure: matplotlib.figure.Figure
        self.axis: np.ndarray

    def add_image(self, image: np.ndarray, title: str, color_map: str | None = None):
        """Add image to figure.

        Args:
          image: Image which should be added to the figure.
          title: Image title shown on the plot.
          color_map: Name of matplotlib color map used to map scalar data to colours. Defaults to None.
        """
        image_data = {"image": image, "title": title, "color_map": color_map}
        self.images.append(image_data)

    def generate(self):
        """Generate the image."""
        default_plt_backend = plt.get_backend()
        plt.switch_backend("Agg")
        num_cols = len(self.images)
        figure_size = (num_cols * 3, 3)
        self.figure, self.axis = plt.subplots(1, num_cols, figsize=figure_size)
        self.figure.subplots_adjust(right=0.9)

        axes = self.axis if len(self.images) > 1 else [self.axis]
        for axis, image_dict in zip(axes, self.images, strict=False):
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)
            axis.imshow(image_dict["image"], image_dict["color_map"], vmin=0, vmax=255)
            axis.title.set_text(image_dict["title"])
        plt.switch_backend(default_plt_backend)

    def show(self):
        """Show image on a matplotlib figure."""
        self.figure.show()

    def save(self, filename: Path):
        """Save image.

        Args:
          filename: Filename to save image
        """
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(filename, dpi=100)

    def close(self):
        """Close figure."""
        plt.close(self.figure)


# TODO: This is a lot different from the 0.3.7 anomalib one
class VisualizerCallback(Callback):
    """Callback that visualizes the inference results of a model.

    The callback generates a figure showing the original image, the ground truth segmentation mask,
    the predicted error heat map, and the predicted segmentation mask.
    To save the images to the filesystem, add the 'local' keyword to the `project.log_images_to` parameter in the
    config.yaml file.

    Args:
        task: either 'segmentation' or 'classification'
        output_path: location where the images will be saved.
        inputs_are_normalized: whether the input images are normalized (like when using MinMax or Treshold callback).
        threshold_type: Either 'pixel' or 'image'. If 'pixel', the threshold is computed on the pixel-level.
        disable: whether to disable the callback.
        plot_only_wrong: whether to plot only the images that are not correctly predicted.
        plot_raw_outputs: Saves the raw images of the segmentation and heatmap output.
    """

    def __init__(
        self,
        task: str = "segmentation",
        output_path: str = "anomaly_output",
        inputs_are_normalized: bool = True,
        threshold_type: str = "pixel",
        disable: bool = False,
        plot_only_wrong: bool = False,
        plot_raw_outputs: bool = False,
    ) -> None:
        self.inputs_are_normalized = inputs_are_normalized
        self.output_path = output_path
        self.threshold_type = threshold_type
        self.disable = disable
        self.task = task
        self.plot_only_wrong = plot_only_wrong
        self.plot_raw_outputs = plot_raw_outputs

    def _add_images(self, visualizer: Visualizer, filename: Path, output_label_folder: str):
        """Save image to logger/local storage.

        Saves the image in `visualizer.figure` to the respective loggers and local storage if specified in
        `log_images_to` in `config.yaml` of the models.

        Args:
            visualizer: Visualizer object from which the `figure` is saved/logged.
            filename: Path of the input image. This name is used as name for the generated image.
            output_label_folder: ok if the image is correctly predicted or wrong if it is not
        """
        visualizer.save(
            Path(self.output_path)
            / "images"
            / output_label_folder
            / filename.parent.name
            / Path(filename.stem + ".png")
        )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log images at the end of every batch.

        Args:
            trainer: Pytorch lightning trainer object (unused).
            pl_module: Lightning modules derived from BaseAnomalyLightning object as
                currently only they support logging images.
            outputs: Outputs of the current test step.
            batch: Input batch of the current test step (unused).
            batch_idx: Index of the current test batch (unused).
            dataloader_idx: Index of the dataloader that yielded the current batch (unused).
        """
        if self.disable:
            return

        assert outputs is not None and isinstance(outputs, dict)

        if any(x not in outputs for x in ["image_path", "image", "mask", "anomaly_maps", "label"]):
            # I'm probably in the classification scenario so I can't use the visualizer
            return

        if self.threshold_type == "pixel":
            if hasattr(pl_module.pixel_metrics.F1Score, "threshold"):
                threshold = pl_module.pixel_metrics.F1Score.threshold
            else:
                raise AttributeError("Metric has no threshold attribute")
        elif hasattr(pl_module.image_metrics.F1Score, "threshold"):
            threshold = pl_module.image_metrics.F1Score.threshold
        else:
            raise AttributeError("Metric has no threshold attribute")

        for (
            filename,
            image,
            true_mask,
            anomaly_map,
            gt_label,
            pred_label,
            anomaly_score,
        ) in tqdm(
            zip(
                outputs["image_path"],
                outputs["image"],
                outputs["mask"],
                outputs["anomaly_maps"],
                outputs["label"],
                outputs["pred_labels"],
                outputs["pred_scores"],
                strict=False,
            )
        ):
            denormalized_image = Denormalize()(image.cpu())
            current_true_mask = true_mask.cpu().numpy()
            current_anomaly_map = anomaly_map.cpu().numpy()
            # Normalize the map and rescale it to 0-1 range
            # In this case we are saying that the anomaly map is in the range [normalized_th - 50, normalized_th + 50]
            # This allow to have a stronger color for the anomalies and a lighter one for really normal regions
            # It's also independent from the max or min anomaly score!
            normalized_map: MapOrValue = (current_anomaly_map - (threshold - 50)) / 100
            normalized_map = np.clip(normalized_map, 0, 1)

            output_label_folder = "ok" if pred_label == gt_label else "wrong"

            if self.plot_only_wrong and output_label_folder == "ok":
                continue

            heatmap = superimpose_anomaly_map(
                normalized_map, denormalized_image, normalize=not self.inputs_are_normalized
            )

            if isinstance(threshold, float):
                pred_mask = compute_mask(current_anomaly_map, threshold)
            else:
                raise TypeError("Threshold should be float")
            vis_img = mark_boundaries(denormalized_image, pred_mask, color=(1, 0, 0), mode="thick")
            visualizer = Visualizer()

            if self.task == "segmentation":
                visualizer.add_image(image=denormalized_image, title="Image")
                if "mask" in outputs:
                    current_true_mask = current_true_mask * 255
                    visualizer.add_image(image=current_true_mask, color_map="gray", title="Ground Truth")
                visualizer.add_image(image=heatmap, title="Predicted Heat Map")
                visualizer.add_image(image=pred_mask, color_map="gray", title="Predicted Mask")
                visualizer.add_image(image=vis_img, title="Segmentation Result")
            elif self.task == "classification":
                gt_im = add_anomalous_label(denormalized_image) if gt_label else add_normal_label(denormalized_image)
                visualizer.add_image(gt_im, title="Image/True label")
                if anomaly_score >= threshold:
                    image_classified = add_anomalous_label(heatmap, anomaly_score)
                else:
                    image_classified = add_normal_label(heatmap, 1 - anomaly_score)
                visualizer.add_image(image=image_classified, title="Prediction")

            visualizer.generate()
            visualizer.figure.suptitle(
                f"F1 threshold: {threshold}, Mask_max: {current_anomaly_map.max():.3f}, "
                f"Anomaly_score: {anomaly_score:.3f}"
            )
            path_filename = Path(filename)
            self._add_images(visualizer, path_filename, output_label_folder)
            visualizer.close()

            if self.plot_raw_outputs:
                for raw_output, raw_name in zip([heatmap, vis_img], ["heatmap", "segmentation"], strict=False):
                    current_raw_output = raw_output
                    if raw_name == "segmentation":
                        current_raw_output = (raw_output * 255).astype(np.uint8)
                    current_raw_output = cv2.cvtColor(current_raw_output, cv2.COLOR_RGB2BGR)
                    raw_filename = (
                        Path(self.output_path)
                        / "images"
                        / output_label_folder
                        / path_filename.parent.name
                        / "raw_outputs"
                        / Path(path_filename.stem + f"_{raw_name}.png")
                    )
                    raw_filename.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(raw_filename), current_raw_output)

    def on_test_end(self, _trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Sync logs.

        Currently only ``AnomalibWandbLogger`` is called from this method. This is because logging as a single batch
        ensures that all images appear as part of the same step.

        Args:
            _trainer: Pytorch Lightning trainer (unused)
            pl_module: Anomaly module
        """
        if self.disable:
            return

        if pl_module.logger is not None and isinstance(pl_module.logger, AnomalibWandbLogger):
            pl_module.logger.save()
