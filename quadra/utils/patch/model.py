from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from label_studio_converter.brush import mask2rle
from omegaconf import DictConfig
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from quadra.utils import utils
from quadra.utils.patch.visualization import plot_patch_reconstruction
from quadra.utils.visualization import UnNormalize, plot_classification_results

log = utils.get_logger(__name__)


def save_classification_result(
    results: pd.DataFrame,
    output_folder: str,
    confusion_matrix: pd.DataFrame | None,
    accuracy: float,
    test_dataloader: DataLoader,
    reconstructions: list[dict],
    config: DictConfig,
    output: DictConfig,
    ignore_classes: list[int] | None = None,
):
    """Save classification results.

    Args:
        results: Dataframe containing the classification results
        output_folder: Folder where to save the results
        confusion_matrix: Confusion matrix
        accuracy: Accuracy of the model
        test_dataloader: Dataloader used for testing
        reconstructions: List of dictionaries containing polygons or masks
        config: Experiment configuration
        output: Output configuration
        ignore_classes: Eventual classes to ignore during reconstruction plot. Defaults to None.
    """
    # Save csv
    results.to_csv(os.path.join(output_folder, "test_results.csv"), index_label="index")

    if confusion_matrix is not None:
        # Save confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=np.array(confusion_matrix),
            display_labels=[x.replace("pred:", "") for x in confusion_matrix.columns.to_list()],
        )
        disp.plot(include_values=True, cmap=plt.cm.Greens, ax=None, colorbar=False, xticks_rotation=90)
        plt.title(f"Confusion Matrix (Accuracy: {(accuracy * 100):.2f}%)")
        plt.savefig(
            os.path.join(output_folder, "test_confusion_matrix.png"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
        plt.close()

    if output.example:
        if not hasattr(test_dataloader.dataset, "idx_to_class"):
            raise ValueError("The provided dataset does not have an attribute 'idx_to_class")

        idx_to_class = test_dataloader.dataset.idx_to_class

        # Get misclassified samples
        example_folder = os.path.join(output_folder, "example")
        if not os.path.isdir(example_folder):
            os.makedirs(example_folder)

        # Skip if no no ground truth is available
        if not all(results["real_label"] == -1):
            for v in np.unique([results["real_label"], results["pred_label"]]):
                if v == -1:
                    continue

                k = idx_to_class[v]

                if ignore_classes is not None and v in ignore_classes:
                    continue

                plot_classification_results(
                    test_dataloader.dataset,
                    unorm=UnNormalize(mean=config.transforms.mean, std=config.transforms.std),
                    pred_labels=results["pred_label"].to_numpy(),
                    test_labels=results["real_label"].to_numpy(),
                    class_name=k,
                    original_folder=example_folder,
                    idx_to_class=idx_to_class,
                    pred_class_to_plot=v,
                    what="con",
                    rows=output.get("rows", 3),
                    cols=output.get("cols", 2),
                    figsize=output.get("figsize", (20, 20)),
                )

                plot_classification_results(
                    test_dataloader.dataset,
                    unorm=UnNormalize(mean=config.transforms.mean, std=config.transforms.std),
                    pred_labels=results["pred_label"].to_numpy(),
                    test_labels=results["real_label"].to_numpy(),
                    class_name=k,
                    original_folder=example_folder,
                    idx_to_class=idx_to_class,
                    pred_class_to_plot=v,
                    what="dis",
                    rows=output.get("rows", 3),
                    cols=output.get("cols", 2),
                    figsize=output.get("figsize", (20, 20)),
                )

        for counter, reconstruction in enumerate(reconstructions):
            is_polygon = True
            if isinstance(reconstruction["prediction"], np.ndarray):
                is_polygon = False

            if is_polygon:
                if len(reconstruction["prediction"]) == 0:
                    continue
            elif reconstruction["prediction"].sum() == 0:
                continue

            if counter > 5:
                break

            to_plot = plot_patch_reconstruction(
                reconstruction,
                idx_to_class,
                class_to_idx=test_dataloader.dataset.class_to_idx,  # type: ignore[attr-defined]
                ignore_classes=ignore_classes,
                is_polygon=is_polygon,
            )

            if to_plot:
                output_name = f"reconstruction_{os.path.splitext(os.path.basename(reconstruction['file_name']))[0]}.png"
                plt.savefig(os.path.join(example_folder, output_name), bbox_inches="tight", pad_inches=0)

            plt.close()


class RleEncoder(json.JSONEncoder):
    """Custom encoder to convert numpy arrays to RLE."""

    def default(self, o: Any):
        """Customize standard encoder behaviour to convert numpy arrays to RLE."""
        if isinstance(o, np.ndarray):
            return mask2rle(o)
        return json.JSONEncoder.default(self, o)
