from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap
from matplotlib.lines import Line2D
from matplotlib.pyplot import Figure

from quadra.utils import utils

log = utils.get_logger(__name__)


def plot_patch_reconstruction(
    reconstruction: dict,
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    ignore_classes: list[int] | None = None,
    is_polygon: bool = True,
) -> Figure:
    """Helper function for plotting the patch reconstruction.

    Args:
        reconstruction: Dict following this structure
            {
                "file_path": str,
                "mask_path": str,
                "prediction": {
                    "label": str,
                    "points": [{"x": int, "y": int}]
                }
            } if is_polygon else
            {
                "file_path": str,
                "mask_path": str,
                "prediction": np.ndarray
            }
        idx_to_class: Dictionary mapping indices to label names
        class_to_idx: Dictionary mapping class names to indices
        ignore_classes: Eventually the classes to not plot
        is_polygon: Boolean indicating if the prediction is a polygon or a mask.

    Returns:
        Matplotlib plot showing predicted patch regions and eventually gt

    """
    cmap_name = "tab10"

    # 10 classes + good
    if len(idx_to_class.values()) > 11:
        cmap_name = "tab20"

    cmap = get_cmap(cmap_name)
    test_img = cv2.imread(reconstruction["image_path"])
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    gt_img = None

    if reconstruction["mask_path"] is not None and os.path.isfile(reconstruction["mask_path"]):
        gt_img = cv2.imread(reconstruction["mask_path"], 0)

    out = np.zeros((test_img.shape[0], test_img.shape[1]), dtype=np.uint8)

    if is_polygon:
        for _, region in enumerate(reconstruction["prediction"]):
            points = [[item["x"], item["y"]] for item in region["points"]]
            c_label = region["label"]

            out = cv2.drawContours(  # type: ignore[call-overload]
                out,
                np.array([points], np.int32),
                -1,
                class_to_idx[c_label],
                thickness=cv2.FILLED,
            )
    else:
        out = reconstruction["prediction"]

    fig = plot_patch_results(
        image=test_img,
        prediction_image=out,
        ground_truth_image=gt_img,
        plot_original=True,
        ignore_classes=ignore_classes,
        save_path=None,
        class_to_idx=class_to_idx,
        cmap=cmap,
    )

    return fig


def show_mask_on_image(image: np.ndarray, mask: np.ndarray):
    """Plot mask on top of the original image."""
    image = image.astype(np.float32) / 255
    mask = mask.astype(np.float32) / 255
    out = mask + image.astype(np.float32)
    out = out / np.max(out)
    return np.uint8(255 * out)


def create_rgb_mask(
    mask: np.ndarray,
    color_map: dict,
    ignore_classes: list[int] | None = None,
    ground_truth_mask: np.ndarray | None = None,
):
    """Convert index mask to RGB mask."""
    output_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for c in np.unique(mask):
        if ignore_classes is not None and c in ignore_classes:
            continue

        output_mask[mask == c] = color_map[str(c)]
    if ignore_classes is not None and ground_truth_mask is not None:
        output_mask[np.isin(ground_truth_mask, ignore_classes)] = [0, 0, 0]

    return output_mask


def plot_patch_results(
    image: np.ndarray,
    prediction_image: np.ndarray,
    ground_truth_image: np.ndarray | None,
    class_to_idx: dict[str, int],
    plot_original: bool = True,
    ignore_classes: list[int] | None = None,
    image_height: int = 10,
    save_path: str | None = None,
    cmap: Colormap | None = None,
) -> Figure:
    """Function used to plot the image predicted.

    Args:
        prediction_image: The prediction image
        image: The original image to plot
        ground_truth_image: The ground truth image
        class_to_idx: Dictionary mapping class names to indices
        plot_original: Boolean to plot the original image
        ignore_classes: The classes to ignore, default is 0
        image_height: The height of the output figure
        save_path: The path to save the figure
        cmap: The colormap to use. If None, tab20 is used

    Returns:
        The matplotlib figure
    """
    if ignore_classes is None:
        ignore_classes = [0]

    if cmap is None:
        cmap = get_cmap("tab20")

    image = image[0 : prediction_image.shape[0], 0 : prediction_image.shape[1], :]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    if ignore_classes is not None:
        class_to_idx = {k: v for k, v in class_to_idx.items() if v not in ignore_classes}

    class_idxs = list(class_to_idx.values())

    cmap = {str(c): tuple(int(i * 255) for i in cmap(c / len(class_idxs))[:-1]) for c in class_idxs}
    output_images = []
    titles = []

    if plot_original:
        output_images.append(image)
        titles.append("Original Image")

    if ground_truth_image is not None:
        ground_truth_image = ground_truth_image[0 : prediction_image.shape[0], 0 : prediction_image.shape[1]]
        ground_truth_mask = create_rgb_mask(ground_truth_image, cmap, ignore_classes=ignore_classes)
        output_images.append(ground_truth_mask)
        titles.append("Ground Truth Mask")

    prediction_mask = create_rgb_mask(
        prediction_image,
        cmap,
        ignore_classes=ignore_classes,
    )

    output_images.append(prediction_mask)
    titles.append("Prediction Mask")
    if ignore_classes is not None and ground_truth_image is not None:
        prediction_mask = create_rgb_mask(
            prediction_image, cmap, ignore_classes=ignore_classes, ground_truth_mask=ground_truth_image
        )

        ignored_classes_str = [idx_to_class[c] for c in ignore_classes]
        prediction_title = f"Prediction Mask \n (Ignoring Ground Truth Class: {ignored_classes_str})"
        output_images.append(prediction_mask)
        titles.append(prediction_title)

    fig, axs = plt.subplots(
        ncols=len(output_images),
        nrows=1,
        figsize=(len(output_images) * image_height, image_height),
        squeeze=False,
        facecolor="white",
    )

    for i, output_image in enumerate(output_images):
        axs[0, i].imshow(show_mask_on_image(image, output_image))
        axs[0, i].set_title(titles[i])
        axs[0, i].axis("off")

    custom_lines = [Line2D([0], [0], color=tuple(i / 255.0 for i in cmap[str(c)]), lw=4) for c in class_idxs]
    custom_labels = list(class_to_idx.keys())
    axs[0, -1].legend(custom_lines, custom_labels, loc="center left", bbox_to_anchor=(1.01, 0.81), borderaxespad=0)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    return fig
