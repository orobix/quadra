from __future__ import annotations

import copy
import os
import random
from collections.abc import Callable, Iterable
from typing import Any

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.augmentations.transforms import Normalize
from albumentations.core.composition import TransformsSeqType
from albumentations.core.transforms_interface import NoOp
from albumentations.pytorch.transforms import ToTensorV2
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.pyplot import get_cmap
from mpl_toolkits.axes_grid1 import ImageGrid
from omegaconf import DictConfig, ListConfig
from pytorch_grad_cam.utils.image import show_cam_on_image

from quadra.utils import utils

log = utils.get_logger(__name__)


class UnNormalize:
    """Unnormalize a tensor image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor, make_copy=True) -> torch.Tensor:
        """Call function to unnormalize a tensor image with mean and standard deviation.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            make_copy (bool): whether to apply normalization to a copied tensor
        Returns:
            Tensor: Normalized image.
        """
        if make_copy:
            new_t = tensor.detach().clone()
        else:
            new_t = tensor
        for t, m, s in zip(new_t, self.mean, self.std, strict=False):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return new_t


def create_grid_figure(
    images: Iterable[Iterable[np.ndarray]],
    nrows: int,
    ncols: int,
    file_path: str,
    bounds: list[tuple[float, float]],
    row_names: Iterable[str] | None = None,
    fig_size: tuple[int, int] = (12, 8),
):
    """Create a grid figure with images.

    Args:
        images: List of images to plot.
        nrows: Number of rows in the grid.
        ncols: Number of columns in the grid.
        file_path: Path to save the figure.
        row_names: Row names. Defaults to None.
        fig_size: Figure size. Defaults to (12, 8).
        bounds: Bounds for the images. Defaults to None.
    """
    default_plt_backend = plt.get_backend()
    plt.switch_backend("Agg")
    _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size, squeeze=False)
    for i, row in enumerate(images):
        for j, image in enumerate(row):
            image_to_plot = image[0] if len(image.shape) == 3 and image.shape[0] == 1 else image
            ax[i][j].imshow(image_to_plot, vmin=bounds[i][0], vmax=bounds[i][1])
            ax[i][j].get_xaxis().set_ticks([])
            ax[i][j].get_yaxis().set_ticks([])
    if row_names is not None:
        for ax, name in zip(ax[:, 0], row_names, strict=False):  # noqa: B020
            ax.set_ylabel(name, rotation=90)

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", dpi=300, facecolor="white", transparent=False)
    plt.close()
    plt.switch_backend(default_plt_backend)


def create_visualization_dataset(dataset: torch.utils.data.Dataset):
    """Create a visualization dataset by updating transforms."""

    def convert_transforms(transforms: Any):
        """Handle different types of transforms."""
        if isinstance(transforms, albumentations.BaseCompose):
            transforms.transforms = convert_transforms(transforms.transforms)
        if isinstance(transforms, list | ListConfig | TransformsSeqType):
            transforms = [convert_transforms(t) for t in transforms]
        if isinstance(transforms, dict | DictConfig):
            for tname, t in transforms.items():
                transforms[tname] = convert_transforms(t)
        if isinstance(transforms, Normalize | ToTensorV2):
            return NoOp(p=1)
        return transforms

    new_dataset = copy.deepcopy(dataset)
    # TODO: Create dataset class that has a transform attribut, we can then use isinstance
    if isinstance(dataset, torch.utils.data.Dataset):
        transform = copy.deepcopy(dataset.transform)  # type: ignore[attr-defined]
        if transform is not None:
            new_transforms = convert_transforms(transform)
            new_dataset.transform = new_transforms  # type: ignore[attr-defined]
        else:
            raise ValueError(f"The dataset transform {type(transform)} is not supported")
    else:
        raise ValueError(f"The dataset type {dataset} is not supported")
    return new_dataset


def show_mask_on_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Show a mask on an image.

    Args:
        image (np.ndarray): The image.
        mask (np.ndarray): The mask.

    Returns:
        np.ndarray: The image with the mask.
    """
    image = image.astype(np.float32) / 255
    mask = mask.astype(np.float32) / 255
    out = mask + image
    out = out / np.max(out)
    return (255 * out).astype(np.uint8)


def reconstruct_multiclass_mask(
    mask: np.ndarray,
    image_shape: tuple[int, ...],
    color_map: ListedColormap,
    ignore_class: int | None = None,
    ground_truth_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct a multiclass mask from a single channel mask.

    Args:
        mask (np.ndarray): A single channel mask.
        image_shape (Tuple[int, ...]): The shape of the image.
        color_map (ListedColormap): The color map to use.
        ignore_class (Optional[int], optional): The class to ignore. Defaults to None.
        ground_truth_mask (Optional[np.ndarray], optional): The ground truth mask. Defaults to None.

    Returns:
        mask: np.ndarray
    """
    output_mask = np.zeros(image_shape)
    for c in np.unique(mask):
        if ignore_class is not None and c == ignore_class:
            continue

        output_mask[mask == c] = color_map[str(c)]

    if ignore_class is not None and ground_truth_mask is not None:
        output_mask[ground_truth_mask == ignore_class] = [0, 0, 0]

    return output_mask


def plot_multiclass_prediction(
    image: np.ndarray,
    prediction_image: np.ndarray,
    ground_truth_image: np.ndarray,
    class_to_idx: dict[str, int],
    plot_original: bool = True,
    ignore_class: int | None = 0,
    image_height: int = 10,
    save_path: str | None = None,
    color_map: str = "tab20",
) -> None:
    """Function used to plot the image predicted.

    Args:
        image: The image to plot
        prediction_image: The prediction image
        ground_truth_image: The ground truth image
        class_to_idx: The class to idx mapping
        plot_original: Whether to plot the original image
        ignore_class: The class to ignore
        image_height: The height of the output figure
        save_path: The path to save the figure
        color_map: The color map to use. Defaults to "tab20".
    """
    image = image[0 : prediction_image.shape[0], 0 : prediction_image.shape[1], :]
    class_idxs = list(class_to_idx.values())
    cm = get_cmap(color_map)
    cmap = {str(c): tuple(int(i * 255) for i in cm(c / len(class_idxs))[:-1]) for c in class_idxs}
    output_images = []
    titles = []
    if plot_original:
        output_images.append(image)
        titles.append("Original Image")

    ground_truth_mask = reconstruct_multiclass_mask(ground_truth_image, image.shape, cmap, ignore_class=ignore_class)
    output_images.append(ground_truth_mask)
    titles.append("Ground Truth Mask")

    prediction_mask = reconstruct_multiclass_mask(
        prediction_image,
        image.shape,
        cmap,
        ignore_class=ignore_class,
    )
    output_images.append(prediction_mask)
    titles.append("Prediction Mask")
    if ignore_class is not None:
        prediction_mask = reconstruct_multiclass_mask(
            prediction_image, image.shape, cmap, ignore_class=ignore_class, ground_truth_mask=ground_truth_image
        )
        prediction_title = f"Prediction Mask \n (Ignoring Ground Truth Class: {ignore_class})"
        output_images.append(prediction_mask)
        titles.append(prediction_title)

    _, axs = plt.subplots(
        ncols=len(output_images),
        nrows=1,
        figsize=(len(output_images) * image_height, image_height),
        squeeze=False,
        facecolor="white",
    )
    for i, output_image in output_images:
        axs[0, i].imshow(show_mask_on_image(image, output_image))
        axs[0, i].set_title(titles[i])
        axs[0, i].axis("off")
    custom_lines = [Line2D([0], [0], color=tuple(i / 255.0 for i in cmap[str(c)]), lw=4) for c in class_idxs]
    custom_labels = list(class_to_idx.keys())
    axs[0, -1].legend(custom_lines, custom_labels, loc="center left", bbox_to_anchor=(1.01, 0.81), borderaxespad=0)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def plot_classification_results(
    test_dataset: torch.utils.data.Dataset,
    pred_labels: np.ndarray,
    test_labels: np.ndarray,
    class_name: str,
    original_folder: str,
    gradcam_folder: str | None = None,
    grayscale_cams: np.ndarray | None = None,
    unorm: Callable[[torch.Tensor], torch.Tensor] | None = None,
    idx_to_class: dict | None = None,
    what: str | None = None,
    real_class_to_plot: int | None = None,
    pred_class_to_plot: int | None = None,
    rows: int | None = 1,
    cols: int = 4,
    figsize: tuple[int, int] = (20, 20),
    gradcam: bool = False,
) -> None:
    """Plot and save images extracted from classification. If gradcam is True, same images
    with a gradcam heatmap (layered on original image) will also be saved.

    Args:
        test_dataset: Test dataset
        pred_labels: Predicted labels
        test_labels: Test labels
        class_name: Name of the examples' class
        original_folder: Folder where original examples will be saved
        gradcam_folder: Folder in which gradcam examples will be saved
        grayscale_cams: Grayscale gradcams (ordered as pred_labels and test_labels)
        unorm: Albumentations function to unormalize image
        idx_to_class: Dictionary of class conversion
        what: Can be "dis" or "conc", used if real_class_to_plot or pred_class_to_plot are None
        real_class_to_plot: Real class to plot.
        pred_class_to_plot: Pred class to plot.
        rows: How many rows in the plot there will be.
        cols: How many cols in the plot there will be.
        figsize: The figure size.
        gradcam: Whether to save also the gradcam version of the examples

    """
    to_plot = True
    if gradcam:
        if grayscale_cams is None:
            raise ValueError("gradcam is True but grayscale_cams is None")
        if gradcam_folder is None:
            raise ValueError("gradcam is True but gradcam_folder is None")

    if real_class_to_plot is not None:
        sample_idx = np.where(test_labels == real_class_to_plot)[0]
        if gradcam and grayscale_cams is not None:
            grayscale_cams = grayscale_cams[test_labels == real_class_to_plot]
        pred_labels = pred_labels[test_labels == real_class_to_plot]
        test_labels = test_labels[test_labels == real_class_to_plot]

    if pred_class_to_plot is not None:
        sample_idx = np.where(pred_labels == pred_class_to_plot)[0]
        if gradcam and grayscale_cams is not None:
            grayscale_cams = grayscale_cams[pred_labels == pred_class_to_plot]
        test_labels = test_labels[pred_labels == pred_class_to_plot]
        pred_labels = pred_labels[pred_labels == pred_class_to_plot]

    if pred_class_to_plot is None and real_class_to_plot is None:
        raise ValueError("'real_class_to_plot' and 'pred_class_to_plot' must not be both None")

    if what is not None:
        if what == "dis":
            cordant = pred_labels != test_labels
        elif what == "con":
            cordant = pred_labels == test_labels
        else:
            raise AssertionError(f"{what} not a valid plot type. Must be con or dis")

        sample_idx = np.array(sample_idx)[cordant]
        pred_labels = np.array(pred_labels)[cordant]
        test_labels = np.array(test_labels)[cordant]
        if gradcam:
            grayscale_cams = np.array(grayscale_cams)[cordant]

    # randomize
    idx_random = random.sample(range(len(sample_idx)), len(sample_idx))

    sample_idx = sample_idx[idx_random]
    pred_labels = pred_labels[idx_random]
    test_labels = test_labels[idx_random]
    if gradcam and grayscale_cams is not None:
        grayscale_cams = grayscale_cams[idx_random]

    cordant_chunks = list(_chunks(sample_idx, cols))

    if len(sample_idx) == 0:
        to_plot = False
        print("Nothing to plot")
    else:
        if rows is None or rows == 0:
            total_rows = len(cordant_chunks)
        else:
            total_rows = len(cordant_chunks[:rows])
        if gradcam:
            modality_list = ["original", "gradcam"]
        else:
            modality_list = ["original"]
        for modality in modality_list:
            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(
                fig,
                111,  # similar to subplot(111)
                nrows_ncols=(total_rows, cols),
                axes_pad=(0.2, 0.5),
            )
            for i, ax in enumerate(grid):
                if idx_to_class is not None:
                    try:
                        pred_label = idx_to_class[pred_labels[i]]
                    except Exception:
                        pred_label = pred_labels[i]
                    try:
                        test_label = idx_to_class[test_labels[i]]
                    except Exception:
                        test_label = test_labels[i]
                else:
                    pred_label = pred_labels[i]
                    test_label = test_labels[i]

                ax.axis("off")
                ax.set_title(f"True: {str(test_label)}\nPred {str(pred_label)}")
                image, _ = test_dataset[sample_idx[i]]

                if unorm is not None:
                    image = np.array(unorm(image))
                if modality == "gradcam" and grayscale_cams is not None:
                    grayscale_cam = grayscale_cams[i]
                    rgb_cam = show_cam_on_image(
                        np.transpose(image, (1, 2, 0)), grayscale_cam, use_rgb=True, image_weight=0.7
                    )

                    ax.imshow(rgb_cam, cmap="gray")
                    if i == len(pred_labels) - 1:
                        break
                else:
                    if isinstance(image, torch.Tensor):
                        image = image.cpu().numpy()

                    if image.max() <= 1:
                        image = image * 255
                    image = image.astype(int)

                    if len(image.shape) == 3:
                        if image.shape[0] == 1:
                            image = image[0]
                        elif image.shape[0] == 3:
                            image = image.transpose((1, 2, 0))
                    ax.imshow(image, cmap="gray")
                    if i == len(pred_labels) - 1:
                        break

            for item in grid:
                item.axis("off")

            if to_plot:
                save_folder: str = ""
                if modality == "gradcam" and gradcam_folder is not None:
                    save_folder = gradcam_folder
                elif modality == "original":
                    save_folder = original_folder
                else:
                    log.warning("modality %s has no corresponding folder", modality)
                    return

                plt.savefig(
                    os.path.join(save_folder, f"{what}cordant_{class_name}_" + modality + ".png"),
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()


def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
