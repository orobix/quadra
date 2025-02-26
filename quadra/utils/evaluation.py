from __future__ import annotations

import os
from ast import literal_eval
from collections.abc import Callable
from functools import wraps
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
import yaml
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.losses.constants import BINARY_MODE, MULTICLASS_MODE
from skimage.measure import label, regionprops  # pylint: disable=no-name-in-module

from quadra.utils.logger import get_logger
from quadra.utils.visualization import UnNormalize, create_grid_figure

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException  # noqa

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

log = get_logger(__name__)


def dice(
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-8,
    reduction: str | None = "mean",
) -> torch.Tensor:
    """Dice loss computation function.

    Args:
        input_tensor:  input tensor coming from a model
        target: target tensor to compare with
        smooth: smoothing factor
        eps: epsilon to avoid zero division
        reduction: reduction method, one of "mean", "sum", "none"

    Returns:
        The computed loss
    """
    bs = input_tensor.size(0)
    iflat = input_tensor.contiguous().view(bs, -1)
    tflat = target.contiguous().view(bs, -1)
    intersection = (iflat * tflat).sum(-1)
    loss = 1 - (2.0 * intersection + smooth) / (iflat.sum(-1) + tflat.sum(-1) + smooth + eps)

    if reduction == "mean":
        loss = loss.mean()
    return loss


def score_dice(
    y_pred,
    y_true,
    reduction=None,
) -> torch.Tensor:
    """Calculate dice score."""
    return 1 - dice(y_pred, y_true, reduction=reduction)


def score_dice_smp(y_pred: torch.Tensor, y_true: torch.Tensor, mode: str = "binary") -> torch.Tensor:
    """Compute dice using smp function. Handle both binary and multiclass scenario.

    Args:
        y_pred: 1xCxHxW one channel for each class
        y_true: 1x1xHxW true mask with value in [0, ..., n_classes]
        mode: "binary" or "multiclass"

    Returns:
        dice score
    """
    if mode not in {BINARY_MODE, MULTICLASS_MODE}:
        raise ValueError(f"Mode {mode} not valid.")

    loss = DiceLoss(mode=mode, from_logits=False)

    return 1 - loss(y_pred, y_true)


def calculate_mask_based_metrics(
    images: np.ndarray,
    th_masks: torch.Tensor,
    th_preds: torch.Tensor,
    threshold: float = 0.5,
    show_orj_predictions: bool = False,
    metric: Callable = score_dice,
    multilabel: bool = False,
    n_classes: int | None = None,
) -> tuple[
    dict[str, float],
    dict[str, list[np.ndarray]],
    dict[str, list[np.ndarray]],
    dict[str, list[str | float]],
]:
    """Calculate metrics based on masks and predictions.

    Args:
        images: Images.
        th_masks: masks are tensors.
        th_preds: predictions are tensors.
        threshold: Threshold to apply. Defaults to 0.5.
        show_orj_predictions: Flag to show original predictions. Defaults to False.
        metric: Metric to use comparison. Defaults to `score_dice`.
        multilabel: True if segmentation is multiclass.
        n_classes: Number of classes. If multilabel is False, this should be None.

    Returns:
        dict: Dictionary with metrics.
    """
    masks = th_masks.cpu().numpy()
    preds = th_preds.squeeze(0).cpu().numpy()
    th_thresh_preds = (th_preds > threshold).float().cpu()
    thresh_preds = th_thresh_preds.squeeze(0).numpy()
    dice_scores = metric(th_thresh_preds, th_masks, reduction=None).numpy()
    result: dict[str, Any] = {}
    if multilabel:
        if n_classes is None:
            raise ValueError("n_classes arg shouldn't be None when multilabel is True")
        preds_multilabel = (
            torch.nn.functional.one_hot(th_preds.to(torch.int64), num_classes=n_classes).squeeze(1).permute(0, 3, 1, 2)
        )
        masks_multilabel = (
            torch.nn.functional.one_hot(th_masks.to(torch.int64), num_classes=n_classes).squeeze(1).permute(0, 3, 1, 2)
        ).to(preds_multilabel.device)
        # get_stats multiclass, not considering background channel
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds_multilabel[:, 1:, :, :].long(), masks_multilabel[:, 1:, :, :].long(), mode="multilabel"
        )
    else:
        tp, fp, fn, tn = smp.metrics.get_stats(th_thresh_preds.long(), th_masks.long(), mode="binary")
    per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
    dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    result["F1_image"] = round(float(smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise").item()), 4)
    result["F1_pixel"] = round(float(smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()), 4)
    result["image_iou"] = round(float(per_image_iou.item()), 4) if not per_image_iou.isnan() else np.nan
    result["dataset_iou"] = round(float(dataset_iou.item()), 4) if not dataset_iou.isnan() else np.nan
    result["TP_pixel"] = tp.sum().item()
    result["FP_pixel"] = fp.sum().item()
    result["FN_pixel"] = fn.sum().item()
    result["TN_pixel"] = tn.sum().item()
    result["TP_image"] = 0
    result["FP_image"] = 0
    result["FN_image"] = 0
    result["TN_image"] = 0
    result["num_good_image"] = 0
    result["num_bad_image"] = 0
    bad_dice, good_dice = [], []
    fg: dict[str, list[np.ndarray]] = {"image": [], "mask": [], "thresh_pred": []}
    fb: dict[str, list[np.ndarray]] = {"image": [], "mask": [], "thresh_pred": []}
    if show_orj_predictions:
        fg["pred"] = []
        fb["pred"] = []

    area_graph: dict[str, list[str | float]] = {
        "Defect Area Percentage": [],
        "Accuracy": [],
    }
    for idx, (image, pred, mask, thresh_pred, dice_score) in enumerate(
        zip(images, preds, masks, thresh_preds, dice_scores, strict=False)
    ):
        if np.sum(mask) == 0:
            good_dice.append(dice_score)
        else:
            bad_dice.append(dice_score)
        if mask.sum() > 0:
            result["num_bad_image"] += 1
            if thresh_pred.sum() == 0:
                result["FN_image"] += 1
                fg["image"].append(image)
                fg["mask"].append(mask)
                if show_orj_predictions:
                    fg["pred"].append(pred)
                fg["thresh_pred"].append(thresh_pred)
            else:
                result["TP_image"] += 1
            rp = regionprops(label(mask[0]))
            for r in rp:
                mask_partial = th_masks[idx, :, r.bbox[0] : r.bbox[2], r.bbox[1] : r.bbox[3]]
                pred_partial = th_thresh_preds[idx, :, r.bbox[0] : r.bbox[2], r.bbox[1] : r.bbox[3]]
                tp, fp, fn, tn = smp.metrics.get_stats(pred_partial.long(), mask_partial.long(), mode="binary")
                area = tp + fn
                area_percentage = area.sum().item() * 100 / (image.shape[0] * image.shape[1])
                defect_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
                area_graph["Accuracy"].append(defect_acc.item() * 100)
                if area_percentage <= 1:
                    area_graph["Defect Area Percentage"].append("Very Small <1%")
                elif area_percentage <= 10:
                    area_graph["Defect Area Percentage"].append("Small <10%")
                elif area_percentage <= 25:
                    area_graph["Defect Area Percentage"].append("Medium <25%")
                else:
                    area_graph["Defect Area Percentage"].append("Large >25%")

        if mask.sum() == 0:
            result["num_good_image"] += 1
            if thresh_pred.sum() > 0:
                result["FP_image"] += 1
                fb["image"].append(image)
                fb["mask"].append(mask)
                if show_orj_predictions:
                    fb["pred"].append(pred)
                fb["thresh_pred"].append(thresh_pred)
            else:
                result["TN_image"] += 1
    result["bad_dice_score_mean"] = np.mean(bad_dice) if len(bad_dice) > 0 else "null"
    result["bad_dice_score_std"] = np.std(bad_dice) if len(bad_dice) > 0 else "null"
    result["good_dice_score_mean"] = np.mean(good_dice) if len(good_dice) > 0 else "null"
    result["good_dice_score_std"] = np.std(good_dice) if len(good_dice) > 0 else "null"
    return result, fg, fb, area_graph


def create_mask_report(
    stage: str,
    output: dict[str, torch.Tensor],
    mean: npt.ArrayLike,
    std: npt.ArrayLike,
    report_path: str,
    nb_samples: int = 6,
    analysis: bool = False,
    apply_sigmoid: bool = True,
    show_all: bool = False,
    threshold: float = 0.5,
    metric: Callable = score_dice,
    show_orj_predictions: bool = False,
) -> list[str]:
    """Create report for segmentation experiment
    Args:
        stage: stage name. Train, validation or test
        output: data produced by model
        report_path: experiment path
        mean: mean values
        std: std values
        nb_samples: number of samples
        analysis: if True, analysis will be created
        apply_sigmoid: if True, sigmoid will be applied to predictions
        show_all: if True, all images will be shown
        threshold: threshold for predictions
        metric: metric function
        show_orj_predictions: if True, original predictions will be shown.

    Returns:
        list of paths to created images.
    """
    if not os.path.exists(report_path):
        os.makedirs(report_path)

    th_images = output["image"]
    th_masks = output["mask"]
    th_preds = output["mask_pred"]
    th_labels = output["label"]
    n_classes = th_preds.shape[1]
    # TODO: Apply sigmoid is a wrong name now
    # TODO: Apply sigmoid false is untested
    if apply_sigmoid:
        if n_classes == 1:
            th_preds = torch.nn.Sigmoid()(th_preds)
            th_thresh_preds = (th_preds > threshold).float()
        else:
            th_preds = torch.nn.Softmax(dim=1)(th_preds)
            th_thresh_preds = torch.argmax(th_preds, dim=1).float().unsqueeze(1)
            # Compute labels from the given masks since by default they are all 0
            th_labels = th_masks.max(dim=2)[0].max(dim=2)[0].squeeze(dim=1)
            show_orj_predictions = False
    elif n_classes == 1:
        th_thresh_preds = (th_preds > threshold).float()
    else:
        th_thresh_preds = torch.argmax(th_preds, dim=1).float().unsqueeze(1)
        # Compute labels from the given masks since by default they are all 0
        th_labels = th_masks.max(dim=2)[0].max(dim=2)[0].squeeze(dim=1)
        show_orj_predictions = False

    mean = np.asarray(mean)
    std = np.asarray(std)
    unnormalize = UnNormalize(mean, std)

    images = np.array(
        [(unnormalize(image).cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8) for image in th_images]
    )
    masks = th_masks.cpu().numpy()
    preds = th_preds.squeeze(0).cpu().numpy()
    thresh_preds = th_thresh_preds.squeeze(0).cpu().numpy()
    dice_scores = metric(th_thresh_preds.cpu(), th_masks.cpu(), reduction=None).numpy()

    labels = th_labels.cpu().numpy()
    binary_labels = labels == 0

    row_names = ["Input", "Mask", "Pred", f"Pred>{threshold}"]
    bounds = [(0, 255), (0.0, float(n_classes - 1)), (0.0, 1.0), (0.0, float(n_classes - 1))]
    if not show_orj_predictions:
        row_names.pop(2)
        bounds.pop(2)

    if not show_all:
        sorted_idx = np.argsort(dice_scores)
    else:
        sorted_idx = np.arange(len(dice_scores))

    binary_labels = binary_labels[sorted_idx]

    non_zero_score_idx = sorted_idx[~binary_labels]
    zero_score_idx = sorted_idx[binary_labels]
    file_paths = []
    for name, current_score_idx in zip(["good", "bad"], [zero_score_idx, non_zero_score_idx], strict=False):
        if len(current_score_idx) == 0:
            continue

        nb_total_samples = len(current_score_idx)
        nb_selected_samples = nb_total_samples if nb_samples > nb_total_samples else nb_samples
        fig_w = int(nb_selected_samples * 2)
        fig_h = int(len(row_names) * 2)
        if not show_all:
            worst_idx = current_score_idx[:nb_selected_samples].tolist()
            best_idx = current_score_idx[-nb_selected_samples:].tolist()
            random_idx = np.random.choice(current_score_idx, nb_selected_samples, replace=False).tolist()

            indexes = {"best": best_idx, "worst": worst_idx, "random": random_idx}
        else:
            indexes = {"all": current_score_idx[:nb_selected_samples].tolist()}
        for k, v in indexes.items():
            file_path = os.path.join(report_path, f"{stage}_{name}_{k}_results.png")
            images_to_show = [images[v], masks[v], preds[v], thresh_preds[v]]
            if not show_orj_predictions or n_classes > 1:
                images_to_show.pop(2)
            create_grid_figure(
                images_to_show,
                nrows=len(row_names),
                ncols=nb_selected_samples,
                row_names=row_names,
                file_path=file_path,
                fig_size=(fig_w, fig_h),
                bounds=bounds,
            )
            file_paths.append(file_path)
    if analysis:
        analysis_file_path = os.path.join(report_path, f"{stage}_analysis.yaml")
        result, fg, fb, area_graph = calculate_mask_based_metrics(
            images=images,
            th_masks=th_masks,
            th_preds=th_thresh_preds,
            threshold=threshold,
            show_orj_predictions=show_orj_predictions,
            metric=metric,
            multilabel=bool(n_classes > 1),
            n_classes=n_classes,
        )

        if len(fg["image"]) > 0:
            if len(fg["image"]) > nb_samples:
                for k, v in fg.items():
                    fg[k] = v[:nb_samples]

            fg_file_path = os.path.join(report_path, f"{stage}_fn_results.png")
            fig_w = int(len(fg["image"]) * 2)
            create_grid_figure(
                [fg for _, fg in fg.items()],
                nrows=len(row_names),
                ncols=len(fg["image"]),
                row_names=row_names,
                file_path=fg_file_path,
                fig_size=(fig_w, fig_h),
                bounds=bounds,
            )
            file_paths.append(fg_file_path)

        if len(fb["image"]) > 0:
            if len(fb["image"]) > nb_samples:
                for k, v in fb.items():
                    fb[k] = v[:nb_samples]
            fb_file_path = os.path.join(report_path, f"{stage}_fp_results.png")

            fig_w = int(len(fb["image"]) * 2)
            create_grid_figure(
                [fb for _, fb in fb.items()],
                nrows=len(row_names),
                ncols=len(fb["image"]),
                row_names=row_names,
                file_path=fb_file_path,
                fig_size=(fig_w, fig_h),
                bounds=bounds,
            )
            file_paths.append(fb_file_path)
        if len(area_graph["Defect Area Percentage"]) > 0:
            fn_area_path = os.path.join(report_path, f"{stage}_acc_area.png")
            fn_area_df = pd.DataFrame(area_graph)
            ax = sns.boxplot(
                x="Defect Area Percentage",
                y="Accuracy",
                data=fn_area_df,
                order=["Very Small <1%", "Small <10%", "Medium <25%", "Large >25%"],
            )
            ax.set_facecolor("white")
            fig = ax.get_figure()
            fig.savefig(fn_area_path)
            plt.close(fig)

            file_paths.append(fn_area_path)
        with open(analysis_file_path, "w") as file:
            yaml.dump(literal_eval(str(result)), file, default_flow_style=False)
        file_paths.append(analysis_file_path)

    return file_paths


def automatic_datamodule_batch_size(batch_size_attribute_name: str = "batch_size"):
    """Automatically scale the datamodule batch size if the given function goes out of memory.

    Args:
        batch_size_attribute_name: The name of the attribute to modify in the datamodule
    """

    def decorator(func: Callable):
        """Decorator function."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            """Wrapper function."""
            is_func_finished = False
            starting_batch_size = None
            automatic_batch_size_completed = False

            if hasattr(self, "automatic_batch_size_completed"):
                automatic_batch_size_completed = self.automatic_batch_size_completed

            if hasattr(self, "automatic_batch_size"):
                if not hasattr(self.automatic_batch_size, "disable") or not hasattr(
                    self.automatic_batch_size, "starting_batch_size"
                ):
                    raise ValueError(
                        "The automatic_batch_size attribute should have the disable and starting_batch_size attributes"
                    )
                starting_batch_size = (
                    self.automatic_batch_size.starting_batch_size if not self.automatic_batch_size.disable else None
                )

            if starting_batch_size is not None and not automatic_batch_size_completed:
                # If we already tried to reduce the batch size, we will start from the last batch size
                log.info("Performing automatic batch size scaling from %d", starting_batch_size)
                setattr(self.datamodule, batch_size_attribute_name, starting_batch_size)

            while not is_func_finished:
                valid_exceptions = (RuntimeError,)

                if ONNX_AVAILABLE:
                    valid_exceptions += (RuntimeException,)

                try:
                    func(self, *args, **kwargs)
                    is_func_finished = True
                    self.automatic_batch_size_completed = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except valid_exceptions as e:
                    current_batch_size = getattr(self.datamodule, batch_size_attribute_name)
                    setattr(self.datamodule, batch_size_attribute_name, current_batch_size // 2)
                    log.warning(
                        "The function %s went out of memory, trying to reduce the batch size to %d",
                        func.__name__,
                        self.datamodule.batch_size,
                    )

                    if self.datamodule.batch_size == 0:
                        raise RuntimeError(
                            f"Unable to run {func.__name__} with batch size 1, the program will exit"
                        ) from e

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        return wrapper

    return decorator
