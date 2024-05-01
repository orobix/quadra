from __future__ import annotations

from typing import cast

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from skimage.measure import label, regionprops  # pylint: disable=no-name-in-module

from quadra.utils.evaluation import dice


def _pad_to_shape(a: np.ndarray, shape: tuple, constant_values: int = 0) -> np.ndarray:
    """Pad lower - right with 0s
    Args:
        a: numpy array to pad
        shape: shape of the resulting np.array
        constant_values: value to pad.

    Returns:
        Padded array
    """
    y_, x_ = shape
    y, x = a.shape
    y_pad = y_ - y
    x_pad = x_ - x
    return np.pad(
        a,
        (
            (0, y_pad),
            (0, x_pad),
        ),
        mode="constant",
        constant_values=constant_values,
    )


def _get_iou(bboxes1: np.ndarray, bboxes2: np.ndarray, approx_iou: bool = False) -> np.ndarray:
    """Intersect over union
    Args:
        bboxes1: extracted bounding boxes
        bboxes2: ground truth
        approx_iou: flag to approximate.

    Returns:
        Intersect over union array
    """
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(x11, x21.T)
    yA = np.maximum(y11, y21.T)
    xB = np.minimum(x12, x22.T)
    yB = np.minimum(y12, y22.T)

    # compute the area of intersection rectangle
    inter_area = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)

    # compute the area of both the prediction and ground-truth rectangles
    box_a_area = (x12 - x11) * (y12 - y11)
    box_b_area = (x22 - x21) * (y22 - y21)

    if approx_iou:
        iou = inter_area / box_b_area.T
    else:
        iou = inter_area / (box_a_area + box_b_area.T - inter_area)

    return iou


def _get_dice_matrix(
    labels_pred: np.ndarray,
    n_labels_pred: int,
    labels_gt: np.ndarray,
    n_labels_gt: int,
) -> np.ndarray:
    """Create dice matrix
    Args:
        labels_pred: predicted label
        n_labels_pred: number of label predicted
        labels_gt: ground truth labels
        n_labels_gt: number of gt labels.

    Returns:
        Dice matrix
    """
    m = np.zeros((n_labels_pred, n_labels_gt))
    for i in range(n_labels_pred):
        pred = labels_pred == i + 1
        for j in range(n_labels_gt):
            gt = labels_gt == j + 1
            m[i, j] = dice(
                torch.Tensor(pred).unsqueeze(0).unsqueeze(0),
                torch.Tensor(gt).unsqueeze(0).unsqueeze(0),
                reduction="none",
            )
    return m


def segmentation_props(
    pred: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float, float, list[float], float, int, int, int, int]:
    """Return some information regarding a segmentation task.

    Args:
        pred (np.ndarray[bool]): Prediction of a segmentation model as
            a binary image.
        mask (np.ndarray[bool]): Ground truth mask as binary image

    Returns:
        1-Dice(pred, mask) Given a matrix (a_ij) = (1-Dice)(prediction_i, ground_truth_j),
            where prediction_i is the i-th prediction connected component and
            ground_truth_j is the j-th ground truth connected component,
            I compute the LSA (Linear Sum Assignment) to find the optimal 1-to-1 assignment
            between predictions and ground truths that minimize the (1-Dice) score.
            Then, for every unique pair of (predictioni, ground_truthj) we compute Average
             (1-Dice)(predictioni, ground_truthj)
        Average (1-Dice)(predictioni, ground_truthj) between True Positives
            (that is predictions associated to a ground truth), which is gratis from a[i,j],
            where the average is computed w.r.t. the total number of True Positives found.
        Average IoU(predictioni, ground_truthj) between True Positives
            (that is predictions associated to a ground truth),
            where the average is computed w.r.t. the total number of True Positives found.
            The IoU is computed between the minimum enclosing bounding box
            of a prediction and a ground truth.
        Average area of False Positives
        Histogram of false positives
        Average area of False Negatives
        Number of True Positives (predictions associated to a ground truth)
        Number of False Positives (predictions without a ground truth associated)
                and their avg. area (avg is taken w.r.t. the total number of False Positives found)
        Number of False Negatives (ground truth without a predictions associated)
            and their avg. area (avg is taken w.r.t. the total number of False Negatives found)
        Number of labels in the mask.
    """
    labels_pred, n_labels_pred = label(pred, connectivity=2, return_num=True, background=0)
    labels_mask, n_labels_mask = label(mask, connectivity=2, return_num=True, background=0)

    labels_pred = cast(np.ndarray, labels_pred)
    labels_mask = cast(np.ndarray, labels_mask)
    n_labels_pred = cast(int, n_labels_pred)
    n_labels_mask = cast(int, n_labels_mask)

    props_pred = regionprops(labels_pred)
    props_mask = regionprops(labels_mask)
    pred_bbox = np.array([props_pred[i].bbox for i in range(len(props_pred))])
    mask_bbox = np.array([props_mask[i].bbox for i in range(len(props_mask))])

    global_dice = float(
        dice(
            torch.Tensor(pred).unsqueeze(0).unsqueeze(0),
            torch.Tensor(mask).unsqueeze(0).unsqueeze(0),
        ).item()
    )
    lsa_iou = 0.0
    lsa_dice = 0.0
    tp_num = 0
    fp_num = 0
    fn_num = 0
    fp_area = 0.0
    fn_area = 0.0
    fp_hist: list[float] = []
    if n_labels_pred > 0 and n_labels_mask > 0:
        dice_mat = _get_dice_matrix(labels_pred, n_labels_pred, labels_mask, n_labels_mask)
        # Thresholding over Dice scores
        dice_mat = np.where(dice_mat <= 0.9, dice_mat, 1.0)
        iou_mat = _get_iou(pred_bbox, mask_bbox, approx_iou=False)
        dice_mat_shape = dice_mat.shape
        max_dim = np.max(dice_mat_shape)
        # Add dummy Dices so LSA is unique and i can compute FP and FN
        dice_mat = _pad_to_shape(dice_mat, (max_dim, max_dim), 1)
        lsa = linear_sum_assignment(dice_mat, maximize=False)
        for row, col in zip(lsa[0], lsa[1], strict=False):
            # More preds than GTs --> False Positive
            if row < n_labels_pred and col >= n_labels_mask:
                min_row = pred_bbox[row][0]
                min_col = pred_bbox[row][1]
                h = pred_bbox[row][2] - min_row
                w = pred_bbox[row][3] - min_col
                fp_num += 1
                area = pred[min_row : min_row + h, min_col : min_col + w].sum()
                fp_area += area
                fp_hist.append(area)
                continue

            # More GTs than preds --> False Negative
            if col < n_labels_mask and row >= n_labels_pred:
                min_row = mask_bbox[col][0]
                min_col = mask_bbox[col][1]
                h = mask_bbox[col][2] - min_row
                w = mask_bbox[col][3] - min_col
                fn_num += 1
                fn_area += mask[min_row : min_row + h, min_col : min_col + w].sum()
                continue

            # Real True Positive: a prediction has been assigned to a gt
            # with at least a 1-Dice score of 0.9
            if dice_mat[row, col] <= 0.9:
                tp_num += 1
                lsa_iou += iou_mat[row, col]
                lsa_dice += dice_mat[row, col]
            else:
                # Here we have both a FP and a FN
                min_row = pred_bbox[row][0]
                min_col = pred_bbox[row][1]
                h = pred_bbox[row][2] - min_row
                w = pred_bbox[row][3] - min_col
                fp_num += 1
                area = pred[min_row : min_row + h, min_col : min_col + w].sum()
                fp_area += area
                fp_hist.append(area)

                min_row = mask_bbox[col][0]
                min_col = mask_bbox[col][1]
                h = mask_bbox[col][2] - min_row
                w = mask_bbox[col][3] - min_col
                fn_num += 1
                fn_area += mask[min_row : min_row + h, min_col : min_col + w].sum()
    elif len(pred_bbox) > 0 and len(mask_bbox) == 0:  # No GTs --> FP
        for p_bbox in pred_bbox:
            min_row = p_bbox[0]
            min_col = p_bbox[1]
            h = p_bbox[2] - min_row
            w = p_bbox[3] - min_col
            fp_num += 1
            # print("FP area:", pred[min_row : min_row + h, min_col : min_col + w].sum())
            area = pred[min_row : min_row + h, min_col : min_col + w].sum()
            fp_area += area
            fp_hist.append(area)
    elif len(pred_bbox) == 0 and len(mask_bbox) > 0:  # No preds --> FN
        for m_bbox in mask_bbox:
            min_row = m_bbox[0]
            min_col = m_bbox[1]
            h = m_bbox[2] - min_row
            w = m_bbox[3] - min_col
            fn_num += 1
            # print("FN area:", mask[min_row : min_row + h, min_col : min_col + w].sum())
            fn_area += mask[min_row : min_row + h, min_col : min_col + w].sum()
    return (
        global_dice,
        lsa_dice,
        lsa_iou,
        fp_area,
        fp_hist,
        fn_area,
        tp_num,
        fp_num,
        fn_num,
        n_labels_mask,
    )
