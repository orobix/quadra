from __future__ import annotations

import os
import warnings

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import label, regionprops  # pylint: disable=no-name-in-module
from tqdm import tqdm

from quadra.utils import utils
from quadra.utils.patch.dataset import PatchDatasetFileFormat, compute_patch_info, compute_patch_info_from_patch_dim

log = utils.get_logger(__name__)


def get_sorted_patches_by_image(test_results: pd.DataFrame, img_name: str) -> pd.DataFrame:
    """Gets the patches of a given image sorted by patch number.

    Args:
        test_results: Pandas dataframe containing test results like the one produced by SklearnClassificationTrainer
        img_name: name of the image used to filter the results.

    Returns:
        test results filtered by image name and sorted by patch number
    """
    img_patches = test_results[test_results["filename"] == os.path.splitext(img_name)[0]]
    patches_idx = np.array(
        [int(os.path.basename(x).split("_")[-1].replace(".png", "")) for x in img_patches["sample"].tolist()]
    )
    patches_idx = np.argsort(patches_idx).tolist()
    img_patches = img_patches.iloc[patches_idx]

    return img_patches


def compute_patch_metrics(
    test_img_info: list[PatchDatasetFileFormat],
    test_results: pd.DataFrame,
    overlap: float,
    idx_to_class: dict,
    patch_num_h: int | None = None,
    patch_num_w: int | None = None,
    patch_w: int | None = None,
    patch_h: int | None = None,
    return_polygon: bool = False,
    patch_reconstruction_method: str = "priority",
    annotated_good: list[int] | None = None,
) -> tuple[int, int, int, list[dict]]:
    """Compute the metrics of a patch dataset.

    Args:
        test_img_info: List of observation paths and mask paths
        test_results: Pandas dataframe containing the results of an SklearnClassificationTrainer utility
        patch_num_h: Number of vertical patches (required if patch_w and patch_h are None)
        patch_num_w: Number of horizontal patches (required if patch_w and patch_h are None)
        patch_h: Patch height (required if patch_num_h and patch_num_w are None)
        patch_w: Patch width (required if patch_num_h and patch_num_w are None)
        overlap: Percentage of overlap between the patches
        idx_to_class: Dict mapping an index to the corresponding class name
        return_polygon: if set to true convert the reconstructed mask into polygons, otherwise return the mask
        patch_reconstruction_method: How to compute the label of overlapping patches, can either be:
            priority: Assign the top priority label (i.e the one with greater index) to overlapping regions
            major_voting: Assign the most present label among the patches label overlapping a pixel
        annotated_good: List of indices of annotations to be treated as good.

    Returns:
        Tuple containing:
            false_region_bad: Number of false bad regions detected in the dataset
            false_region_good: Number of missed defects
            true_region_bad: Number of correctly identified defects
            reconstructions: If polygon is true this is a List of dict containing
                {
                    "file_path": image_path,
                    "mask_path": mask_path,
                    "file_name": observation_name,
                    "prediction": [{
                        "label": predicted_label,
                        "points": List of dict coordinates "x" and "y" representing the points of a polygon that
                        surrounds an image area covered by patches of label = predicted_label
                    }]
                }
            else its a list of dict containing
                {
                    "file_path": image_path,
                    "mask_path": mask_path,
                    "file_name": observation_name,
                    "prediction": numpy array containing the reconstructed mask
                }
    """
    assert patch_reconstruction_method in [
        "priority",
        "major_voting",
    ], "Patch reconstruction method not recognized, valid values are priority, major_voting"

    if (patch_h is not None and patch_w is not None) and (patch_num_h is not None and patch_num_w is not None):
        raise ValueError("Either number of patches or patch size is required for reconstruction")

    assert (patch_h is not None and patch_w is not None) or (patch_num_h is not None and patch_num_w is not None), (
        "Either number of patches or patch size is required for reconstruction"
    )

    if patch_h is not None and patch_w is not None and patch_num_h is not None and patch_num_w is not None:
        warnings.warn(
            "Both number of patches and patch dimension are specified, using number of patches by default",
            UserWarning,
            stacklevel=2,
        )

    log.info("Computing patch metrics!")

    false_region_bad = 0
    false_region_good = 0
    true_region_bad = 0
    reconstructions = []
    test_results["filename"] = test_results["sample"].apply(
        lambda x: "_".join(os.path.basename(x).replace("#DISCARD#", "").split("_")[0:-1])
    )

    for info in tqdm(test_img_info):
        img_path = info.image_path
        mask_path = info.mask_path

        img_json_entry = {
            "image_path": img_path,
            "mask_path": mask_path,
            "file_name": os.path.basename(img_path),
            "prediction": None,
        }

        test_img = cv2.imread(img_path)

        img_name = os.path.basename(img_path)

        h = test_img.shape[0]
        w = test_img.shape[1]

        gt_img = None

        if mask_path is not None and os.path.exists(mask_path):
            gt_img = cv2.imread(mask_path, 0)
            if test_img.shape[0:2] != gt_img.shape:
                # Ensure that the mask has the same size as the image by padding it with zeros
                log.warning("Found mask with different size than the image, padding it with zeros!")
                gt_img = np.pad(
                    gt_img, ((0, test_img.shape[0] - gt_img.shape[0]), (0, test_img.shape[1] - gt_img.shape[1]))
                )
        if patch_num_h is not None and patch_num_w is not None:
            patch_size, step = compute_patch_info(h, w, patch_num_h, patch_num_w, overlap)
        elif patch_h is not None and patch_w is not None:
            [patch_num_h, patch_num_w], step = compute_patch_info_from_patch_dim(h, w, patch_h, patch_w, overlap)
            patch_size = (patch_h, patch_w)
        else:
            raise ValueError(
                "Either number of patches or patch size is required for reconstruction, this should not happen"
                " at this stage"
            )

        img_patches = get_sorted_patches_by_image(test_results, img_name)
        pred = img_patches["pred_label"].to_numpy().reshape(patch_num_h, patch_num_w)

        # Treat annotated good predictions as background, this is an optimistic assumption that assumes that the
        # remaining background is good, but it is not always true so maybe on non annotated areas we are missing
        # defects and it would be necessary to handle this in a different way.
        if annotated_good is not None:
            pred[np.isin(pred, annotated_good)] = 0
        if patch_num_h is not None and patch_num_w is not None:
            output_mask, predicted_defect = reconstruct_patch(
                input_img_shape=test_img.shape,
                patch_size=patch_size,
                pred=pred,
                patch_num_h=patch_num_h,
                patch_num_w=patch_num_w,
                idx_to_class=idx_to_class,
                step=step,
                return_polygon=return_polygon,
                method=patch_reconstruction_method,
            )
        else:
            raise ValueError("`patch_num_h` and `patch_num_w` cannot be None at this point")

        if return_polygon:
            img_json_entry["prediction"] = predicted_defect
        else:
            img_json_entry["prediction"] = output_mask

        reconstructions.append(img_json_entry)
        if gt_img is not None:
            if annotated_good is not None:
                gt_img[np.isin(gt_img, annotated_good)] = 0

            gt_img_binary = (gt_img > 0).astype(bool)
            regions_pred = label(output_mask).astype(np.uint8)

            for k in range(1, regions_pred.max() + 1):
                region = (regions_pred == k).astype(bool)
                # If there's no overlap with the gt
                if np.sum(np.bitwise_and(region, gt_img_binary)) == 0:
                    false_region_bad += 1

            output_mask = (output_mask > 0).astype(np.uint8)
            gt_img = label(gt_img)

            if gt_img is None:
                raise RuntimeError("Ground truth mask is None after label and it should not be")

            for i in range(1, gt_img.max() + 1):
                region = (gt_img == i).astype(bool)
                if np.sum(np.bitwise_and(region, output_mask)) == 0:
                    false_region_good += 1
                else:
                    true_region_bad += 1

    return false_region_bad, false_region_good, true_region_bad, reconstructions


def reconstruct_patch(
    input_img_shape: tuple[int, ...],
    patch_size: tuple[int, int],
    pred: np.ndarray,
    patch_num_h: int,
    patch_num_w: int,
    idx_to_class: dict,
    step: tuple[int, int],
    return_polygon: bool = True,
    method: str = "priority",
) -> tuple[np.ndarray, list[dict]]:
    """Reconstructs the prediction image from the patches.

    Args:
        input_img_shape: The size of the reconstructed image
        patch_size: Array defining the patch size
        pred: Numpy array containing reconstructed prediction (patch_num_h x patch_num_w)
        patch_num_h: Number of vertical patches
        patch_num_w: Number of horizontal patches
        idx_to_class: Dictionary mapping indices to labels
        step: Array defining the step size to be used for reconstruction
        return_polygon: If true compute predicted polygons. Defaults to True.
        method: Reconstruction method to be used. Currently supported: "priority" and "major_voting"

    Returns:
        (reconstructed_prediction_image, predictions) where predictions is an array of objects
            [{
                "label": Predicted_label,
                "points": List of dict coordinates "x" and "y" representing the points of a polygon that
                    surrounds an image area covered by patches of label = predicted_label
            }]
    """
    if method == "priority":
        return _reconstruct_patch_priority(
            input_img_shape,
            patch_size,
            pred,
            patch_num_h,
            patch_num_w,
            idx_to_class,
            step,
            return_polygon,
        )
    if method == "major_voting":
        return _reconstruct_patch_major_voting(
            input_img_shape,
            patch_size,
            pred,
            patch_num_h,
            patch_num_w,
            idx_to_class,
            step,
            return_polygon,
        )

    raise ValueError(f"Invalid reconstruction method {method}")


def _reconstruct_patch_priority(
    input_img_shape: tuple[int, ...],
    patch_size: tuple[int, int],
    pred: np.ndarray,
    patch_num_h: int,
    patch_num_w: int,
    idx_to_class: dict,
    step: tuple[int, int],
    return_polygon: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Reconstruct patch polygons using the priority method."""
    final_mask = np.zeros([input_img_shape[0], input_img_shape[1]], dtype=np.uint8)
    predicted_defect = []

    for i in range(1, pred.max() + 1):
        white_patch = np.full((patch_size[0], patch_size[1]), i, dtype=np.uint8)
        masked_pred = (pred == i).astype(np.uint8)

        if masked_pred.sum() == 0:
            continue

        mask_img = np.zeros([input_img_shape[0], input_img_shape[1]], dtype=np.uint8)

        for h in range(patch_num_h):
            for w in range(patch_num_w):
                if masked_pred[h, w] == 1:
                    patch_location_h = step[0] * h
                    patch_location_w = step[1] * w

                    # Move replicated patches prediction in the correct position of the original image if needed
                    if patch_location_h + patch_size[0] > mask_img.shape[0]:
                        patch_location_h = mask_img.shape[0] - patch_size[0]

                    if patch_location_w + patch_size[1] > mask_img.shape[1]:
                        patch_location_w = mask_img.shape[1] - patch_size[1]

                    mask_img[
                        patch_location_h : patch_location_h + patch_size[0],
                        patch_location_w : patch_location_w + patch_size[1],
                    ] = white_patch

        mask_img = mask_img[0 : input_img_shape[0], 0 : input_img_shape[1]]

        # Priority is given by the index of the class, the larger, the more important
        final_mask = np.maximum(mask_img, final_mask)

    if final_mask.sum() != 0 and return_polygon:
        for lab in np.unique(final_mask):
            if lab == 0:
                continue

            polygon = from_mask_to_polygon((final_mask == lab).astype(np.uint8))

            for pol in polygon:
                class_entry = {
                    "label": idx_to_class.get(lab),
                    "points": pol,
                }

                predicted_defect.append(class_entry)

    return final_mask, predicted_defect


def _reconstruct_patch_major_voting(
    input_img_shape: tuple[int, ...],
    patch_size: tuple[int, int],
    pred: np.ndarray,
    patch_num_h: int,
    patch_num_w: int,
    idx_to_class: dict,
    step: tuple[int, int],
    return_polygon: bool = True,
):
    """Reconstruct patch polygons using the major voting method."""
    predicted_defect = []

    final_mask = np.zeros([input_img_shape[0], input_img_shape[1], np.max(pred) + 1], dtype=np.uint8)
    white_patch = np.ones((patch_size[0], patch_size[1]), dtype=np.uint8)

    for i in range(1, pred.max() + 1):
        masked_pred = (pred == i).astype(np.uint8)

        if masked_pred.sum() == 0:
            continue

        mask_img = np.zeros([input_img_shape[0], input_img_shape[1]], dtype=np.uint8)

        for h in range(patch_num_h):
            for w in range(patch_num_w):
                if masked_pred[h, w] == 1:
                    patch_location_h = step[0] * h
                    patch_location_w = step[1] * w

                    # Move replicated patches prediction in the correct position of the original image if needed
                    if patch_location_h + patch_size[0] > mask_img.shape[0]:
                        patch_location_h = mask_img.shape[0] - patch_size[0]

                    if patch_location_w + patch_size[1] > mask_img.shape[1]:
                        patch_location_w = mask_img.shape[1] - patch_size[1]

                    mask_img[
                        patch_location_h : patch_location_h + patch_size[0],
                        patch_location_w : patch_location_w + patch_size[1],
                    ] += white_patch

        mask_img = mask_img[0 : input_img_shape[0], 0 : input_img_shape[1]]
        final_mask[:, :, i] = mask_img

    # Since argmax returns first element on ties and the priority is defined from 0 to n_classes,
    # I needed a way to get the last element on ties, this code achieves that
    final_mask = ((final_mask.shape[-1] - 1) - np.argmax(final_mask[..., ::-1], axis=-1)) * np.invert(
        np.all(final_mask == 0, axis=-1)
    )

    if final_mask.sum() != 0 and return_polygon:
        for lab in np.unique(final_mask):
            if lab == 0:
                continue

            polygon = from_mask_to_polygon((final_mask == lab).astype(np.uint8))

            for pol in polygon:
                class_entry = {
                    "label": idx_to_class.get(lab),
                    "points": pol,
                }

                predicted_defect.append(class_entry)

    return final_mask, predicted_defect


def from_mask_to_polygon(mask_img: np.ndarray) -> list:
    """Convert a mask of pattern to a list of polygon vertices.

    Args:
        mask_img: masked patch reconstruction image
    Returns:
        a list of lists containing the coordinates of the polygons containing each region of the mask:
        [
            [
                {
                    "x": 1.1,
                    "y": 2.2
                },
                {
                    "x": 2.1,
                    "y": 3.2
                }
            ], ...
        ].
    """
    points_dict = []
    # find vertices of polygon: points -> list of array of dim n_vertex, 1, 2(x,y)
    polygon_points, hier = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    if not hier[:, :, 2:].all(-1).all():  # there are holes
        holes = ndimage.binary_fill_holes(mask_img).astype(int)
        holes -= mask_img
        holes = (holes > 0).astype(np.uint8)
        if holes.sum() > 0:  # there are holes
            for hole in regionprops(label(holes)):
                a, _, _, _d = hole.bbox
                mask_img[a] = 0

        polygon_points, hier = cv2.findContours(mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

    for pol in polygon_points:
        #  pol: n_vertex, 1, 2
        current_poly = []
        for point in pol:
            current_poly.append({"x": int(point[0, 0]), "y": int(point[0, 1])})
        points_dict.append(current_poly)

    return points_dict
