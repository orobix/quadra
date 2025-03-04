from __future__ import annotations

import glob
import itertools
import json
import math
import os
import random
import shutil
import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Any

import cv2
import h5py
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops  # pylint: disable=no-name-in-module
from skimage.util import view_as_windows
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm
from tripy import earclip

from quadra.utils import utils

log = utils.get_logger(__name__)


@dataclass
class PatchDatasetFileFormat:
    """Model representing the content of the patch dataset split_files field in the info.json file."""

    image_path: str
    mask_path: str | None = None


@dataclass
class PatchDatasetInfo:
    """Model representing the content of the patch dataset info.json file."""

    patch_size: tuple[int, int] | None
    patch_number: tuple[int, int] | None
    annotated_good: list[int] | None
    overlap: float
    train_files: list[PatchDatasetFileFormat]
    val_files: list[PatchDatasetFileFormat]
    test_files: list[PatchDatasetFileFormat]

    @staticmethod
    def _map_files(files: list[Any]):
        """Convert a list of dict to a list of PatchDatasetFileFormat."""
        mapped_files = []
        for file in files:
            current_file = file
            if isinstance(file, dict):
                current_file = PatchDatasetFileFormat(**current_file)
            mapped_files.append(current_file)

        return mapped_files

    def __post_init__(self):
        self.train_files = self._map_files(self.train_files)
        self.val_files = self._map_files(self.val_files)
        self.test_files = self._map_files(self.test_files)


def get_image_mask_association(
    data_folder: str,
    mask_folder: str | None = None,
    mask_extension: str = "",
    warning_on_missing_mask: bool = True,
) -> list[dict]:
    """Function used to match images and mask from a folder or sub-folders.

    Args:
        data_folder: root data folder containing images or images and masks
        mask_folder: Optional root directory used to search only the masks
        mask_extension: extension used to identify the mask file, it's mandatory if mask_folder is not specified
            warning_on_missing_mask: if set to True a warning will be raised if a mask is missing, disable if you know
            that many images do not have a mask.
        warning_on_missing_mask: if set to True a warning will be raised if a mask is missing, disable if you know

    Returns:
        List of dict like:
        [
        {
            'base_name': '161927.tiff',
            'path': 'test_dataset_patch/images/161927.tiff',
            'mask': 'test_dataset_patch/masks/161927_mask.tiff'
        }, ...
        ]
    """
    # get all the images from the data folder
    data_images = glob.glob(os.path.join(data_folder, "**", "*"), recursive=True)

    basenames = [os.path.splitext(os.path.basename(image))[0] for image in data_images]

    if len(set(basenames)) != len(basenames):
        raise ValueError("Found multiple images with the same name and different extension, this is not supported.")

    log.info("Found: %d images in %s", len(data_images), data_folder)
    # divide images and mask if in the same folder
    # if mask folder is specified search mask in that folder
    if mask_folder:
        masks_images = []
        for basename in basenames:
            mask_path = os.path.join(mask_folder, f"{basename}{mask_extension}.*")
            mask_path_list = glob.glob(mask_path)

            if len(mask_path_list) == 1:
                masks_images.append(mask_path_list[0])
            elif warning_on_missing_mask:
                log.warning("Mask for %s not found", basename)
    else:
        if mask_extension == "":
            raise ValueError("If no mask folder is provided, mask extension is mandatory it cannot be empty.")

        masks_images = [image for image in data_images if mask_extension in image]
        data_images = [image for image in data_images if mask_extension not in image]

    # build support dictionary
    unique_images = [{"base_name": os.path.basename(image), "path": image, "mask": None} for image in data_images]

    images_stem = [os.path.splitext(str(image["base_name"]))[0] + mask_extension for image in unique_images]
    masks_stem = [os.path.splitext(os.path.basename(mask))[0] for mask in masks_images]

    # search corrispondency between file or folders
    for i, image_stem in enumerate(images_stem):
        if image_stem in masks_stem:
            unique_images[i]["mask"] = masks_images[masks_stem.index(image_stem)]

    log.info("Unique images with mask: %d", len([uni for uni in unique_images if uni.get("mask") is not None]))
    log.info("Unique images with no mask: %d", len([uni for uni in unique_images if uni.get("mask") is None]))

    return unique_images


def compute_patch_info(
    img_h: int,
    img_w: int,
    patch_num_h: int,
    patch_num_w: int,
    overlap: float = 0.0,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute the patch size and step size given the number of patches and the overlap.

    Args:
        img_h: height of the image
        img_w: width of the image
        patch_num_h: number of vertical patches
        patch_num_w: number of horizontal patches
        overlap: percentage of overlap between patches.

    Returns:
        Tuple containing:
            patch_size: [size_y, size_x] Dimension of the patch
            step_size: [step_y, step_x]  Step size
    """
    patch_size_h = np.ceil(img_h / (1 + (patch_num_h - 1) - (patch_num_h - 1) * overlap)).astype(int)
    step_h = patch_size_h - np.ceil(overlap * patch_size_h).astype(int)

    patch_size_w = np.ceil(img_w / (1 + (patch_num_w - 1) - (patch_num_w - 1) * overlap)).astype(int)
    step_w = patch_size_w - np.ceil(overlap * patch_size_w).astype(int)

    # We want a combination of patch size and step that if the image is not divisible by the number of patches
    # will try to fit the maximum number of patches in the image + ONLY 1 extra patch that will be taken from the end
    # of the image.

    counter = 0
    original_patch_size_h = patch_size_h
    original_patch_size_w = patch_size_w
    original_step_h = step_h
    original_step_w = step_w

    while (patch_num_h - 1) * step_h + patch_size_h < img_h or (patch_num_h - 2) * step_h + patch_size_h > img_h:
        counter += 1
        if (patch_num_h - 1) * (step_h + 1) + patch_size_h < img_h:
            step_h += 1
        else:
            patch_size_h += 1

        if counter == 100:
            # We probably entered an infinite loop, restart with smaller step size
            step_h = original_step_h - 1
            patch_size_h = original_patch_size_h
            counter = 0

    counter = 0
    while (patch_num_w - 1) * step_w + patch_size_w < img_w or (patch_num_w - 2) * step_w + patch_size_w > img_w:
        counter += 1
        if (patch_num_w - 1) * (step_w + 1) + patch_size_w < img_w:
            step_w += 1
        else:
            patch_size_w += 1

        if counter == 100:
            # We probably entered an infinite loop, restart with smaller step size
            step_w = original_step_w - 1
            patch_size_w = original_patch_size_w
            counter = 0

    return (patch_size_h, patch_size_w), (step_h, step_w)


def compute_patch_info_from_patch_dim(
    img_h: int,
    img_w: int,
    patch_height: int,
    patch_width: int,
    overlap: float = 0.0,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute patch info given the patch dimension
    Args:
        img_h: height of the image
        img_w: width of the image
        patch_height: patch height
        patch_width: patch width
        overlap: overlap percentage [0, 1].

    Returns:
        Tuple of number of patches, step

    """
    assert 1 >= overlap >= 0, f"Invalid overlap. Must be between [0, 1], received {overlap}"
    step_h = patch_height - int(overlap * patch_height)
    step_w = patch_width - int(overlap * patch_width)

    patch_num_h = np.ceil(((img_h - patch_height) / step_h) + 1).astype(int)
    patch_num_w = np.ceil(((img_w - patch_width) / step_w) + 1).astype(int)

    # Handle the case where the last patch does not cover the full image, I need to do this rather than np.ceil
    # because I don't want to add a new patch if the last one exceeds already the image!
    if ((patch_num_h - 1) * step_h) + patch_height < img_h:
        patch_num_h += 1
    if ((patch_num_w - 1) * step_w) + patch_width < img_w:
        patch_num_w += 1

    return (patch_num_h, patch_num_w), (step_h, step_w)


def from_rgb_to_idx(img: np.ndarray, class_to_color: dict, class_to_idx: dict) -> np.ndarray:
    """Args:
        img: Rgb mask in which each different color is associated with a class
        class_to_color: Dict "key": [R, G, B]
        class_to_idx: Dict "key": class_idx.

    Returns:
        Grayscale mask in which each class is associated with a specific index
    """
    img = img.astype(int)
    # Use negative values to avoid strange behaviour in the remote eventuality
    # of someone using a color like [1, 255, 255]
    for classe, color in class_to_color.items():
        img[np.all(img == color, axis=-1).astype(bool), 0] = -class_to_idx[classe]

    img = np.abs(img[:, :, 0])

    return img.astype(np.uint8)


def __save_patch_dataset(
    image_patches: np.ndarray,
    labelled_patches: np.ndarray | None = None,
    mask_patches: np.ndarray | None = None,
    labelled_mask: np.ndarray | None = None,
    output_folder: str = "extraction_data",
    image_name: str = "example",
    area_threshold: float = 0.45,
    area_defect_threshold: float = 0.2,
    mask_extension: str = "_mask",
    save_mask: bool = False,
    mask_output_folder: str | None = None,
    class_to_idx: dict | None = None,
) -> None:
    """Given a view_as_window computed patches, masks and labelled mask, save all the images in subdirectory
    divided by name and position in the grid, ambiguous patches i.e. the one that contains defects but with not enough
    to go above defined thresholds are marked as #DISCARD# and should be discarded in training.
    Patches of images without ground truth are saved inside the None folder.

    Args:
        image_patches: [n, m, patch_w, patch_h, channel] numpy array of the image patches
        mask_patches: [n, m, patch_w, patch_h] numpy array of mask patches
        labelled_patches: [n, m, patch_w, patch_h] numpy array of labelled mask patch
        labelled_mask: numpy array in which each defect in the image is labelled using connected components
        class_to_idx: Dictionary with the mapping {"class" -> class in mask}, it must cover all indices
            contained in the masks
        save_mask: flag to save or ignore mask
        output_folder: folder where to save data
        mask_extension: postfix of the saved mask based on the image name
        mask_output_folder: Optional folder in which to save the masks
        image_name: name to use in order to save the data
        area_threshold: minimum percentage of defected patch area present in the mask to classify the patch as defect
        area_defect_threshold: minimum percentage of single defect present in the patch to classify the patch as defect

    Returns:
        None
    """
    if class_to_idx is not None:
        log.debug("Classes from dict: %s", class_to_idx)
        index_to_class = {v: k for k, v in class_to_idx.items()}
        log.debug("Inverse class: %s", index_to_class)
        reference_classes = index_to_class

        if mask_patches is not None:
            classes_in_mask = set(np.unique(mask_patches))
            missing_classes = set(classes_in_mask).difference(class_to_idx.values())

            assert len(missing_classes) == 0, f"Found index in mask that has no corresponding class {missing_classes}"
    elif mask_patches is not None:
        reference_classes = {k: str(v) for k, v in enumerate(list(np.unique(mask_patches)))}
    else:
        raise ValueError("If no `class_to_idx` is provided, `mask_patches` must be provided")

    log.debug("Classes from mask: %s", reference_classes)
    class_to_idx = {v: k for k, v in reference_classes.items()}
    log.debug("Final reference classes: %s", reference_classes)

    # create subdirectory for the saving data
    for cl in reference_classes.values():
        os.makedirs(os.path.join(output_folder, str(cl)), exist_ok=True)

        if mask_output_folder is not None:
            os.makedirs(os.path.join(output_folder, mask_output_folder, str(cl)), exist_ok=True)

    if mask_output_folder is None:
        mask_output_folder = output_folder
    else:
        mask_output_folder = os.path.join(output_folder, mask_output_folder)

    log.debug("Mask out: %s", mask_output_folder)

    if mask_patches is None:
        os.makedirs(os.path.join(output_folder, str(None)), exist_ok=True)
    # for [i, j] in patches location
    for row_index in range(image_patches.shape[0]):
        for col_index in range(image_patches.shape[1]):
            # default class it's the one in index 0
            output_class = reference_classes.get(0)
            image = image_patches[row_index, col_index]

            discard_in_training = True
            if mask_patches is not None and labelled_patches is not None:
                discard_in_training = False
                max_defected_area = 0
                mask = mask_patches[row_index, col_index]
                patch_area_th = mask.shape[0] * mask.shape[1] * area_threshold

                if mask.sum() > 0:
                    discard_in_training = True
                    for k, v in class_to_idx.items():
                        if v == 0:
                            continue

                        mask_patch = mask == int(v)
                        defected_area = mask_patch.sum()

                        if defected_area > 0:
                            # If enough defected area is inside the patch
                            if defected_area > patch_area_th:
                                if defected_area > max_defected_area:
                                    output_class = k
                                    max_defected_area = defected_area
                                    discard_in_training = False
                            else:
                                all_defects_in_patch = mask_patch * labelled_patches[row_index, col_index]

                                # For each different defect inside the area check
                                # if enough part of it is contained in the patch
                                for defect_id in np.unique(all_defects_in_patch):
                                    if defect_id == 0:
                                        continue

                                    defect_area_in_patch = (all_defects_in_patch == defect_id).sum()
                                    defect_area_th = (labelled_mask == defect_id).sum() * area_defect_threshold

                                    if defect_area_in_patch > defect_area_th:
                                        output_class = k
                                        if defect_area_in_patch > max_defected_area:
                                            max_defected_area = defect_area_in_patch
                                            discard_in_training = False
                        else:
                            discard_in_training = False

                if save_mask:
                    mask_name = f"{image_name}_{row_index * image_patches.shape[1] + col_index}{mask_extension}.png"

                    if discard_in_training:
                        mask_name = "#DISCARD#" + mask_name
                    cv2.imwrite(
                        os.path.join(
                            mask_output_folder,
                            output_class,  # type: ignore[arg-type]
                            mask_name,
                        ),
                        mask.astype(np.uint8),
                    )
            else:
                output_class = "None"

            patch_name = f"{image_name}_{row_index * image_patches.shape[1] + col_index}.png"
            if discard_in_training:
                patch_name = "#DISCARD#" + patch_name

            cv2.imwrite(
                os.path.join(
                    output_folder,
                    output_class,  # type: ignore[arg-type]
                    patch_name,
                ),
                image,
            )


def generate_patch_dataset(
    data_dictionary: list[dict],
    class_to_idx: dict,
    val_size: float = 0.3,
    test_size: float = 0.0,
    seed: int = 42,
    patch_number: tuple[int, int] | None = None,
    patch_size: tuple[int, int] | None = None,
    overlap: float = 0.0,
    output_folder: str = "extraction_data",
    save_original_images_and_masks: bool = True,
    area_threshold: float = 0.45,
    area_defect_threshold: float = 0.2,
    mask_extension: str = "_mask",
    mask_output_folder: str | None = None,
    save_mask: bool = False,
    clear_output_folder: bool = False,
    mask_preprocessing: Callable | None = None,
    train_filename: str = "dataset.txt",
    repeat_good_images: int = 1,
    balance_defects: bool = True,
    annotated_good: list[str] | None = None,
    num_workers: int = 1,
) -> dict | None:
    """Giving a data_dictionary as:
    >>> {
    >>>     'base_name': '163931_1_5.jpg',
    >>>     'path': 'extraction_data/1/163931_1_5.jpg',
    >>>     'mask': 'extraction_data/1/163931_1_5_mask.jpg'
    >>>}
    This function will generate patches datasets based on the defined split number, one for training, one for validation
    and one for testing respectively under output_folder/train, output_folder/val and output_folder/test, the training
    dataset will contain h5 files and a txt file resulting from a call to the
    generate_classification_patch_train_dataset, while the test dataset will contain patches saved on disk divided
    in subfolders per class, patch extraction is done in a sliding window fashion.
    Original images and masks (preprocessed if mask_preprocessing is present) will also be saved under
    output_folder/original/images and output_folder/original/masks.
    If patch number is specified the patch size will be calculated accordingly, if the image is not divisible by the
    patch number two possible behaviours can occur:
        - if the patch reconstruction is smaller than the original image a new patch will be generated containing the
        pixels from the edge of the image (E.g the new patch will contain the last patch_size pixels of the original
        image)
        - if the patch reconstruction is bigger than the original image the last patch will contain the pixels from the
        edge of the image same as above, but without adding a new patch to the count.

    Args:
        data_dictionary: Dictionary as above
        val_size: percentage of the dictionary entries to be used for validation
        test_size: percentage of the dictionary entries to be used for testing
        seed: seed for rng based operations
        clear_output_folder: flag used to delete all the data in subfolder
        class_to_idx: Dictionary {"defect": value in mask.. }
        output_folder: root_folder where to extract the data
        save_original_images_and_masks: If True, images and masks will be copied inside output_folder/original/
        area_threshold: Minimum percentage of defected patch area present in the mask to classify the patch as defect
        area_defect_threshold: Minimum percentage of single defect present in the patch to classify the patch as defect
        mask_extension: Extension used to assign image to mask
        mask_output_folder: Optional folder in which to save the masks
        save_mask: Flag to save the mask
        patch_number: Optional number of patches for each side, required if patch_size is None
        patch_size: Optional dimension of the patch, required if patch_number is None
        overlap: Overlap of the patches [0, 1]
        mask_preprocessing: Optional function applied to masks, this can be useful for example to convert an image in
            range [0-255] to the required [0-1]
        train_filename: Name of the file containing mapping between h5 files and labels for training
        repeat_good_images: Number of repetition for images with emtpy or None mask
        balance_defects: If true add one good entry for each defect extracted
        annotated_good: List of labels that are annotated but considered as good
        num_workers: Number of workers used for the h5 creation

    Returns:
        None if data_dictionary is empty, otherwise return a dictionary containing informations about the dataset

    """
    if len(data_dictionary) == 0:
        warnings.warn("Input data dictionary is empty!", UserWarning, stacklevel=2)
        return None

    if val_size < 0 or test_size < 0 or (val_size + test_size) > 1:
        raise ValueError("Validation and Test size must be greater or equal than zero and sum up to maximum 1")
    if clear_output_folder and os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "original"), exist_ok=True)
    if save_original_images_and_masks:
        log.info("Moving original images and masks to dataset folder...")
        os.makedirs(os.path.join(output_folder, "original", "images"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "original", "masks"), exist_ok=True)

        for i, item in enumerate(data_dictionary):
            img_new_path = os.path.join("original", "images", item["base_name"])
            shutil.copy(item["path"], os.path.join(output_folder, img_new_path))
            data_dictionary[i]["path"] = img_new_path

            if item["mask"] is not None:
                mask = cv2.imread(item["mask"])
                if mask_preprocessing is not None:
                    mask = mask_preprocessing(mask).astype(np.uint8)
                mask_new_path = os.path.join("original", "masks", os.path.splitext(item["base_name"])[0] + ".png")
                cv2.imwrite(os.path.join(output_folder, mask_new_path), mask)
                data_dictionary[i]["mask"] = mask_new_path

    shuffled_indices = np.random.default_rng(seed).permutation(len(data_dictionary))
    data_dictionary = [data_dictionary[i] for i in shuffled_indices]
    log.info("Performing multilabel stratification...")
    train_data_dictionary, val_data_dictionary, test_data_dictionary = multilabel_stratification(
        output_folder=output_folder,
        data_dictionary=data_dictionary,
        num_classes=len(class_to_idx.values()),
        val_size=val_size,
        test_size=test_size,
    )

    log.info("Train set size: %d", len(train_data_dictionary))
    log.info("Validation set size: %d", len(val_data_dictionary))
    log.info("Test set size: %d", len(test_data_dictionary))

    idx_to_class = {v: k for (k, v) in class_to_idx.items()}

    os.makedirs(output_folder, exist_ok=True)

    dataset_info = {
        "patch_size": patch_size,
        "patch_number": patch_number,
        "overlap": overlap,
        "annotated_good": annotated_good,
        "train_files": [{"image_path": x["path"], "mask_path": x["mask"]} for x in train_data_dictionary],
        "val_files": [{"image_path": x["path"], "mask_path": x["mask"]} for x in val_data_dictionary],
        "test_files": [{"image_path": x["path"], "mask_path": x["mask"]} for x in test_data_dictionary],
    }

    with open(os.path.join(output_folder, "info.json"), "w") as f:
        json.dump(dataset_info, f)

    if len(train_data_dictionary) > 0:
        log.info("Generating train set")
        generate_patch_sampling_dataset(
            data_dictionary=train_data_dictionary,
            patch_number=patch_number,
            patch_size=patch_size,
            overlap=overlap,
            idx_to_class=idx_to_class,
            balance_defects=balance_defects,
            repeat_good_images=repeat_good_images,
            output_folder=output_folder,
            subfolder_name="train",
            train_filename=train_filename,
            annotated_good=annotated_good if annotated_good is None else [class_to_idx[x] for x in annotated_good],
            num_workers=num_workers,
        )

    for phase, split_dict in zip(["val", "test"], [val_data_dictionary, test_data_dictionary], strict=False):
        if len(split_dict) > 0:
            log.info("Generating %s set", phase)
            generate_patch_sliding_window_dataset(
                data_dictionary=split_dict,
                patch_number=patch_number,
                patch_size=patch_size,
                overlap=overlap,
                output_folder=output_folder,
                subfolder_name=phase,
                area_threshold=area_threshold,
                area_defect_threshold=area_defect_threshold,
                mask_extension=mask_extension,
                mask_output_folder=mask_output_folder,
                save_mask=save_mask,
                class_to_idx=class_to_idx,
            )

    log.info("All done! Datasets saved to %s", output_folder)

    return dataset_info


def multilabel_stratification(
    output_folder: str,
    data_dictionary: list[dict],
    num_classes: int,
    val_size: float,
    test_size: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split data dictionary using multilabel based stratification, place every sample with None
        mask inside the test set,for all the others read the labels contained in the masks
        to create one-hot encoded labels.

    Args:
        output_folder: root folder of the dataset
        data_dictionary: Data dictionary as described in generate patch dataset
        num_classes: Number of classes contained in the dataset, required for one hot encoding
        val_size: Percentage of data to be used for validation
        test_size: Percentage of data to be used for test
    Returns:
        Three data dictionaries, one for training, one for validation and one for test

    """
    if val_size + test_size == 0:
        return data_dictionary, [], []
    if val_size == 1:
        return [], data_dictionary, []
    if test_size == 1:
        return [], [], data_dictionary

    test_data_dictionary = list(filter(lambda q: q["mask"] is None, data_dictionary))
    log.info("Number of images with no mask inserted in test_data_dictionary: %d", len(test_data_dictionary))
    empty_test_size = len(test_data_dictionary) / len(data_dictionary)
    data_dictionary = list(filter(lambda q: q["mask"] is not None, data_dictionary))

    if len(data_dictionary) == 0:
        # All the item in the data dictionary have None mask, put everything in test
        warnings.warn(
            "All the images have None mask and the test size is not equal to 1! Put everything in test",
            UserWarning,
            stacklevel=2,
        )
        return [], [], test_data_dictionary

    x = []
    y = None
    for item in data_dictionary:
        one_hot = np.zeros([1, num_classes], dtype=np.int16)
        if item["mask"] is None:
            continue
        # this works even if item["mask"] is already an absolute path
        mask = cv2.imread(os.path.join(output_folder, item["mask"]), 0)

        labels = np.unique(mask)

        one_hot[:, labels] = 1
        x.append(item["base_name"])
        if y is None:
            y = one_hot
        else:
            y = np.concatenate([y, one_hot])

    x_test: list[Any] | np.ndarray

    if empty_test_size > test_size:
        warnings.warn(
            (
                "The percentage of images with None label is greater than the test_size, the newest test_size is"
                f" {empty_test_size}!"
            ),
            UserWarning,
            stacklevel=2,
        )
        x_train, _, x_val, _ = iterative_train_test_split(np.expand_dims(np.array(x), 1), y, val_size)
        x_test = [q["base_name"] for q in test_data_dictionary]
    else:
        test_size -= empty_test_size
        x_train, _, x_remaining, y_remaining = iterative_train_test_split(
            np.expand_dims(np.array(x), 1), y, val_size + test_size
        )

        if x_remaining.shape[0] == 1:
            if test_size == 0:
                x_val = x_remaining
                x_test = np.array([])
            elif val_size == 0:
                x_test = x_remaining
                x_val = np.array([])
            else:
                log.warning("Not enough data to create the test split, only a validation set of size 1 will be created")
                x_val = x_remaining
                x_test = np.array([])
        else:
            x_val, _, x_test, _ = iterative_train_test_split(
                x_remaining, y_remaining, test_size / (val_size + test_size)
            )
        # Here x_test should be always a numpy array, but mypy does not recognize it
        x_test = [q[0] for q in x_test.tolist()]  # type: ignore[union-attr]
        x_test.extend([q["base_name"] for q in test_data_dictionary])

    train_data_dictionary = list(filter(lambda q: q["base_name"] in x_train, data_dictionary))
    val_data_dictionary = list(filter(lambda q: q["base_name"] in x_val, data_dictionary))
    test_data_dictionary = list(filter(lambda q: q["base_name"] in x_test, data_dictionary + test_data_dictionary))

    return train_data_dictionary, val_data_dictionary, test_data_dictionary


def generate_patch_sliding_window_dataset(
    data_dictionary: list[dict],
    subfolder_name: str,
    patch_number: tuple[int, int] | None = None,
    patch_size: tuple[int, int] | None = None,
    overlap: float = 0.0,
    output_folder: str = "extraction_data",
    area_threshold: float = 0.45,
    area_defect_threshold: float = 0.2,
    mask_extension: str = "_mask",
    mask_output_folder: str | None = None,
    save_mask: bool = False,
    class_to_idx: dict | None = None,
) -> None:
    """Giving a data_dictionary as:
    >>> {
    >>>     'base_name': '163931_1_5.jpg',
    >>>     'path': 'extraction_data/1/163931_1_5.jpg',
    >>>     'mask': 'extraction_data/1/163931_1_5_mask.jpg'
    >>>}
    This function will extract the patches and save the file and the mask in subdirectory
    Args:
        data_dictionary: Dictionary as above
        subfolder_name: Name of the subfolder where to save the extracted patches (output_folder/subfolder_name)
        class_to_idx: Dictionary {"defect": value in mask.. }
        output_folder: root_folder where to extract the data
        area_threshold: minimum percentage of defected patch area present in the mask to classify the patch as defect
        area_defect_threshold: minimum percentage of single defect present in the patch to classify the patch as defect
        mask_extension: extension used to assign image to mask
        mask_output_folder: Optional folder in which to save the masks
        save_mask: flag to save the mask
        patch_number: Optional number of patches for each side, required if patch_size is None
        patch_size: Optional dimension of the patch, required if patch_number is None
        overlap: overlap of the patches [0, 1].

    Returns:
        None.

    """
    if save_mask and len(mask_extension) == 0 and mask_output_folder is None:
        raise InvalidParameterCombinationException(
            "If mask output folder is not set you must specify a mask extension in order to save masks!"
        )

    if patch_number is None and patch_size is None:
        raise InvalidParameterCombinationException("One between patch number or patch size must be specified!")

    for data in tqdm(data_dictionary):
        base_id = data.get("base_name")
        base_path = data.get("path")
        base_mask = data.get("mask")

        assert base_id is not None, "Cannot find base id in data_dictionary"
        assert base_path is not None, "Cannot find image in data_dictionary"

        image = cv2.imread(os.path.join(output_folder, base_path))
        h = image.shape[0]
        w = image.shape[1]

        log.debug("Processing %s with shape %s", base_id, image.shape)
        mask = mask_patches = None
        labelled_mask = labelled_patches = None

        if base_mask is not None:
            mask = cv2.imread(os.path.join(output_folder, base_mask), 0)
            labelled_mask = label(mask)

        if patch_size is not None:
            [patch_height, patch_width] = patch_size
            [patch_num_h, patch_num_w], step = compute_patch_info_from_patch_dim(
                h, w, patch_height, patch_width, overlap
            )
        elif patch_number is not None:
            [patch_height, patch_width], step = compute_patch_info(h, w, patch_number[0], patch_number[1], overlap)
            [patch_num_h, patch_num_w] = patch_number
        else:
            # mypy does not recognize that this is unreachable
            raise InvalidParameterCombinationException("One between patch number or patch size must be specified!")

        log.debug(
            "Extracting %s patches with size %s, step %s", [patch_num_h, patch_num_w], [patch_height, patch_width], step
        )
        image_patches = extract_patches(image, (patch_num_h, patch_num_w), (patch_height, patch_width), step, overlap)

        if mask is not None:
            if labelled_mask is None:
                raise ValueError("Labelled mask cannot be None!")
            mask_patches = extract_patches(mask, (patch_num_h, patch_num_w), (patch_height, patch_width), step, overlap)
            labelled_patches = extract_patches(
                labelled_mask, (patch_num_h, patch_num_w), (patch_height, patch_width), step, overlap
            )
            assert image_patches.shape[:-1] == mask_patches.shape, "Image patches and mask patches mismatch!"

        log.debug("Image patches shape: %s", image_patches.shape)
        __save_patch_dataset(
            image_patches=image_patches,
            mask_patches=mask_patches,
            labelled_patches=labelled_patches,
            labelled_mask=labelled_mask,
            image_name=os.path.splitext(base_id)[0],
            output_folder=os.path.join(output_folder, subfolder_name),
            area_threshold=area_threshold,
            area_defect_threshold=area_defect_threshold,
            mask_extension=mask_extension,
            save_mask=save_mask,
            mask_output_folder=mask_output_folder,
            class_to_idx=class_to_idx,
        )


def extract_patches(
    image: np.ndarray,
    patch_number: tuple[int, ...],
    patch_size: tuple[int, ...],
    step: tuple[int, ...],
    overlap: float,
) -> np.ndarray:
    """From an image extract N x M Patch[h, w] if the image is not perfectly divided by the number of patches of given
    dimension the last patch will contain a replica of the original image taken in range [-img_h:, :] or [:, -img_w:].

    Args:
        image: Numpy array of the image
        patch_number: number of patches to be extracted
        patch_size: dimension of the patch
        step: step of the patch extraction
        overlap: horizontal and vertical patch overlapping in range [0, 1]

    Returns:
        Patches [N, M, 1, image_w, image_h, image_c]

    """
    assert 1.0 >= overlap >= 0.0, f"Overlap must be between 0 and 1. Received {overlap}"
    (patch_num_h, patch_num_w) = patch_number
    (patch_height, patch_width) = patch_size

    pad_h = (patch_num_h - 1) * step[0] + patch_size[0] - image.shape[0]
    pad_w = (patch_num_w - 1) * step[1] + patch_size[1] - image.shape[1]
    # if the image has 3 channel change dimension
    if len(image.shape) == 3:
        patch_size = (patch_size[0], patch_size[1], image.shape[2])
        step = (step[0], step[1], image.shape[2])

    # If this is not true there's some strange case I didn't take into account
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Something went wrong with the patch extraction, expected positive padding values")

    if pad_h > 0 or pad_w > 0:
        # We work with copies as view_as_windows returns a view of the original image
        crop_img = deepcopy(image)

        if pad_h:
            crop_img = crop_img[0 : (patch_num_h - 2) * step[0] + patch_height, :]

        if pad_w:
            crop_img = crop_img[:, 0 : (patch_num_w - 2) * step[1] + patch_width]

        # Extract safe patches inside the image
        patches = view_as_windows(crop_img, patch_size, step=step)
    else:
        patches = view_as_windows(image, patch_size, step=step)

    extra_patches_h = None
    extra_patches_w = None

    if pad_h > 0:
        # Append extra patches taken from the edge of the image
        extra_patches_h = view_as_windows(image[-patch_height:, :], patch_size, step=step)

    if pad_w > 0:
        extra_patches_w = view_as_windows(image[:, -patch_width:], patch_size, step=step)

        if extra_patches_h is not None:
            # Add an extra column and set is content to the bottom right patch area of the original image if both
            # dimension requires extra patches
            if extra_patches_h.ndim == 6:
                # RGB
                extra_patches_h = np.concatenate(
                    [
                        extra_patches_h,
                        (np.zeros([1, 1, 1, patch_size[0], patch_size[1], extra_patches_h.shape[-1]], dtype=np.uint8)),
                    ],
                    axis=1,
                )
            else:
                extra_patches_h = np.concatenate(
                    [extra_patches_h, (np.zeros([1, 1, patch_size[0], patch_size[1]], dtype=np.uint8))], axis=1
                )

            if extra_patches_h is None:
                # Required by mypy as it cannot infer that extra_patch_h cannot be None
                raise ValueError("Extra patch h cannot be None!")

            extra_patches_h[:, -1, :] = image[-patch_height:, -patch_width:]

    if patches.ndim == 6:
        # With RGB images there's an extra dimension, axis 2 is important don't use plain squeeze or it breaks if
        # the number of patches is set to 1!
        patches = patches.squeeze(axis=2)

    if extra_patches_w is not None:
        if extra_patches_w.ndim == 6:
            # RGB
            patches = np.concatenate([patches, extra_patches_w.squeeze(2)], axis=1)
        else:
            patches = np.concatenate([patches, extra_patches_w], axis=1)

    if extra_patches_h is not None:
        if extra_patches_h.ndim == 6:
            # RGB
            patches = np.concatenate([patches, extra_patches_h.squeeze(2)], axis=0)
        else:
            patches = np.concatenate([patches, extra_patches_h], axis=0)

    # If this is not true there's some strange case I didn't take into account
    assert patches.shape[0] == patch_num_h and patches.shape[1] == patch_num_w, (
        f"Patch shape {patches.shape} does not match the expected shape {patch_number}"
    )

    return patches


def generate_patch_sampling_dataset(
    data_dictionary: list[dict[Any, Any]],
    output_folder: str,
    idx_to_class: dict,
    overlap: float,
    repeat_good_images: int = 1,
    balance_defects: bool = True,
    patch_number: tuple[int, int] | None = None,
    patch_size: tuple[int, int] | None = None,
    subfolder_name: str = "train",
    train_filename: str = "dataset.txt",
    annotated_good: list[int] | None = None,
    num_workers: int = 1,
) -> None:
    """Generate a dataset of patches.

    Args:
        data_dictionary: Dictionary containing image and mask mapping
        output_folder: root folder
        idx_to_class: Dict mapping an index to the corresponding class name
        repeat_good_images: Number of repetition for images with emtpy or None mask
        balance_defects: If true add one good entry for each defect extracted
        patch_number: Optional number of patches for each side, required if patch_size is None
        patch_size: Optional dimension of the patch, required if patch_number is None
        overlap: Percentage of overlap between patches
        subfolder_name: name of the subfolder where to store h5 files for defected images and dataset txt
        train_filename: Name of the file in which to store the mappings between h5 files and labels
        annotated_good: List of class indices that are considered good other than the background
        num_workers: Number of processes used to create h5 files.

    Returns:
        Create a txt file containing tuples path,label where path is a pointer to the generated h5 file and label is the
            corresponding label

            Each generated h5 file contains five fields:
                img_path: Pointer to the location of the original image
                mask_path: Optional pointer to the mask file, is missing if the mask is completely empty or is
                not present
                patch_size: dimension of the patches on the interested image
                triangles: List of triangles that covers the defect
                triangles_weights: Which weight should be given to each triangle for sampling

    """
    if patch_number is None and patch_size is None:
        raise InvalidParameterCombinationException("One between patch number or patch size must be specified!")

    sampling_dataset_folder = os.path.join(output_folder, subfolder_name)

    os.makedirs(sampling_dataset_folder, exist_ok=True)
    labelled_masks_path = os.path.join(output_folder, "original", "labelled_masks")
    os.makedirs(labelled_masks_path, exist_ok=True)

    with open(os.path.join(sampling_dataset_folder, train_filename), "w") as output_file:
        if num_workers < 1:
            raise InvalidNumWorkersNumberException("Workers must be >= 1")

        if num_workers > 1:
            log.info("Executing generate_patch_sampling_dataset w/ more than 1 worker!")

            split_data_dictionary = np.array_split(np.asarray(data_dictionary), num_workers)

            with Pool(num_workers) as pool:
                res_list = pool.map(
                    partial(
                        create_h5,
                        patch_size=patch_size,
                        patch_number=patch_number,
                        idx_to_class=idx_to_class,
                        overlap=overlap,
                        repeat_good_images=repeat_good_images,
                        balance_defects=balance_defects,
                        annotated_good=annotated_good,
                        output_folder=output_folder,
                        labelled_masks_path=labelled_masks_path,
                        sampling_dataset_folder=sampling_dataset_folder,
                    ),
                    split_data_dictionary,
                )

            res = list(itertools.chain(*res_list))
        else:
            res = create_h5(
                data_dictionary=data_dictionary,
                patch_size=patch_size,
                patch_number=patch_number,
                idx_to_class=idx_to_class,
                overlap=overlap,
                repeat_good_images=repeat_good_images,
                balance_defects=balance_defects,
                annotated_good=annotated_good,
                output_folder=output_folder,
                labelled_masks_path=labelled_masks_path,
                sampling_dataset_folder=sampling_dataset_folder,
            )

        for line in res:
            output_file.write(line)


def create_h5(
    data_dictionary: list[dict[Any, Any]],
    idx_to_class: dict,
    overlap: float,
    repeat_good_images: int,
    balance_defects: bool,
    output_folder: str,
    labelled_masks_path: str,
    sampling_dataset_folder: str,
    annotated_good: list[int] | None = None,
    patch_size: tuple[int, int] | None = None,
    patch_number: tuple[int, int] | None = None,
) -> list[str]:
    """Create h5 files for each image in the dataset.

    Args:
        data_dictionary: Dictionary containing image and mask mapping
        idx_to_class: Dict mapping an index to the corresponding class name
        overlap: Percentage of overlap between patches
        repeat_good_images: Number of repetition for images with emtpy or None mask
        balance_defects: If true add one good entry for each defect extracted
        output_folder: root folder
        overlap: Percentage of overlap between patches
        annotated_good: List of class indices that are considered good other than the background
        labelled_masks_path: paths of labelled masks
        sampling_dataset_folder: folder of the dataset
        patch_size: Dimension of the patch, required if patch_number is None
        patch_number: Number of patches for each side, required if patch_size is None.

    Returns:
        output_list: List of h5 files' names

    """
    if patch_number is None and patch_size is None:
        raise InvalidParameterCombinationException("One between patch number or patch size must be specified!")

    output_list = []
    for item in tqdm(data_dictionary):
        log.debug("Processing %s", item["base_name"])
        # this works even if item["path"] is already an absolute path
        img = cv2.imread(os.path.join(output_folder, item["path"]))

        h = img.shape[0]
        w = img.shape[1]

        mask: np.ndarray
        if item["mask"] is None:
            mask = np.zeros([h, w], dtype=np.uint8)
        else:
            # this works even if item["mask"] is already an absolute path
            mask = cv2.imread(os.path.join(output_folder, item["mask"]), 0)

        if patch_size is not None:
            patch_height = patch_size[1]
            patch_width = patch_size[0]
        else:
            # Mypy complains because patch_number is Optional, but we already checked that it is not None.
            [patch_height, patch_width], _ = compute_patch_info(
                h,
                w,
                patch_number[0],  # type: ignore[index]
                patch_number[1],  # type: ignore[index]
                overlap,
            )

        h5_file_name_good = os.path.join(sampling_dataset_folder, f"{os.path.splitext(item['base_name'])[0]}_good.h5")

        disable_good = False

        with h5py.File(h5_file_name_good, "w") as f:
            f.create_dataset("img_path", data=item["path"])
            f.create_dataset("patch_size", data=np.array([patch_height, patch_width]))

            target = idx_to_class[0]

            if mask.sum() == 0:
                f.create_dataset("triangles", data=np.array([], dtype=np.uint8), dtype=np.uint8)
                f.create_dataset("triangles_weights", data=np.array([], dtype=np.uint8), dtype=np.uint8)

                for _ in range(repeat_good_images):
                    output_list.append(f"{os.path.basename(h5_file_name_good)},{target}\n")

                continue

            binary_mask = (mask > 0).astype(np.uint8)

            # Dilate the defects and take the background
            binary_mask = np.logical_not(cv2.dilate(binary_mask, np.ones([patch_height, patch_width]))).astype(np.uint8)

            temp_binary_mask = deepcopy(binary_mask)
            # Remove the edges of the image as they are unsafe for sampling without padding
            temp_binary_mask[0 : patch_height // 2, :] = 0
            temp_binary_mask[:, 0 : patch_width // 2] = 0
            temp_binary_mask[-patch_height // 2 :, :] = 0
            temp_binary_mask[:, -patch_width // 2 :] = 0

            if temp_binary_mask.sum() != 0:
                # If the mask without the edges is not empty use it, otherwise use the original mask as it is not
                # possible to sample a patch that will not exceed the edges, this must be taken care by the patch
                # sampler used during training
                binary_mask = temp_binary_mask

            # In the case of hx1 or 1xw number of patches we must make sure that the sampling row or the sampling
            # column is empty, if it isn't remove it from the possible sampling area
            if patch_height == img.shape[0]:
                must_clear_indices = np.where(binary_mask.sum(axis=0) != img.shape[0])[0]
                binary_mask[:, must_clear_indices] = 0

            if patch_width == img.shape[1]:
                must_clear_indices = np.where(binary_mask.sum(axis=1) != img.shape[1])[0]
                binary_mask[must_clear_indices, :] = 0

            # If there's no space for sampling good patches skip it
            if binary_mask.sum() == 0:
                disable_good = True
            else:
                triangles, weights = triangulate_region(binary_mask)
                if triangles is None:
                    disable_good = True
                else:
                    log.debug(
                        "Saving %s triangles for %s with label %s",
                        triangles.shape[0],
                        os.path.basename(h5_file_name_good),
                        target,
                    )

                    f.create_dataset("mask_path", data=item["mask"])
                    # Points from extracted triangles should be sufficiently far from all the defects allowing to sample
                    # good patches almost all the time
                    f.create_dataset("triangles", data=triangles, dtype=np.int32)
                    f.create_dataset("triangles_weights", data=weights, dtype=np.float64)

                    # Avoid saving the good h5 file here because otherwise I'll have one more good compared to the
                    # number of defects
                    if not balance_defects:
                        output_list.append(f"{os.path.basename(h5_file_name_good)},{target}\n")

        if disable_good:
            os.remove(h5_file_name_good)

        labelled_mask = label(mask)
        cv2.imwrite(os.path.join(labelled_masks_path, f"{os.path.splitext(item['base_name'])[0]}.png"), labelled_mask)

        real_defects_mask = None

        if annotated_good is not None:
            # Remove true defected area from the good labeled mask
            # If we want this to be even more restrictive we could also include the background as we don't know for sure
            # it will not contain any defects
            real_defects_mask = (~np.isin(mask, [0] + annotated_good)).astype(np.uint8)
            real_defects_mask = cv2.dilate(real_defects_mask, np.ones([patch_height, patch_width])).astype(bool)

        for i in np.unique(labelled_mask):
            if i == 0:
                continue

            current_mask = (labelled_mask == i).astype(np.uint8)
            target_idx = (mask * current_mask).max()

            # When we have good annotations we want to avoid sampling patches containing true defects, to do so we
            # reduce the extraction area based on the area covered by the other defects
            if annotated_good is not None and real_defects_mask is not None and target_idx in annotated_good:
                # a - b = a & ~b
                # pylint: disable=invalid-unary-operand-type
                current_mask = np.bitwise_and(current_mask.astype(bool), ~real_defects_mask).astype(np.uint8)
            else:
                # When dealing with small defects the number of points that will be sampled will be limited and patches
                # will mostly be centered around the defect, to overcome this issue enlarge defect bounding box by 50%
                # of the difference between the patch_size and the defect bb size, we don't do this on good labels to
                # avoid invalidating the reduction applied before.
                props = regionprops(current_mask)[0]
                bbox_size = [props.bbox[2] - props.bbox[0], props.bbox[3] - props.bbox[1]]
                diff_bbox = np.array([max(0, patch_height - bbox_size[0]), max(0, patch_width - bbox_size[1])])

                if diff_bbox[0] != 0:
                    current_mask = cv2.dilate(current_mask, np.ones([diff_bbox[0] // 2, 1]))
                if diff_bbox[1] != 0:
                    current_mask = cv2.dilate(current_mask, np.ones([1, diff_bbox[1] // 2]))

            if current_mask.sum() == 0:
                # If it's not possible to sample a labelled good patch basically
                continue

            temp_current_mask = deepcopy(current_mask)
            # Remove the edges of the image as they are unsafe for sampling without padding
            temp_current_mask[0 : patch_height // 2, :] = 0
            temp_current_mask[:, 0 : patch_width // 2] = 0
            temp_current_mask[-patch_height // 2 :, :] = 0
            temp_current_mask[:, -patch_width // 2 :] = 0

            if temp_current_mask.sum() != 0:
                # If the mask without the edges is not empty use it, otherwise use the original mask as it is not
                # possible to sample a patch that will not exceed the edges, this must be taken care by the patch
                # sampler used during training
                current_mask = temp_current_mask

            triangles, weights = triangulate_region(current_mask)

            if triangles is not None:
                h5_file_name = os.path.join(sampling_dataset_folder, f"{os.path.splitext(item['base_name'])[0]}_{i}.h5")

                target = idx_to_class[target_idx]

                log.debug(
                    "Saving %s triangles for %s with label %s",
                    triangles.shape[0],
                    os.path.basename(h5_file_name),
                    target,
                )

                with h5py.File(h5_file_name, "w") as f:
                    f.create_dataset("img_path", data=item["path"])
                    f.create_dataset("mask_path", data=item["mask"])
                    f.create_dataset("patch_size", data=np.array([patch_height, patch_width]))
                    f.create_dataset("triangles", data=triangles, dtype=np.int32)
                    f.create_dataset("triangles_weights", data=weights, dtype=np.float64)
                    f.create_dataset("labelled_index", data=i, dtype=np.int32)

                if annotated_good is not None and target_idx in annotated_good:
                    # I treat annotate good images exactly the same as I would treat background
                    for _ in range(repeat_good_images):
                        output_list.append(f"{os.path.basename(h5_file_name)},{target}\n")
                else:
                    output_list.append(f"{os.path.basename(h5_file_name)},{target}\n")

                if balance_defects:
                    if not disable_good:
                        output_list.append(f"{os.path.basename(h5_file_name_good)},{idx_to_class[0]}\n")
                    else:
                        log.debug(
                            "Unable to add a good defect for %s, since there's no way to sample good patches",
                            h5_file_name,
                        )
    return output_list


def triangle_area(triangle: np.ndarray) -> float:
    """Compute the area of a triangle defined by 3 points.

    Args:
        triangle: Array of shape 3x2 containing the coordinates of a triangle.

    Returns:
        The area of the triangle

    """
    [y1, x1], [y2, x2], [y3, x3] = triangle
    return abs(0.5 * (((x2 - x1) * (y3 - y1)) - ((x3 - x1) * (y2 - y1))))


def triangulate_region(mask: ndimage) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract from a binary image containing a single roi (with or without holes) a list of triangles
    (and their normalized area) that completely subdivide an approximated polygon defined around mask contours,
    the output can be used to easily sample uniformly points that are almost guarantee to lie inside the roi.

    Args:
        mask: Binary image defining a region of interest

    Returns:
        Tuple containing:
            triangles: a numpy array containing a list of list of vertices (y, x) of the triangles defined over a
                polygon that contains the entire region
            weights: areas of each triangle rescaled (area_i / sum(areas))

    """
    polygon_points, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    if not np.all(hier[:, :, 3] == -1):  # there are holes
        holes = ndimage.binary_fill_holes(mask).astype(np.uint8)
        holes -= mask
        holes = (holes > 0).astype(np.uint8)
        if holes.sum() > 0:  # there are holes
            for hole in regionprops(label(holes)):
                y_hole_center = int(hole.centroid[0])
                mask[y_hole_center] = 0

        polygon_points, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

    final_approx = []

    # Extract a simpler approximation of the contour
    for cnt in polygon_points:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        final_approx.append(approx)

    triangles = None

    for approx in final_approx:
        contours_tripy = [x[0] for x in approx]
        current_triangles = earclip(contours_tripy)

        if len(current_triangles) == 0:
            # This can only happen is a defect is like one pixel wide...
            continue

        current_triangles = np.array([list(x) for x in current_triangles])

        triangles = current_triangles if triangles is None else np.concatenate([triangles, current_triangles])

    if triangles is None:
        return None, None

    # Swap x and y to match cv2
    triangles = triangles[..., ::-1]

    weights = np.array([triangle_area(x) for x in triangles])
    weights = weights / weights.sum()

    return triangles, weights


class InvalidParameterCombinationException(Exception):
    """Exception raised when an invalid combination of parameters is passed to a function."""


class InvalidNumWorkersNumberException(Exception):
    """Exception raised when an invalid number of workers is passed to a function."""


def load_train_file(
    train_file_path: str,
    include_filter: list[str] | None = None,
    exclude_filter: list[str] | None = None,
    class_to_skip: list | None = None,
) -> tuple[list[str], list[str]]:
    """Load a train file and return a list of samples and a list of targets. It is expected that train files will be in
        the same location as the train_file_path.

    Args:
        train_file_path: Training file location
        include_filter: Include only samples that contain one of the element of this list
        exclude_filter: Exclude all samples that contain one of the element of this list
        class_to_skip: if not None, exlude all the samples with labels present in this list.

    Returns:
        List of samples and list of targets

    """
    samples = []
    targets = []

    with open(train_file_path) as f:
        lines = f.read().splitlines()
        for line in lines:
            sample, target = line.split(",")
            if class_to_skip is not None and target in class_to_skip:
                continue
            samples.append(sample)
            targets.append(target)

    include_filter = [] if include_filter is None else include_filter
    exclude_filter = [] if exclude_filter is None else exclude_filter

    valid_samples_indices = [
        i
        for (i, x) in enumerate(samples)
        if (len(include_filter) == 0 or any(f in x for f in include_filter))
        and (len(exclude_filter) == 0 or not any(f in x for f in exclude_filter))
    ]

    samples = [samples[i] for i in valid_samples_indices]
    targets = [targets[i] for i in valid_samples_indices]

    train_folder = os.path.dirname(train_file_path)
    samples = [os.path.join(train_folder, x) for x in samples]

    return samples, targets


def compute_safe_patch_range(sampled_point: int, patch_size: int, image_size: int) -> tuple[int, int]:
    """Computes the safe patch size for the given image size.

    Args:
        sampled_point: the sampled point
        patch_size: the size of the patch
        image_size: the size of the image.

    Returns:
        Tuple containing the safe patch range [left, right] such that
        [sampled_point - left : sampled_point + right] will be within the image size.
    """
    left = patch_size // 2
    right = patch_size // 2

    if sampled_point + right > image_size:
        right = image_size - sampled_point
        left = patch_size - right

    if sampled_point - left < 0:
        left = sampled_point
        right = patch_size - left

    return left, right


def trisample(triangle: np.ndarray) -> tuple[int, int]:
    """Sample a point uniformly in a triangle.

    Args:
        triangle: Array of shape 3x2 containing the coordinates of a triangle.

    Returns:
        Sample point uniformly in the triangle

    """
    [y1, x1], [y2, x2], [y3, x3] = triangle

    r1 = random.random()
    r2 = random.random()

    s1 = math.sqrt(r1)

    x = x1 * (1.0 - s1) + x2 * (1.0 - r2) * s1 + x3 * r2 * s1
    y = y1 * (1.0 - s1) + y2 * (1.0 - r2) * s1 + y3 * r2 * s1

    return int(y), int(x)
