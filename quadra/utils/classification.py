import os
import random
import re
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from quadra.utils import utils
from quadra.utils.visualization import UnNormalize, plot_classification_results

log = utils.get_logger(__name__)


def get_file_condition(
    file_name: str, root: str, exclude_filter: Optional[List[str]] = None, include_filter: Optional[List[str]] = None
):
    """Check if a file should be included or excluded based on the filters provided.

    Args:
        file_name: Name of the file
        root: Root directory of the file
        exclude_filter: List of string filter to be used to exclude images. If None no filter will be applied.
        include_filter: List of string filter to be used to include images. If None no filter will be applied.
    """
    if exclude_filter is not None:
        if any(fil in file_name for fil in exclude_filter):
            return False

        if any(fil in root for fil in exclude_filter):
            return False

    if include_filter is not None:
        if not any(fil in file_name for fil in include_filter) and not any(fil in root for fil in include_filter):
            return False

    return True


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def find_images_and_targets(
    folder: str,
    types: Optional[list] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
    leaf_name_only: bool = True,
    sort: bool = True,
    exclude_filter: Optional[list] = None,
    include_filter: Optional[list] = None,
    label_map: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Given a folder, extract the absolute path of all the files with a valid extension.
    Then assign a label based on subfolder name.

    Args:
        folder: path to main folder
        types: valid file extentions
        class_to_idx: dictionary of conversion btw folder name and index.
            Only file whose label is in dictionary key list will be considered. If None all files will
            be considered and a custom conversion is created.
        leaf_name_only: if True use only the leaf folder name as label, otherwise use the full path
        sort: if True sort the images and labels based on the image name
        exclude_filter: list of string filter to be used to exclude images.
            If None no filter will be applied.
        include_filter: list of string filder to be used to include images.
            Only images that satisfied at list one of the filter will be included.
        label_map: dictionary of conversion btw folder name and label.
    """
    if types is None:
        types = [".png", ".jpg", ".jpeg", ".bmp"]
    labels = []
    filenames = []

    for root, _, files in os.walk(folder, topdown=False, followlinks=True):
        if root != folder:
            rel_path = os.path.relpath(root, folder)
        else:
            rel_path = ""

        if leaf_name_only:
            label = os.path.basename(rel_path)
        else:
            aa = rel_path.split(os.path.sep)
            if len(aa) == 2:
                aa = aa[-1:]
            else:
                aa = aa[-2:]
            label = "_".join(aa)  # rel_path.replace(os.path.sep, "_")
            # label = rel_path.replace(os.path.sep, "_")

        for f in files:
            if not get_file_condition(
                file_name=f, root=root, exclude_filter=exclude_filter, include_filter=include_filter
            ):
                continue

            if f.startswith(".") or "checkpoint" in f:
                continue
            _, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)

    if label_map is not None:
        labels, _ = group_labels(labels, label_map)

    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {str(c): idx for idx, c in enumerate(sorted_labels)}

    images_and_targets = [(f, l) for f, l in zip(filenames, labels) if l in class_to_idx]

    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))

    return np.array(images_and_targets)[:, 0], np.array(images_and_targets)[:, 1], class_to_idx


def find_test_image(
    folder: str,
    types: Optional[List[str]] = None,
    exclude_filter: Optional[List[str]] = None,
    include_filter: Optional[List[str]] = None,
    include_none_class: bool = True,
    test_split_file: Optional[str] = None,
    label_map=None,
) -> Tuple[List[str], List[Optional[str]]]:
    """Given a path extract images and labels with filters, labels are based on the parent folder name of the images
    Args:
        folder: root directory containing the images
        types: only choose images with the extensions specified, if None use default extensions
        exclude_filter: list of string filter to be used to exclude images. If None no filter will be applied.
        include_filter: list of string filter to be used to include images. If None no filter will be applied.
        include_none_class: if set to True convert all 'None' labels to None, otherwise ignore the image
        test_split_file: if defined use the split defined inside the file
    Returns:
        Two lists, one containing the images path and the other one containing the labels. Labels can be None.
    """
    if types is None:
        types = [".png", ".jpg", ".jpeg", ".bmp"]

    labels = []
    filenames = []

    for root, _, files in os.walk(folder, topdown=False, followlinks=True):
        if root != folder:
            rel_path = os.path.relpath(root, folder)
        else:
            rel_path = ""
        label: Optional[str] = os.path.basename(rel_path)
        for f in files:
            if not get_file_condition(
                file_name=f, root=root, exclude_filter=exclude_filter, include_filter=include_filter
            ):
                continue
            if f.startswith(".") or "checkpoint" in f:
                continue
            _, ext = os.path.splitext(f)
            if ext.lower() in types:
                if label == "None":
                    if include_none_class:
                        label = None
                    else:
                        continue
                filenames.append(os.path.join(root, f))
                labels.append(label)

    if test_split_file is not None:
        if not os.path.isabs(test_split_file):
            log.info(
                "test_split_file is not an absolute path. Trying to using folder argument %s as parent folder", folder
            )
            test_split_file = os.path.join(folder, test_split_file)

        if not os.path.exists(test_split_file):
            raise FileNotFoundError(f"test_split_file {test_split_file} does not exist")

        with open(test_split_file, "r") as test_file:
            test_split = test_file.read().splitlines()

        file_samples = []
        for row in test_split:
            csv_values = row.split(",")
            if len(csv_values) == 1:
                # ensuring backward compatibility with old split file format
                # old_format: sample, new_format: sample,class
                sample_path = os.path.join(folder, csv_values[0])
            else:
                sample_path = os.path.join(folder, ",".join(csv_values[:-1]))

            file_samples.append(sample_path)

        test_split = [os.path.join(folder, sample.strip()) for sample in file_samples]
        labels = [t for s, t in zip(filenames, labels) if s in file_samples]
        filenames = [s for s in filenames if s in file_samples]
        log.info("Selected %d images using test_split_file for the test", len(filenames))
        if len(filenames) != len(file_samples):
            log.warning(
                "test_split_file contains %d images but only %d images were found in the folder."
                "This may be due to duplicate lines in the test_split_file.",
                len(file_samples),
                len(filenames),
            )
    else:
        log.info("No test_split_file. Selected all %s images for the test", folder)

    if label_map is not None:
        labels, _ = group_labels(labels, label_map)

    return filenames, labels


def group_labels(
    labels: Sequence[Optional[str]], class_mapping: Dict[str, Union[Optional[str], List[str]]]
) -> Tuple[List, Dict]:
    """Group labels based on class_mapping.

    Raises:
        ValueError: if a label is not in class_mapping
        ValueError: if a label is in class_mapping but has no corresponding value

    Returns:
       List of labels and a dictionary of labels and their corresponding group

    Example:
        ```python
        grouped_labels, class_to_idx = group_labels(labels, class_mapping={"Good": "A", "Bad": None})
        assert grouped_labels.count("Good") == labels.count("A")
        assert len(class_to_idx.keys()) == 2

        grouped_labels, class_to_idx = group_labels(labels, class_mapping={"Good": "A", "Defect": "B", "Bad": None})
        assert grouped_labels.count("Bad") == labels.count("C") + labels.count("D")
        assert len(class_to_idx.keys()) == 3

        grouped_labels, class_to_idx = group_labels(labels, class_mapping={"Good": "A", "Bad": ["B", "C", "D"]})
        assert grouped_labels.count("Bad") == labels.count("B") + labels.count("C") + labels.count("D")
        assert len(class_to_idx.keys()) == 2
        ```
    """
    grouped_labels = []
    specified_targets = [k for k in class_mapping.keys() if class_mapping[k] is not None]
    non_specified_targets = [k for k in class_mapping.keys() if class_mapping[k] is None]
    if len(non_specified_targets) > 1:
        raise ValueError(f"More than one non specified target: {non_specified_targets}")
    for label in labels:
        found = False
        for target in specified_targets:
            if not found:
                current_mapping = class_mapping[target]
                if current_mapping is None:
                    continue

                if any(label in list(related_label) for related_label in current_mapping if related_label is not None):
                    grouped_labels.append(target)
                    found = True
        if not found:
            if len(non_specified_targets) > 0:
                grouped_labels.append(non_specified_targets[0])
            else:
                raise ValueError(f"No target found for label: {label}")
    class_to_idx = {k: i for i, k in enumerate(class_mapping.keys())}
    return grouped_labels, class_to_idx


def filter_with_file(list_of_full_paths: List[str], file_path: str, root_path: str) -> Tuple[List[str], List[bool]]:
    """Filter a list of items using a file containing the items to keep. Paths inside file
    should be relative to root_path not absolute to avoid user related issues.

    Args:
        list_of_full_paths: list of items to filter
        file_path: path to the file containing the items to keep
        root_path: root path of the dataset

    Returns:
        list of items to keep
        the mask list to apply different lists later.
    """
    filtered_full_paths = []
    filter_mask = []

    with open(file_path, "r") as f:
        for relative_path in f.read().splitlines():
            full_path = os.path.join(root_path, relative_path)
            if full_path in list_of_full_paths:
                filtered_full_paths.append(full_path)
                filter_mask.append(True)
            else:
                filter_mask.append(False)

    return filtered_full_paths, filter_mask


def get_split(
    image_dir: str,
    exclude_filter: Optional[List[str]] = None,
    include_filter: Optional[List[str]] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    class_to_idx: Optional[Dict[str, int]] = None,
    label_map: Optional[Dict] = None,
    n_splits: int = 1,
    include_none_class: bool = False,
    limit_training_data: Optional[int] = None,
    train_split_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Generator[List, None, None], Dict]:
    """Given a folder, extract the absolute path of all the files with a valid extension and name
    and split them into train/test.

    Args:
        image_dir: Path to the folder containing the images
        exclude_filter: List of file name filter to be excluded: If None no filter will be applied
        include_filter: List of file name filter to be included: If None no filter will be applied
        test_size: Percentage of data to be used for test
        random_state: Random state to be used for reproducibility
        class_to_idx: Dictionary of conversion btw folder name and index.
            Only file whose label is in dictionary key list will be considered.
            If None all files will be considered and a custom conversion is created.
        label_map: Dictionary of conversion btw folder name and label.
        n_splits: Number of dataset subdivision (default 1 -> train/test)
        include_none_class: If set to True convert all 'None' labels to None
        limit_training_data: If set to a value, limit the number of training samples to this value
        train_split_file: If set to a path, use the file to split the dataset
    """
    # TODO: Why is include_none_class not used?
    # pylint: disable=unused-argument
    assert os.path.isdir(image_dir), f"Folder {image_dir} does not exist."
    # Get samples and target
    samples, targets, class_to_idx = find_images_and_targets(
        folder=image_dir,
        exclude_filter=exclude_filter,
        include_filter=include_filter,
        class_to_idx=class_to_idx,
        label_map=label_map
        # include_none_class=include_none_class,
    )

    cl, counts = np.unique(targets, return_counts=True)

    for num, _cl in zip(counts, cl):
        if num == 1:
            to_remove = np.where(np.array(targets) == _cl)[0][0]
            samples = np.delete(np.array(samples), to_remove)
            targets = np.delete(np.array(targets), to_remove)
            class_to_idx.pop(_cl)

    if train_split_file is not None:
        with open(train_split_file, "r") as f:
            train_split = f.read().splitlines()

        file_samples = []
        for row in train_split:
            csv_values = row.split(",")

            if len(csv_values) == 1:
                # ensuring backward compatibility with the old split file format
                # old_format: sample, new_format: sample,class
                sample_path = os.path.join(image_dir, csv_values[0])
            else:
                sample_path = os.path.join(image_dir, ",".join(csv_values[:-1]))

            file_samples.append(sample_path)

        train_split = [os.path.join(image_dir, sample.strip()) for sample in file_samples]
        targets = np.array([t for s, t in zip(samples, targets) if s in file_samples])
        samples = np.array([s for s in samples if s in file_samples])

    if limit_training_data is not None:
        idx_to_keep = []
        for cl in np.unique(targets):
            cl_idx = np.where(np.array(targets) == cl)[0].tolist()
            random.seed(random_state)
            random.shuffle(cl_idx)
            idx_to_keep.extend(cl_idx[:limit_training_data])

        samples = np.asarray([samples[i] for i in idx_to_keep])
        targets = np.asarray([targets[i] for i in idx_to_keep])

    _, counts = np.unique(targets, return_counts=True)

    if n_splits == 1:
        split_technique = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    else:
        split_technique = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    split = split_technique.split(samples, targets)

    return np.array(samples), np.array(targets), split, class_to_idx


def save_classification_result(
    results: pd.DataFrame,
    output_folder: str,
    confmat: pd.DataFrame,
    accuracy: float,
    test_dataloader: DataLoader,
    config: DictConfig,
    output: DictConfig,
    grayscale_cams: Optional[np.ndarray] = None,
):
    """Save csv results, confusion matrix and example images.

    Args:
        results: Dataframe containing the results
        output_folder: Path to the output folder
        confmat: Confusion matrix in a pandas dataframe
        accuracy: Accuracy of the model
        test_dataloader: Dataloader used for testing
        config: Configuration file
        output: Output configuration
        grayscale_cams: List of grayscale grad_cam outputs ordered as the results
    """
    # Save csv
    results.to_csv(os.path.join(output_folder, "test_results.csv"), index_label="index")
    if grayscale_cams is None:
        save_gradcams = False
    else:
        log.info("Plotting original and gradcam examples")
        save_gradcams = True

    # Save confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(confmat),
        display_labels=[x.replace("pred:", "") for x in confmat.columns.to_list()],
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

    if output is not None and output.example:
        log.info("Saving discordant/concordant examples in test folder")
        idx_to_class = test_dataloader.dataset.idx_to_class  # type: ignore[attr-defined]

        # Get misclassified samples
        images_folder = os.path.join(output_folder, "example")
        if not os.path.isdir(images_folder):
            os.makedirs(images_folder)
        original_images_folder = os.path.join(images_folder, "original")
        if not os.path.isdir(original_images_folder):
            os.makedirs(original_images_folder)

        gradcam_folder = os.path.join(images_folder, "gradcam")
        if save_gradcams:
            if not os.path.isdir(gradcam_folder):
                os.makedirs(gradcam_folder)

        for v in np.unique([results["real_label"], results["pred_label"]]):
            if np.isnan(v):
                continue

            k = idx_to_class[v]
            plot_classification_results(
                test_dataloader.dataset,
                unorm=UnNormalize(mean=config.transforms.mean, std=config.transforms.std),
                pred_labels=results["pred_label"].to_numpy(),
                test_labels=results["real_label"].to_numpy(),
                grayscale_cams=grayscale_cams,
                class_name=k,
                original_folder=original_images_folder,
                gradcam_folder=gradcam_folder,
                idx_to_class=idx_to_class,
                pred_class_to_plot=v,
                what="con",
                rows=output.get("rows", 3),
                cols=output.get("cols", 2),
                figsize=output.get("figsize", (20, 20)),
                gradcam=save_gradcams,
            )

            plot_classification_results(
                test_dataloader.dataset,
                unorm=UnNormalize(mean=config.transforms.mean, std=config.transforms.std),
                pred_labels=results["pred_label"].to_numpy(),
                test_labels=results["real_label"].to_numpy(),
                grayscale_cams=grayscale_cams,
                class_name=k,
                original_folder=original_images_folder,
                gradcam_folder=gradcam_folder,
                idx_to_class=idx_to_class,
                pred_class_to_plot=v,
                what="dis",
                rows=output.get("rows", 3),
                cols=output.get("cols", 2),
                figsize=output.get("figsize", (20, 20)),
                gradcam=save_gradcams,
            )

    else:
        log.info("Not generating discordant/concordant examples. Check task:output:example in config file")


def get_results(
    test_labels: Union[np.ndarray, List[int]],
    pred_labels: Union[np.ndarray, List[int]],
    idx_to_labels: Optional[Dict] = None,
    cl_rep_digits: int = 3,
) -> Tuple[Union[str, Dict], pd.DataFrame, float]:
    """Get prediction results from predicted and test labels.

    Args:
        test_labels : test labels
        pred_labels : predicted labels
        idx_to_labels : dictionary mapping indices to labels
        cl_rep_digits : number of digits to use in the classification report. Default: 3

    Returns:
        A tuple that contains classification report as dictionary, `cm` is a pd.Dataframe representing
        the Confusion Matrix, acc is the computed accuracy
    """
    unique_labels = np.unique([test_labels, pred_labels])
    cl_rep = classification_report(
        y_true=test_labels,
        y_pred=pred_labels,
        labels=unique_labels,
        digits=cl_rep_digits,
        zero_division=0,
    )

    cm = confusion_matrix(y_true=test_labels, y_pred=pred_labels, labels=unique_labels)

    acc = accuracy_score(y_true=test_labels, y_pred=pred_labels)

    if idx_to_labels:
        pd_cm = pd.DataFrame(
            cm,
            index=[f"true:{idx_to_labels[x]}" for x in unique_labels],
            columns=[f"pred:{idx_to_labels[x]}" for x in unique_labels],
        )
    else:
        pd_cm = pd.DataFrame(
            cm,
            index=[f"true:{x}" for x in unique_labels],
            columns=[f"pred:{x}" for x in unique_labels],
        )
    return cl_rep, pd_cm, acc
