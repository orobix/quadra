import numpy as np
import skimage
from skimage.morphology import medial_axis

from quadra.utils import utils

log = utils.get_logger(__name__)


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    """Smooths for segmentation.

    Args:
        mask: Input mask

    Returns:
        Smoothed mask
    """
    labeled_mask = skimage.measure.label(mask)
    labels = np.arange(0, np.max(labeled_mask) + 1)
    output_mask = np.zeros_like(mask).astype(np.float32)
    for l in labels:
        component_mask = labeled_mask == l
        _, distance = medial_axis(component_mask, return_distance=True)
        component_mask_norm = distance ** (1 / 2.2)
        component_mask_norm = (component_mask_norm - np.min(component_mask_norm)) / (
            np.max(component_mask_norm) - np.min(component_mask_norm)
        )
        output_mask += component_mask_norm
    output_mask = output_mask * mask
    return output_mask
