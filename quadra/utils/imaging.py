from __future__ import annotations

import cv2
import numpy as np


def crop_image(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop an image given a roi in proper format.

    Args:
        image: array of size HxW or HxWxC
        roi: (w_upper_left, h_upper_left, w_bottom_right, h_bottom_right)

    Returns:
        Cropped image based on roi
    """
    return image[roi[1] : roi[3], roi[0] : roi[2]]


def keep_aspect_ratio_resize(image: np.ndarray, size: int = 224, interpolation: int = 1) -> np.ndarray:
    """Resize input image while keeping its aspect ratio."""
    (h, w) = image.shape[:2]

    if h < w:
        height = size
        width = int(w * size / h)
    else:
        width = size
        height = int(h * size / w)

    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    return resized
