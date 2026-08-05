import cv2
import numpy as np
from config_parameters import *


def contour_detection(
    sobel_map: np.ndarray,
    brain_mask: np.ndarray = None,
    min_pixels_per_blob: int = MIN_NUM_PIXELS_PER_BLOB_DEFAULT,
    allow_internal: bool = True,
) -> np.ndarray:
    """Binarizes Sobel map, seals edges, and extracts closed binary blob channels.

    Args:
        allow_internal: If True, uses RETR_TREE to extract both outer and nested
            internal contours. If False, uses RETR_EXTERNAL for outer boundaries only.
    """
    sobel_work = sobel_map.copy()

    # 1. Mask out background and erode brain mask slightly to remove outer skull boundary
    if brain_mask is not None:
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded_brain = cv2.erode(brain_mask.astype(np.uint8), kernel_erode)
        sobel_work[eroded_brain == 0] = 0.0

    max_val = np.max(sobel_work)
    if max_val == 0:
        return np.zeros((*sobel_map.shape, 0), dtype=bool)

    norm_edges = (sobel_work / max_val * 255).astype(np.uint8)

    # 2. Scaled Otsu thresholding for edge detection
    otsu_val, _ = cv2.threshold(
        norm_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    target_thresh = otsu_val * SOBEL_BINARIZATION_OTSU_FACTOR
    thresh_used, binary_edges = cv2.threshold(
        norm_edges, target_thresh, 255, cv2.THRESH_BINARY
    )

    # 3. Morphological closing to seal edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sealed_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel)

    # 4. Select contour retrieval mode based on allow_internal flag
    retrieval_mode = cv2.RETR_TREE if allow_internal else cv2.RETR_EXTERNAL

    contours, _ = cv2.findContours(
        sealed_edges, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_blobs = []
    H, W = sobel_map.shape

    for cnt in contours:
        temp_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Filter by minimum area
        if np.sum(temp_mask > 0) >= min_pixels_per_blob:
            valid_blobs.append(temp_mask.astype(bool))

    if not valid_blobs:
        return np.zeros((H, W, 0), dtype=bool)

    return np.dstack(valid_blobs)