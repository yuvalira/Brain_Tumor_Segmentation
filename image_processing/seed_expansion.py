from collections import deque
import cv2
import numpy as np
from config import *



def ambiguious_space_expansion(
    blob_mask: np.ndarray,
    entropy_map: np.ndarray,
    posterior_array: np.ndarray,
    brain_mask: np.ndarray,
    entropy_thresh,
    posterior_min,
    max_expansion_diameter,
) -> np.ndarray:
    """Expands a valid tumor blob into surrounding ambiguous space."""
    tumor_sum = np.sum(posterior_array[:, :, -3:], axis=-1)
    H, W = blob_mask.shape
    expanded_mask = blob_mask.copy()

    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    seed_uint8 = blob_mask.astype(np.uint8)
    eroded_seed = cv2.erode(seed_uint8, kernel_cross)
    boundary_seed = (seed_uint8 ^ eroded_seed).astype(bool)

    queue = deque()
    distance_map = np.full((H, W), -1, dtype=int)

    for y, x in zip(*np.where(boundary_seed)):
        queue.append((y, x))
        distance_map[y, x] = 0

    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    while queue:
        cy, cx = queue.popleft()
        current_dist = distance_map[cy, cx]

        if current_dist >= max_expansion_diameter:
            continue

        for dy, dx in neighbors:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and brain_mask[ny, nx]:
                if not expanded_mask[ny, nx]:
                    if (
                        entropy_map[ny, nx] >= entropy_thresh
                        and tumor_sum[ny, nx] >= posterior_min
                    ):
                        expanded_mask[ny, nx] = True
                        distance_map[ny, nx] = current_dist + 1
                        queue.append((ny, nx))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(
        expanded_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel
    ).astype(bool)


def expansion_loop(
    classified_blobs: np.ndarray,
    entropy_map: np.ndarray,
    posterior_array: np.ndarray,
    brain_mask: np.ndarray,
    entropy_thresh = ENTROPY_THRESHOLD_ALL,
    posterior_min= POSTERIOR_THRESHOLD_ALL,
    max_expansion_diameter = MAX_EXPANSION_DIAMETER_DEFAULT
) -> np.ndarray:
    """Iterates over all classified tumor blobs, expands each, and unions them with logical OR."""
    H, W = entropy_map.shape
    total_segmentation = np.zeros((H, W), dtype=bool)

    if classified_blobs.shape[-1] == 0:
        return total_segmentation

    for i in range(classified_blobs.shape[-1]):
        single_blob = classified_blobs[:, :, i]
        expanded_blob = ambiguious_space_expansion(
            single_blob,
            entropy_map,
            posterior_array,
            brain_mask,
            max_expansion_diameter=max_expansion_diameter,
            entropy_thresh=entropy_thresh,
            posterior_min=posterior_min,
        )
        total_segmentation |= expanded_blob

    return total_segmentation