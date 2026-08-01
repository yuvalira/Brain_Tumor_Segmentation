from collections import deque
import cv2
import numpy as np


def contour_detection(
    sobel_map: np.ndarray, min_pixels_per_blob: int = 150
) -> np.ndarray:
    """Binarizes Sobel map, seals edges, and extracts closed binary blob channels.

    Returns:
        blob_array: ndarray of shape (H, W, N) with N binary blob channels.
    """
    max_val = np.max(sobel_map)
    norm_edges = (
        (sobel_map / max_val * 255).astype(np.uint8)
        if max_val > 0
        else np.zeros_like(sobel_map, dtype=np.uint8)
    )

    _, binary_edges = cv2.threshold(norm_edges, 38, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sealed_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        sealed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_blobs = []
    H, W = sobel_map.shape

    for cnt in contours:
        temp_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        if np.sum(temp_mask > 0) >= min_pixels_per_blob:
            valid_blobs.append(temp_mask.astype(bool))

    if not valid_blobs:
        return np.zeros((H, W, 0), dtype=bool)

    return np.dstack(valid_blobs)


def entropy_weighted_mean(
    blob_mask: np.ndarray, posterior_array: np.ndarray, entropy_map: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Computes entropy-weighted mean posterior vector for a single blob.

    Weight = (1.0 - Entropy).
    """
    weights = (1.0 - entropy_map[blob_mask])[:, np.newaxis]
    probs = posterior_array[blob_mask]  # (K_pixels, 6)

    raw_scores = np.sum(probs * weights, axis=0) / (np.sum(weights) + 1e-12)
    normalized_scores = raw_scores / (np.sum(raw_scores) + 1e-12)

    return raw_scores, normalized_scores


def contour_classification(
    blob_array: np.ndarray,
    posterior_array: np.ndarray,
    entropy_map: np.ndarray,
    blob_class_threshold: float = 0.5,
) -> tuple[np.ndarray, list[bool]]:
    """Scores and filters blobs based on sum of tumor class probabilities.

    Returns:
        classified_blobs: (H, W, K) boolean array of tumor-classified blobs.
        is_tumor_list: Boolean list indicating tumor status for all N input blobs.
    """
    if blob_array.shape[-1] == 0:
        return np.zeros((*blob_array.shape[:2], 0), dtype=bool), []

    num_blobs = blob_array.shape[-1]
    tumor_blobs = []
    is_tumor_list = []

    for i in range(num_blobs):
        mask_i = blob_array[:, :, i]
        _, norm_scores = entropy_weighted_mean(
            mask_i, posterior_array, entropy_map
        )
        tumor_score = np.sum(norm_scores[0:3])

        if tumor_score >= blob_class_threshold:
            tumor_blobs.append(mask_i)
            is_tumor_list.append(True)
        else:
            is_tumor_list.append(False)

    if not tumor_blobs:
        return np.zeros((*blob_array.shape[:2], 0), dtype=bool), is_tumor_list

    return np.dstack(tumor_blobs), is_tumor_list


def ambiguious_space_expansion(
    blob_mask: np.ndarray,
    entropy_map: np.ndarray,
    posterior_array: np.ndarray,
    brain_mask: np.ndarray,
    entropy_thresh: float = 0.25,
    posterior_min: float = 0.05,
    max_expansion_diameter: int = 20,
) -> np.ndarray:
    """Expands a valid tumor blob into surrounding ambiguous space."""
    tumor_sum = np.sum(posterior_array[:, :, 0:3], axis=-1)
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
    max_expansion_diameter: int = 20,
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
        )
        total_segmentation |= expanded_blob

    return total_segmentation