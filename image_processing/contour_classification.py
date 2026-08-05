import cv2
import numpy as np
from config_parameters import *

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
    blob_class_threshold = WEIGHTED_POSTERIOR_MEAN_THRESHOLD,
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
        tumor_score = np.sum(norm_scores[-3:])

        if tumor_score >= blob_class_threshold:
            tumor_blobs.append(mask_i)
            is_tumor_list.append(True)
        else:
            is_tumor_list.append(False)

    if not tumor_blobs:
        return np.zeros((*blob_array.shape[:2], 0), dtype=bool), is_tumor_list

    return np.dstack(tumor_blobs), is_tumor_list