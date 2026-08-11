import numpy as np
from config import *


def entropy_weighted_mean(
    blob_mask: np.ndarray,
    posterior_array: np.ndarray,
    entropy_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the entropy-weighted mean posterior vector for one contour."""
    weights = (1.0 - entropy_map[blob_mask])[:, np.newaxis]
    probabilities = posterior_array[blob_mask]
    raw_scores = np.sum(probabilities * weights, axis=0) / (np.sum(weights) + 1e-12)
    normalized_scores = raw_scores / (np.sum(raw_scores) + 1e-12)
    return raw_scores, normalized_scores


def contour_classification(
    blob_array: np.ndarray,
    posterior_array: np.ndarray,
    entropy_map: np.ndarray,
    blob_class_threshold=WEIGHTED_POSTERIOR_MEAN_THRESHOLD,
    large_contour_min_area=LARGE_CONTOUR_MIN_AREA_DEFAULT,
    top_posterior_mean_threshold=TOP_POSTERIOR_MEAN_THRESHOLD_DEFAULT,
    high_posterior_fraction_threshold=HIGH_POSTERIOR_FRACTION_THRESHOLD_DEFAULT,
) -> tuple[np.ndarray, list[bool]]:
    """Accept contours using the original score or concentrated tumor evidence.

    A contour is accepted when its entropy-weighted tumor score reaches the
    normal threshold. Large heterogeneous contours can also be accepted when
    their strongest 20% of pixels and high-posterior pixel fraction both show
    sufficient tumor evidence.
    """
    if blob_array.shape[-1] == 0:
        return np.zeros((*blob_array.shape[:2], 0), dtype=bool), []

    tumor_posterior = np.sum(posterior_array[:, :, -3:], axis=-1)
    tumor_blobs = []
    is_tumor_list = []

    for index in range(blob_array.shape[-1]):
        contour = blob_array[:, :, index]
        _, normalized_scores = entropy_weighted_mean(
            contour, posterior_array, entropy_map
        )
        weighted_score = np.sum(normalized_scores[-3:])
        pixel_posteriors = tumor_posterior[contour]
        top_count = max(1, int(np.ceil(0.2 * pixel_posteriors.size)))
        top_mean = np.partition(pixel_posteriors, -top_count)[-top_count:].mean()
        high_fraction = np.mean(pixel_posteriors >= blob_class_threshold)

        accepted = weighted_score >= blob_class_threshold or (
            pixel_posteriors.size >= large_contour_min_area
            and top_mean >= top_posterior_mean_threshold
            and high_fraction >= high_posterior_fraction_threshold
        )
        is_tumor_list.append(bool(accepted))
        if accepted:
            tumor_blobs.append(contour)

    if not tumor_blobs:
        return np.zeros((*blob_array.shape[:2], 0), dtype=bool), is_tumor_list
    return np.dstack(tumor_blobs), is_tumor_list
