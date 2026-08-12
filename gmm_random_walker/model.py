import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label
from skimage.segmentation import random_walker

from config import SLICE_NUM
from evaluation.evaluate_single_slice import segment_likelihoods
from statistical_models.healthy_likelihood import healthy_gmm_joint_likelihood
from statistical_models.tumor_likelihoods import tumor_joint_likelihood
from utilities.utils import load_and_normalize_slice


def tumor_posterior(healthy_likelihood, tumor_likelihoods):
    tumor_sum = np.sum(tumor_likelihoods, axis=-1)
    evidence = healthy_likelihood + tumor_sum
    return np.divide(
        tumor_sum,
        evidence,
        out=np.zeros_like(tumor_sum),
        where=evidence > 0,
    )


def remove_small_components(mask, min_pixels):
    if min_pixels <= 1 or not np.any(mask):
        return mask
    components, _ = label(mask)
    sizes = np.bincount(components.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[components]


def spatial_gmm_segmentation(healthy, tumor, brain_mask, spatial_params):
    """Run the original central-slice Spatial-GMM post-processing unchanged."""
    return segment_likelihoods(
        healthy,
        tumor,
        brain_mask,
        min_pixels_per_blob=spatial_params["min_pixels_per_blob"],
        binarization_factor=spatial_params["binarization_factor"],
        blob_class_threshold=spatial_params["blob_class_threshold"],
        large_contour_min_area=spatial_params["large_contour_min_area"],
        top_posterior_mean_threshold=spatial_params["top_posterior_mean_threshold"],
        high_posterior_fraction_threshold=spatial_params[
            "high_posterior_fraction_threshold"
        ],
        entropy_thresh=spatial_params["entropy_thresh"],
        posterior_min=spatial_params["posterior_min"],
        max_expansion_diameter=spatial_params["max_expansion_diameter"],
    )["prediction"]


def normalize_graph_features(image, brain_mask, posterior, posterior_weight):
    features = np.zeros_like(image, dtype=np.float64)
    for channel in range(image.shape[-1]):
        values = image[:, :, channel][brain_mask]
        low, high = np.percentile(values, [1, 99])
        features[:, :, channel] = np.clip(
            (image[:, :, channel] - low) / max(high - low, 1e-8), 0.0, 1.0
        )
    return np.dstack([features, posterior_weight * posterior])


def filter_added_components(added_mask, posterior, min_mean_posterior):
    components, count = label(added_mask)
    accepted = np.zeros_like(added_mask, dtype=bool)
    for component_index in range(1, count + 1):
        component = components == component_index
        if posterior[component].mean() >= min_mean_posterior:
            accepted |= component
    return accepted


def segment_with_hybrid_random_walker(
    image,
    brain_mask,
    posterior,
    base_prediction,
    rescue_seed_threshold=0.65,
    healthy_seed_threshold=0.05,
    min_rescue_seed_pixels=5,
    roi_dilation=10,
    base_seed_erosion=2,
    beta=90.0,
    posterior_weight=1.0,
    min_added_component_mean_posterior=0.08,
):
    """Add locally supported Random-Walker pixels to the Spatial-GMM mask.

    The original mask is always preserved, so the hybrid cannot introduce a
    new completely missed tumor that the Spatial GMM already detected.
    Setting roi_dilation=0 provides an exact no-op validation fallback.
    """
    base_prediction = base_prediction.astype(bool) & brain_mask
    empty = np.zeros_like(brain_mask, dtype=bool)
    if roi_dilation <= 0:
        return {
            "prediction": base_prediction.copy(),
            "added_prediction": empty,
            "base_tumor_seeds": empty,
            "rescue_tumor_seeds": empty,
            "healthy_seeds": empty,
            "random_walker_roi": empty,
        }

    base_seeds = (
        binary_erosion(base_prediction, iterations=base_seed_erosion, border_value=0)
        if base_seed_erosion > 0
        else base_prediction.copy()
    )
    if np.any(base_prediction) and not np.any(base_seeds):
        base_seeds = base_prediction.copy()

    rescue_seeds = remove_small_components(
        (posterior >= rescue_seed_threshold) & brain_mask & (~base_prediction),
        min_rescue_seed_pixels,
    )
    tumor_seeds = base_seeds | rescue_seeds
    if not np.any(tumor_seeds):
        return {
            "prediction": base_prediction.copy(),
            "added_prediction": empty,
            "base_tumor_seeds": base_seeds,
            "rescue_tumor_seeds": rescue_seeds,
            "healthy_seeds": empty,
            "random_walker_roi": empty,
        }

    roi = binary_dilation(
        base_prediction | rescue_seeds, iterations=roi_dilation
    ) & brain_mask
    roi_boundary = roi & (~binary_erosion(roi, iterations=1, border_value=0))
    healthy_seeds = (
        ((posterior <= healthy_seed_threshold) & roi) | roi_boundary
    ) & (~tumor_seeds)

    markers = np.full(brain_mask.shape, -1, dtype=np.int8)
    markers[roi] = 0
    markers[healthy_seeds] = 1
    markers[tumor_seeds] = 2
    graph_features = normalize_graph_features(
        image, brain_mask, posterior, posterior_weight
    )
    labels = random_walker(
        graph_features,
        markers,
        beta=beta,
        mode="cg_j",
        channel_axis=-1,
    )
    added = (labels == 2) & roi & (~base_prediction)
    added = filter_added_components(
        added, posterior, min_added_component_mean_posterior
    )
    return {
        "prediction": base_prediction | added,
        "added_prediction": added,
        "base_tumor_seeds": base_seeds,
        "rescue_tumor_seeds": rescue_seeds,
        "healthy_seeds": healthy_seeds,
        "random_walker_roi": roi,
    }


def calculate_metrics(prediction, ground_truth):
    intersection = int(np.sum(prediction & ground_truth))
    union = int(np.sum(prediction | ground_truth))
    pred_size = int(np.sum(prediction))
    gt_size = int(np.sum(ground_truth))
    if pred_size == 0 and gt_size == 0:
        dice = iou = precision = recall = 1.0
    else:
        dice = 2.0 * intersection / (pred_size + gt_size)
        iou = intersection / union
        precision = intersection / pred_size if pred_size else 0.0
        recall = intersection / gt_size if gt_size else 0.0
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "intersection": intersection,
        "union": union,
        "pred_size": pred_size,
        "gt_size": gt_size,
    }


def evaluate_hybrid_volume(
    vol_num,
    spatial_params,
    return_details=False,
    **hybrid_params,
):
    image, brain_mask, gt_mask = load_and_normalize_slice(
        vol_num, SLICE_NUM, symmetric=False
    )
    healthy = healthy_gmm_joint_likelihood(
        vol_num, lambda_val=spatial_params["lambda_val"], symmetric=False
    )
    tumor = spatial_params["tumor_prior_scale"] * tumor_joint_likelihood(
        vol_num, symmetric=False
    )
    posterior = tumor_posterior(healthy, tumor)
    base_prediction = spatial_gmm_segmentation(
        healthy, tumor, brain_mask, spatial_params
    )
    hybrid = segment_with_hybrid_random_walker(
        image, brain_mask, posterior, base_prediction, **hybrid_params
    )
    ground_truth = np.any(gt_mask > 0, axis=-1) if gt_mask.ndim == 3 else gt_mask > 0
    metrics = calculate_metrics(hybrid["prediction"], ground_truth)
    base_metrics = calculate_metrics(base_prediction, ground_truth)
    if return_details:
        return {
            "volume": vol_num,
            "image": image,
            "brain_mask": brain_mask,
            "ground_truth": ground_truth,
            "posterior": posterior,
            "base_prediction": base_prediction,
            "base_dice": base_metrics["dice"],
            **hybrid,
            **metrics,
        }
    return metrics
