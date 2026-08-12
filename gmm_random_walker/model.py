import numpy as np
from scipy.ndimage import label
from skimage.segmentation import random_walker

from config import LAMBDA, SLICE_NUM
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


def remove_small_seed_components(seed_mask, min_pixels):
    if min_pixels <= 1 or not np.any(seed_mask):
        return seed_mask
    components, count = label(seed_mask)
    sizes = np.bincount(components.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[components]


def normalize_graph_features(image, brain_mask, posterior, posterior_weight):
    features = np.zeros_like(image, dtype=np.float64)
    for channel in range(image.shape[-1]):
        values = image[:, :, channel][brain_mask]
        low, high = np.percentile(values, [1, 99])
        features[:, :, channel] = np.clip(
            (image[:, :, channel] - low) / max(high - low, 1e-8), 0.0, 1.0
        )
    return np.dstack([features, posterior_weight * posterior])


def segment_with_random_walker(
    image,
    brain_mask,
    posterior,
    tumor_seed_threshold=0.65,
    healthy_seed_threshold=0.03,
    min_tumor_seed_pixels=5,
    beta=90.0,
    posterior_weight=1.0,
):
    """Segment the central slice from automatic GMM posterior seeds."""
    tumor_seeds = remove_small_seed_components(
        (posterior >= tumor_seed_threshold) & brain_mask,
        min_tumor_seed_pixels,
    )
    healthy_seeds = (posterior <= healthy_seed_threshold) | (~brain_mask)
    healthy_seeds &= ~tumor_seeds

    if not np.any(tumor_seeds):
        return {
            "prediction": np.zeros_like(brain_mask, dtype=bool),
            "tumor_seeds": tumor_seeds,
            "healthy_seeds": healthy_seeds,
        }

    markers = np.zeros(brain_mask.shape, dtype=np.uint8)
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
    return {
        "prediction": (labels == 2) & brain_mask,
        "tumor_seeds": tumor_seeds,
        "healthy_seeds": healthy_seeds,
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


def evaluate_random_walker_volume(
    vol_num,
    lambda_val=LAMBDA,
    tumor_prior_scale=1.0,
    return_details=False,
    **walker_params,
):
    image, brain_mask, gt_mask = load_and_normalize_slice(
        vol_num, SLICE_NUM, symmetric=False
    )
    healthy = healthy_gmm_joint_likelihood(
        vol_num, lambda_val=lambda_val, symmetric=False
    )
    tumor = tumor_prior_scale * tumor_joint_likelihood(vol_num, symmetric=False)
    posterior = tumor_posterior(healthy, tumor)
    segmentation = segment_with_random_walker(
        image, brain_mask, posterior, **walker_params
    )
    ground_truth = np.any(gt_mask > 0, axis=-1) if gt_mask.ndim == 3 else gt_mask > 0
    metrics = calculate_metrics(segmentation["prediction"], ground_truth)
    if return_details:
        return {
            "volume": vol_num,
            "image": image,
            "brain_mask": brain_mask,
            "ground_truth": ground_truth,
            "posterior": posterior,
            **segmentation,
            **metrics,
        }
    return metrics
