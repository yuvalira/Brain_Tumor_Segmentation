import os

import numpy as np
from scipy.ndimage import maximum_filter

from config import *
from image_processing.compute_entropy import compute_entropy
from image_processing.contour_classification import contour_classification
from image_processing.contour_detection import contour_detection
from image_processing.edge_detection import sobel_edge_detection
from image_processing.seed_expansion import expansion_loop
from image_processing.visualizations import (
    visualize_contours,
    visualize_entropy,
    visualize_expansion,
    visualize_probability,
    visualize_segmentation,
    visualize_sobel_edges,
)
from statistical_models.healthy_likelihood import healthy_gmm_joint_likelihood
from statistical_models.tumor_likelihoods import tumor_joint_likelihood
from utilities.utils import load_and_normalize_slice


def tumor_posterior(healthy_likelihood, tumor_likelihoods):
    """Return binary whole-tumor posterior from healthy and tumor likelihoods."""
    tumor_sum = np.sum(tumor_likelihoods, axis=-1)
    evidence = healthy_likelihood + tumor_sum
    return np.divide(
        tumor_sum,
        evidence,
        out=np.zeros_like(tumor_sum),
        where=evidence > 0,
    )


def load_z_neighbor_likelihoods(vol_num, z_radius=2):
    """Load global-GMM likelihoods for slices neighboring the central slice."""
    neighbors = []
    for offset in range(-z_radius, z_radius + 1):
        if offset == 0:
            continue
        slice_num = SLICE_NUM + offset
        neighbors.append((
            healthy_gmm_joint_likelihood(
                vol_num, lambda_val=0.0, symmetric=False, slice_num=slice_num
            ),
            tumor_joint_likelihood(vol_num, symmetric=False, slice_num=slice_num),
        ))
    return neighbors


def build_z_context_score(
    healthy_likelihood,
    tumor_likelihoods,
    z_neighbor_likelihoods,
    tumor_prior_scale=1.0,
):
    """Require tumor evidence in at least two of five nearby axial slices.

    A 3x3 maximum filter tolerates small in-plane shifts between slices. The
    second-highest posterior across z is retained, so an isolated response in
    one slice cannot create Z support by itself.
    """
    posterior_maps = [tumor_posterior(healthy_likelihood, tumor_likelihoods)]
    posterior_maps.extend(
        tumor_posterior(healthy, tumor_prior_scale * tumor)
        for healthy, tumor in z_neighbor_likelihoods
    )
    aligned_maps = [maximum_filter(posterior, size=3) for posterior in posterior_maps]
    return np.sort(np.stack(aligned_maps), axis=0)[-2]


def apply_z_context_fusion(
    tumor_likelihoods,
    healthy_likelihood,
    z_context_score,
    z_strength=0.0,
    z_posterior_gate=0.0,
):
    """Boost central-slice tumor evidence only where central and Z evidence agree."""
    if z_strength <= 0:
        return tumor_likelihoods
    central_posterior = tumor_posterior(healthy_likelihood, tumor_likelihoods)
    supported = z_context_score * (central_posterior >= z_posterior_gate)
    return tumor_likelihoods * (1.0 + z_strength * supported[:, :, np.newaxis])


def apply_ndi_fusion(
    tumor_likelihoods,
    ndi_features,
    symmetric_brain_mask,
    ndi_strength=0.0,
    ndi_percentile=90.0,
    healthy_likelihood=None,
    ndi_posterior_gate=0.0,
):
    """Use strong left-right asymmetry as a bounded tumor-likelihood boost."""
    if ndi_strength <= 0:
        return tumor_likelihoods, np.zeros(tumor_likelihoods.shape[:2])

    ndi_score = np.sqrt(np.mean(np.square(ndi_features), axis=-1))
    valid_scores = ndi_score[symmetric_brain_mask]
    if valid_scores.size == 0:
        return tumor_likelihoods, np.zeros_like(ndi_score)

    threshold = np.percentile(valid_scores, ndi_percentile)
    ceiling = np.percentile(valid_scores, 99.5)
    normalized_ndi = np.clip(
        (ndi_score - threshold) / max(ceiling - threshold, 1e-8), 0.0, 1.0
    )
    normalized_ndi *= symmetric_brain_mask
    if healthy_likelihood is not None and ndi_posterior_gate > 0:
        tumor_sum = np.sum(tumor_likelihoods, axis=-1)
        base_posterior = np.divide(
            tumor_sum,
            healthy_likelihood + tumor_sum,
            out=np.zeros_like(tumor_sum),
            where=(healthy_likelihood + tumor_sum) > 0,
        )
        normalized_ndi *= base_posterior >= ndi_posterior_gate
    fused_likelihoods = tumor_likelihoods * (
        1.0 + ndi_strength * normalized_ndi[:, :, np.newaxis]
    )
    return fused_likelihoods, normalized_ndi


def segment_likelihoods(
    healthy_likelihood,
    tumor_likelihoods,
    brain_mask,
    min_pixels_per_blob=MIN_NUM_PIXELS_PER_BLOB_DEFAULT,
    allow_internal=ALLOW_INTERNAL_CONTOURS,
    binarization_factor=SOBEL_BINARIZATION_OTSU_FACTOR,
    blob_class_threshold=WEIGHTED_POSTERIOR_MEAN_THRESHOLD,
    large_contour_min_area=LARGE_CONTOUR_MIN_AREA_DEFAULT,
    top_posterior_mean_threshold=TOP_POSTERIOR_MEAN_THRESHOLD_DEFAULT,
    high_posterior_fraction_threshold=HIGH_POSTERIOR_FRACTION_THRESHOLD_DEFAULT,
    entropy_thresh=ENTROPY_THRESHOLD_DEFAULT,
    posterior_min=POSTERIOR_THRESHOLD_DEFAULT,
    max_expansion_diameter=MAX_EXPANSION_DIAMETER_DEFAULT,
):
    """Convert healthy/tumor likelihood maps into a binary segmentation mask."""
    joint_likelihoods = np.dstack([healthy_likelihood, tumor_likelihoods])
    evidence = np.sum(joint_likelihoods, axis=-1, keepdims=True)
    posteriors = np.divide(
        joint_likelihoods,
        evidence,
        out=np.zeros_like(joint_likelihoods),
        where=evidence > 0,
    )
    entropy_map = compute_entropy(posteriors, brain_mask)
    sobel_map = sobel_edge_detection(posteriors, brain_mask)
    blob_array = contour_detection(
        sobel_map,
        brain_mask=brain_mask,
        min_pixels_per_blob=min_pixels_per_blob,
        allow_internal=allow_internal,
        binarization_factor=binarization_factor,
    )
    classified_blobs, is_tumor_list = contour_classification(
        blob_array,
        posteriors,
        entropy_map,
        blob_class_threshold=blob_class_threshold,
        large_contour_min_area=large_contour_min_area,
        top_posterior_mean_threshold=top_posterior_mean_threshold,
        high_posterior_fraction_threshold=high_posterior_fraction_threshold,
    )
    segmentation_mask = expansion_loop(
        classified_blobs,
        entropy_map,
        posteriors,
        brain_mask,
        entropy_thresh=entropy_thresh,
        posterior_min=posterior_min,
        max_expansion_diameter=max_expansion_diameter,
    )
    return {
        "prediction": (segmentation_mask > 0) & brain_mask.astype(bool),
        "posteriors": posteriors,
        "entropy_map": entropy_map,
        "sobel_map": sobel_map,
        "blob_array": blob_array,
        "classified_blobs": classified_blobs,
        "is_tumor_list": is_tumor_list,
    }


def calculate_metrics(prediction, ground_truth):
    """Calculate segmentation counts and metrics, including empty-empty cases."""
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


def eval_vol(
    vol_num,
    target_row=None,
    diagnostic_figures=False,
    verbose=False,
    symmetric=False,
    lambda_val=LAMBDA,
    tumor_prior_scale=1.0,
    z_strength=0.0,
    z_posterior_gate=0.0,
    ndi_strength=0.0,
    ndi_percentile=90.0,
    ndi_posterior_gate=0.0,
    min_pixels_per_blob=MIN_NUM_PIXELS_PER_BLOB_DEFAULT,
    allow_internal=ALLOW_INTERNAL_CONTOURS,
    binarization_factor=SOBEL_BINARIZATION_OTSU_FACTOR,
    blob_class_threshold=WEIGHTED_POSTERIOR_MEAN_THRESHOLD,
    large_contour_min_area=LARGE_CONTOUR_MIN_AREA_DEFAULT,
    top_posterior_mean_threshold=TOP_POSTERIOR_MEAN_THRESHOLD_DEFAULT,
    high_posterior_fraction_threshold=HIGH_POSTERIOR_FRACTION_THRESHOLD_DEFAULT,
    entropy_thresh=ENTROPY_THRESHOLD_DEFAULT,
    posterior_min=POSTERIOR_THRESHOLD_DEFAULT,
    max_expansion_diameter=MAX_EXPANSION_DIAMETER_DEFAULT,
    model_name=None,
    return_details=False,
):
    """Run one central-slice segmentation and return its Dice and IoU scores."""
    slice_output = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=symmetric)
    features, brain_mask, gt_mask = slice_output[:3]
    slice_im = features[:, :, :4]
    healthy_likelihood = healthy_gmm_joint_likelihood(
        vol_num, lambda_val=lambda_val, symmetric=False
    )
    tumor_likelihoods = tumor_prior_scale * tumor_joint_likelihood(
        vol_num, symmetric=False
    )
    raw_tumor_posterior = tumor_posterior(healthy_likelihood, tumor_likelihoods)
    z_context_score = np.zeros(brain_mask.shape, dtype=np.float64)
    if z_strength > 0:
        z_neighbors = load_z_neighbor_likelihoods(vol_num)
        z_context_score = build_z_context_score(
            healthy_likelihood,
            tumor_likelihoods,
            z_neighbors,
            tumor_prior_scale=tumor_prior_scale,
        )
        tumor_likelihoods = apply_z_context_fusion(
            tumor_likelihoods,
            healthy_likelihood,
            z_context_score,
            z_strength=z_strength,
            z_posterior_gate=z_posterior_gate,
        )
    ndi_score = np.zeros(brain_mask.shape, dtype=np.float64)
    if symmetric:
        tumor_likelihoods, ndi_score = apply_ndi_fusion(
            tumor_likelihoods,
            features[:, :, 4:],
            slice_output[3],
            ndi_strength=ndi_strength,
            ndi_percentile=ndi_percentile,
            healthy_likelihood=healthy_likelihood,
            ndi_posterior_gate=ndi_posterior_gate,
        )
    segmentation = segment_likelihoods(
        healthy_likelihood,
        tumor_likelihoods,
        brain_mask,
        min_pixels_per_blob=min_pixels_per_blob,
        allow_internal=allow_internal,
        binarization_factor=binarization_factor,
        blob_class_threshold=blob_class_threshold,
        large_contour_min_area=large_contour_min_area,
        top_posterior_mean_threshold=top_posterior_mean_threshold,
        high_posterior_fraction_threshold=high_posterior_fraction_threshold,
        entropy_thresh=entropy_thresh,
        posterior_min=posterior_min,
        max_expansion_diameter=max_expansion_diameter,
    )
    gt_binary = np.any(gt_mask > 0, axis=-1) if gt_mask.ndim == 3 else gt_mask > 0
    metrics = calculate_metrics(segmentation["prediction"], gt_binary)

    if verbose:
        print(
            f"Dice: {metrics['dice']:.3f}, IoU: {metrics['iou']:.3f}, "
            f"precision: {metrics['precision']:.3f}, recall: {metrics['recall']:.3f}"
        )

    if diagnostic_figures:
        label = model_name or ("spatial_gmm_ndi" if symmetric else "gmm")
        output_dir = os.path.join(
            PROJECT_ROOT, "output", "diagnostic_figures", f"{label}_vol{vol_num}"
        )
        os.makedirs(output_dir, exist_ok=True)
        visualize_probability(
            slice_im,
            segmentation["posteriors"],
            brain_mask,
            gt_mask,
            os.path.join(output_dir, "probability.png"),
        )
        visualize_entropy(
            segmentation["entropy_map"],
            brain_mask,
            os.path.join(output_dir, "entropy.png"),
        )
        visualize_sobel_edges(
            segmentation["sobel_map"],
            brain_mask,
            os.path.join(output_dir, "edges.png"),
        )
        visualize_contours(
            slice_im,
            segmentation["posteriors"],
            segmentation["sobel_map"],
            brain_mask,
            gt_mask,
            segmentation["blob_array"],
            segmentation["is_tumor_list"],
            os.path.join(output_dir, "contours.png"),
        )
        visualize_expansion(
            segmentation["prediction"],
            slice_im,
            segmentation["posteriors"],
            brain_mask,
            gt_mask,
            segmentation["blob_array"],
            segmentation["is_tumor_list"],
            os.path.join(output_dir, "seed_expansion.png"),
        )
        visualize_segmentation(
            segmentation["prediction"],
            gt_mask,
            brain_mask,
            os.path.join(output_dir, "segmentation_results.png"),
        )

    if return_details:
        return {
            "volume": vol_num,
            "image": slice_im,
            "brain_mask": brain_mask,
            "ground_truth": gt_binary,
            "raw_tumor_posterior": raw_tumor_posterior,
            "z_context_score": z_context_score,
            "ndi_score": ndi_score,
            **segmentation,
            **metrics,
        }
    return (
        metrics["dice"],
        metrics["iou"],
        metrics["intersection"],
        metrics["union"],
        metrics["pred_size"],
        metrics["gt_size"],
    )
