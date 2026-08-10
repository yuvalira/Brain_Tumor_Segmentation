import os

import numpy as np

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


def segment_likelihoods(
    healthy_likelihood,
    tumor_likelihoods,
    brain_mask,
    min_pixels_per_blob=MIN_NUM_PIXELS_PER_BLOB_DEFAULT,
    allow_internal=ALLOW_INTERNAL_CONTOURS,
    binarization_factor=SOBEL_BINARIZATION_OTSU_FACTOR,
    blob_class_threshold=WEIGHTED_POSTERIOR_MEAN_THRESHOLD,
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
    min_pixels_per_blob=MIN_NUM_PIXELS_PER_BLOB_DEFAULT,
    allow_internal=ALLOW_INTERNAL_CONTOURS,
    binarization_factor=SOBEL_BINARIZATION_OTSU_FACTOR,
    blob_class_threshold=WEIGHTED_POSTERIOR_MEAN_THRESHOLD,
    entropy_thresh=ENTROPY_THRESHOLD_DEFAULT,
    posterior_min=POSTERIOR_THRESHOLD_DEFAULT,
    max_expansion_diameter=MAX_EXPANSION_DIAMETER_DEFAULT,
    model_name=None,
    return_details=False,
):
    """Run one central-slice segmentation and return its Dice and IoU scores."""
    slice_output = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=symmetric)
    slice_im, brain_mask, gt_mask = slice_output[:3]
    healthy_likelihood = healthy_gmm_joint_likelihood(
        vol_num, lambda_val=lambda_val, symmetric=symmetric
    )
    tumor_likelihoods = tumor_prior_scale * tumor_joint_likelihood(
        vol_num, symmetric=symmetric
    )
    segmentation = segment_likelihoods(
        healthy_likelihood,
        tumor_likelihoods,
        brain_mask,
        min_pixels_per_blob=min_pixels_per_blob,
        allow_internal=allow_internal,
        binarization_factor=binarization_factor,
        blob_class_threshold=blob_class_threshold,
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
