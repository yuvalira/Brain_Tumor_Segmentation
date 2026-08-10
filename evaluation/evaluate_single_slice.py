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


def eval_vol(
    vol_num,
    target_row=None,
    diagnostic_figures=False,
    verbose=False,
    symmetric=False,
    lambda_val=LAMBDA,
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
    """Run one central-slice segmentation and return its Dice and IoU scores.

    ``symmetric=False, lambda_val=0`` is the baseline GMM.
    ``symmetric=False, lambda_val>0`` is the spatial GMM.
    ``symmetric=True, lambda_val>0`` is the spatial GMM with NDI features.
    """
    slice_output = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=symmetric)
    slice_im, brain_mask, gt_mask = slice_output[:3]

    healthy_likelihood = healthy_gmm_joint_likelihood(
        vol_num, lambda_val=lambda_val, symmetric=symmetric
    )
    tumor_likelihoods = tumor_joint_likelihood(vol_num, symmetric=symmetric)
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

    gt_binary = np.any(gt_mask > 0, axis=-1) if gt_mask.ndim == 3 else gt_mask > 0
    pred_mask = (segmentation_mask > 0) & brain_mask.astype(bool)
    intersection = int(np.sum(pred_mask & gt_binary))
    union = int(np.sum(pred_mask | gt_binary))
    pred_size = int(np.sum(pred_mask))
    gt_size = int(np.sum(gt_binary))

    if pred_size == 0 and gt_size == 0:
        dice = iou = 1.0
    else:
        dice = 2.0 * intersection / (pred_size + gt_size)
        iou = intersection / union

    if verbose:
        print(f"Dice: {dice:.3f}, IoU: {iou:.3f}")

    if diagnostic_figures:
        label = model_name or ("spatial_gmm_ndi" if symmetric else "gmm")
        figure_output_path = os.path.join(
            PROJECT_ROOT, "output", "diagnostic_figures", f"{label}_vol{vol_num}"
        )
        os.makedirs(figure_output_path, exist_ok=True)
        visualize_probability(
            slice_im,
            posteriors,
            brain_mask,
            gt_mask,
            os.path.join(figure_output_path, "probability.png"),
        )
        visualize_entropy(
            entropy_map,
            brain_mask,
            os.path.join(figure_output_path, "entropy.png"),
        )
        visualize_sobel_edges(
            sobel_map,
            brain_mask,
            os.path.join(figure_output_path, "edges.png"),
        )
        visualize_contours(
            slice_im,
            posteriors,
            sobel_map,
            brain_mask,
            gt_mask,
            blob_array,
            is_tumor_list,
            os.path.join(figure_output_path, "contours.png"),
        )
        visualize_expansion(
            segmentation_mask,
            slice_im,
            posteriors,
            brain_mask,
            gt_mask,
            blob_array,
            is_tumor_list,
            os.path.join(figure_output_path, "seed_expansion.png"),
        )
        visualize_segmentation(
            segmentation_mask,
            gt_mask,
            brain_mask,
            os.path.join(figure_output_path, "segmentation_results.png"),
        )

    if return_details:
        return {
            "volume": vol_num,
            "dice": dice,
            "iou": iou,
            "intersection": intersection,
            "union": union,
            "pred_size": pred_size,
            "gt_size": gt_size,
            "image": slice_im,
            "brain_mask": brain_mask,
            "ground_truth": gt_binary,
            "prediction": pred_mask,
            "posteriors": posteriors,
        }
    return dice, iou, intersection, union, pred_size, gt_size
