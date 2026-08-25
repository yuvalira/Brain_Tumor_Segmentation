import numpy as np
from config import *
from utilities.utils import load_and_normalize_slice

# Pipeline Modules

from image_processing.compute_entropy import compute_entropy
from image_processing.edge_detection import sobel_edge_detection
from image_processing.contour_detection import contour_detection
from image_processing.contour_classification import contour_classification
from image_processing.seed_expansion import expansion_loop
from image_processing.diagnostic_visualization import produce_diagnostic_figure

from statistical_models.healthy_likelihood import healthy_gmm_joint_likelihood
from statistical_models.tumor_likelihoods import tumor_gmm_joint_likelihood

# Visualization Functions
from image_processing.visualizations import (
    visualize_probability,
    visualize_entropy,
    visualize_sobel_edges,
    visualize_contours,
    visualize_expansion,
    visualize_segmentation,
)


def eval_vol(
    vol_num: int,
    slice_num: int = SLICE_NUM,
    healthy_model_file: str = "healthy_gmm_all_modalities.npz",
    tumor_model_file: str = "tumor_gmm_all_modalities.npz",
    posterior_mean_threshold: float = WEIGHTED_POSTERIOR_MEAN_THRESHOLD_ALL,
    entropy_expansion_threshold: float = ENTROPY_THRESHOLD_ALL,
    posterior_expansion_threshold: float = POSTERIOR_THRESHOLD_ALL,
    min_pixels_per_blob: int = MIN_NUM_PIXELS_PER_BLOB_DEFAULT,
    sobel_binarization_factor: float = SOBEL_BINARIZATION_OTSU_FACTOR,
    allow_internal_contours: bool = ALLOW_INTERNAL_CONTOURS,
    max_expansion_diameter: int = MAX_EXPANSION_DIAMETER_DEFAULT,
    show_plots: bool = True,
    return_details: bool = False,
):
    """
    Runs the statistical GMM and spatial edge-expansion segmentation pipeline on a single slice.

    :return: Dictionary containing the intermediate maps and final binary segmentation mask.
    """
    # -------------------------------------------------------------------------
    # Step 1: Load Slice & Compute Statistical Posteriors
    # -------------------------------------------------------------------------
    image, brain_mask, gt_mask, _ = load_and_normalize_slice(vol_num, slice_num)

    healthy_joint = healthy_gmm_joint_likelihood(
        vol_num=vol_num, filename=healthy_model_file, slice_num=slice_num
    )  # (H, W)
    tumor_joint = tumor_gmm_joint_likelihood(
        vol_num=vol_num, filename=tumor_model_file, slice_num=slice_num
    )  # (H, W, 3)

    # Stack joint probabilities: [Healthy, Class 0, Class 1, Class 2]
    joint_stack = np.dstack([healthy_joint, tumor_joint])
    evidence = np.sum(joint_stack, axis=-1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        posteriors = np.where(evidence > 1e-12, joint_stack / (evidence + 1e-12), 0.0)
    posteriors *= brain_mask[:, :, np.newaxis]

    # -------------------------------------------------------------------------
    # Step 2: Information Theory & Spatial Boundary Extraction
    # -------------------------------------------------------------------------
    # Normalized Shannon entropy map[cite: 1]
    entropy_map = compute_entropy(posteriors, brain_mask)

    # Gradient magnitude of the merged tumor posterior layer[cite: 4]
    edge_map = sobel_edge_detection(posteriors, brain_mask)

    # Extract binary closed blob channels[cite: 3]
    blob_array = contour_detection(
        edge_map,
        brain_mask=brain_mask,
        min_pixels_per_blob=min_pixels_per_blob,
        allow_internal=allow_internal_contours,
        binarization_factor=sobel_binarization_factor,
    )

    # -------------------------------------------------------------------------
    # Step 3: Contour Classification & Seed Expansion
    # -------------------------------------------------------------------------
    # Classify candidate contours using entropy-weighted scoring[cite: 2]
    tumor_blobs, is_tumor_list = contour_classification(
        blob_array,
        posterior_array=posteriors,
        entropy_map=entropy_map,
        blob_class_threshold=posterior_mean_threshold,
    )

    # Region grow valid seeds into adjacent ambiguous high-entropy space[cite: 7]
    final_segmentation = expansion_loop(
        classified_blobs=tumor_blobs,
        entropy_map=entropy_map,
        posterior_array=posteriors,
        brain_mask=brain_mask,
        entropy_thresh=entropy_expansion_threshold,
        posterior_min=posterior_expansion_threshold,
        max_expansion_diameter=max_expansion_diameter,
    )

    # -------------------------------------------------------------------------
    # Step 4: Diagnostic Figures
    # -------------------------------------------------------------------------
    if show_plots:
        # # 1. Posterior probabilities composite[cite: 8]
        # visualize_probability(image, posteriors, brain_mask, gt_mask)

        # # 2. Entropy uncertainty map[cite: 8]
        # visualize_entropy(entropy_map, brain_mask)

        # # 3. Sobel edge detection map[cite: 8]
        # visualize_sobel_edges(edge_map, brain_mask)

        # # 4. Classified contours overlay (Red: Tumor, Green: Non-tumor)[cite: 8]
        # visualize_contours(
        #     image, posteriors, edge_map, brain_mask, gt_mask, blob_array, is_tumor_list
        # )

        # # 5. Seed expansion progression (Yellow: Seed, Magenta: Expanded, Cyan: GT)[cite: 8]
        # visualize_expansion(
        #     final_segmentation, image, posteriors, brain_mask, gt_mask, blob_array, is_tumor_list
        # )

        # # 6. Final pixel-wise segmentation error map (TP, FP, FN, TN)[cite: 8]
        # visualize_segmentation(final_segmentation, gt_mask, brain_mask)

        produce_diagnostic_figure(
        image,
        posteriors,
        brain_mask,
        gt_mask,
        entropy_map,
        edge_map,
        blob_array,
        is_tumor_list,
        final_segmentation,
        save_path=None)

    result = {
        "final_segmentation": final_segmentation,
        "posteriors": posteriors,
        "entropy_map": entropy_map,
        "edge_map": edge_map,
        "blob_array": blob_array,
        "is_tumor_list": is_tumor_list,
        "gt_mask": gt_mask,
        "brain_mask": brain_mask,
    }
    if return_details:
        prediction = final_segmentation.astype(bool)
        ground_truth = np.any(gt_mask > 0, axis=-1) if gt_mask.ndim == 3 else gt_mask > 0
        intersection = int(np.logical_and(prediction, ground_truth).sum())
        union = int(np.logical_or(prediction, ground_truth).sum())
        pred_size = int(prediction.sum())
        gt_size = int(ground_truth.sum())
        if pred_size == 0 and gt_size == 0:
            dice = iou = precision = recall = 1.0
        else:
            dice = 2.0 * intersection / (pred_size + gt_size) if pred_size + gt_size else 0.0
            iou = intersection / union if union else 0.0
            precision = intersection / pred_size if pred_size else 0.0
            recall = intersection / gt_size if gt_size else 0.0
        result.update({
            "image": image,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "intersection": intersection,
            "pred_size": pred_size,
            "gt_size": gt_size,
        })
    return result
