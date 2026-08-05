import os
from pathlib import Path
import numpy as np

from statistical_models.tumor_single_gaussian.tumor_single_gaussian_posterior import tumor_single_gaussian_stat_inference
from statistical_models.healthy_single_gaussian.healthy_single_gaussian_posterior import healthy_single_gaussian_stat_inference

from image_processing.visualizations          import visualize_probability, visualize_entropy, visualize_sobel_edges, visualize_contours, visualize_expansion, visualize_segmentation
from image_processing.compute_entropy         import compute_entropy
from image_processing.edge_detection          import sobel_edge_detection
from image_processing.contour_detection       import contour_detection
from image_processing.contour_classification  import contour_classification
from image_processing.seed_expansion          import expansion_loop

from utilities.utils import load_and_normalize_slice


def eval_vol(vol_num):

    num_slices_per_volume = 155

    # 1. Initialize arrays with explicit size
    pred_AND_gt_arr = np.zeros(num_slices_per_volume, dtype=np.int64)
    pred_OR_gt_arr  = np.zeros(num_slices_per_volume, dtype=np.int64)
    pred_arr        = np.zeros(num_slices_per_volume, dtype=np.int64)
    gt_arr          = np.zeros(num_slices_per_volume, dtype=np.int64)

    for slice_num in range(num_slices_per_volume):

        print(f'slice: {slice_num}/{num_slices_per_volume}')

        # Load slice
        norm_slice, brain_mask, mask = load_and_normalize_slice(vol_num, slice_num)

        # Perform inference (pass norm_slice)
        healthy_probabilities = healthy_single_gaussian_stat_inference(norm_slice, brain_mask)
        tumor_probabilities   = tumor_single_gaussian_stat_inference(norm_slice, brain_mask)
        stacked_probabilities = np.dstack([healthy_probabilities, tumor_probabilities])
        total_evidence        = np.sum(stacked_probabilities, axis=-1, keepdims=True)
        posteriors            = np.divide(
            stacked_probabilities,
            total_evidence,
            out=np.zeros_like(stacked_probabilities),
            where=total_evidence > 0,
        )

        entropy_map                     = compute_entropy(posteriors, brain_mask)
        sobel_map                       = sobel_edge_detection(posteriors, brain_mask)
        blob_array                      = contour_detection(sobel_map, brain_mask=brain_mask)
        classified_blobs, is_tumor_list = contour_classification(blob_array, posteriors, entropy_map)
        total_segmentation_mask         = expansion_loop(classified_blobs, entropy_map, posteriors, brain_mask)

        # Handle ground truth shape (2D or 3D)
        if mask.ndim == 3:
            gt_mask_1D = np.sum(mask, axis=-1) > 0
        else:
            gt_mask_1D = mask > 0

        # Define predicted binary mask
        pred_mask = (total_segmentation_mask > 0) & brain_mask.astype(bool)

        # Slice pixel counts
        pred_AND_gt = np.sum(pred_mask & gt_mask_1D)
        pred_OR_gt  = np.sum(pred_mask | gt_mask_1D)
        pred        = np.sum(pred_mask)
        gt          = np.sum(gt_mask_1D)

        pred_AND_gt_arr[slice_num] = pred_AND_gt
        pred_OR_gt_arr[slice_num]  = pred_OR_gt
        pred_arr[slice_num]        = pred
        gt_arr[slice_num]          = gt

    total_intersection = np.sum(pred_AND_gt_arr)
    total_union        = np.sum(pred_OR_gt_arr)
    total_pred         = np.sum(pred_arr)
    total_gt           = np.sum(gt_arr)

    # Volumetric metrics calculation
    dice = (2.0 * total_intersection) / (total_pred + total_gt + 1e-12)
    iou  = total_intersection / (total_union + 1e-12)

    print(f'Dice: {dice}, IOU: {iou}')
    return float(dice), float(iou)