import os
from pathlib import Path
import numpy as np

import time
from config_parameters import *

from statistical_models.tumor_single_gaussian.tumor_single_gaussian_posterior import tumor_single_gaussian_stat_inference
from statistical_models.healthy_single_gaussian.healthy_single_gaussian_posterior import healthy_single_gaussian_stat_inference

from statistical_models.tumor_single_skew_t.tumor_single_skew_t_posterior import tumor_single_skew_t_stat_inference
from statistical_models.healthy_single_skew_t.healthy_single_skew_t_posterior import healthy_single_skew_t_stat_inference

from statistical_models.healthy_gmm.healthy_gmm_posterior import (healthy_gmm_stat_inference,)
from statistical_models.tumor_gmm.tumor_gmm_posterior import (tumor_gmm_stat_inference,)

from image_processing.visualizations          import visualize_probability,visualize_entropy,visualize_sobel_edges, visualize_contours, visualize_expansion, visualize_segmentation
from image_processing.compute_entropy         import compute_entropy
from image_processing.edge_detection          import sobel_edge_detection
from image_processing.contour_detection       import contour_detection
from image_processing.contour_classification  import contour_classification
from image_processing.seed_expansion          import expansion_loop

from evaluation.eval_loop import eval_vol, eval_dataset

from utilities.utils import load_and_normalize_slice



if __name__ == "__main__":
    # ==============================================================================
    # PIPELINE CONFIGURATION & PARAMETERS
    # ==============================================================================

    volume_num = 305
    slice_num = 90
    target_row = 115



    figure_output_path = (f"Brain_Tumor_Segmentation/output_figures/vol{volume_num}_slice{slice_num}_row{target_row}")
    os.makedirs(figure_output_path,exist_ok=True)

    # load slice
    slice_im, brain_mask, gt_mask = load_and_normalize_slice(volume_num, slice_num)


    # compute full posterior map
    if MODEL == "gaussian":
        healthy_probabilities = healthy_single_gaussian_stat_inference(slice_im, brain_mask)  # Shape: (H, W, K_healthy)
        tumor_probabilities   = tumor_single_gaussian_stat_inference  (slice_im, brain_mask)  # Shape: (H, W, K_tumor)
    if MODEL == "skew_t":
        healthy_probabilities = healthy_single_skew_t_stat_inference(slice_im, brain_mask)  # Shape: (H, W, K_healthy)
        tumor_probabilities   = tumor_single_skew_t_stat_inference  (slice_im, brain_mask)  # Shape: (H, W, K_tumor)
    if MODEL == "gmm":
        healthy_probabilities = healthy_gmm_stat_inference(slice_im, brain_mask)
        tumor_probabilities = tumor_gmm_stat_inference(slice_im, brain_mask)
    stacked_probabilities = np.dstack([healthy_probabilities, tumor_probabilities])
    total_evidence        = np.sum(stacked_probabilities, axis=-1, keepdims=True)
    posteriors            = np.divide(stacked_probabilities,total_evidence,out=np.zeros_like(stacked_probabilities),where=total_evidence > 0)


    entropy_map                       = compute_entropy(posteriors,brain_mask)
    sobel_map                         = sobel_edge_detection(posteriors,brain_mask)
    blob_array                        = contour_detection(sobel_map)
    classified_blobs, is_tumor_list   = contour_classification(blob_array,posteriors,entropy_map)
    total_segmentation_mask           = expansion_loop(classified_blobs, entropy_map, posteriors, brain_mask)




    fig1_path = os.path.join(figure_output_path, "probability.png")
    visualize_probability(slice_im, posteriors, brain_mask, gt_mask, fig1_path)

    fig2_path = os.path.join(figure_output_path, "entropy.png")
    visualize_entropy(entropy_map, brain_mask, fig2_path)

    fig3_path = os.path.join(figure_output_path, "edges.png")
    visualize_sobel_edges(sobel_map, brain_mask, fig3_path)

    fig4_path = os.path.join(figure_output_path, "contours.png")
    visualize_contours(slice_im,posteriors,sobel_map,brain_mask,gt_mask,blob_array,is_tumor_list,fig4_path,)

    fig5_path = os.path.join(figure_output_path, "seed_expansion.png")
    visualize_expansion(total_segmentation_mask,slice_im,posteriors,brain_mask,gt_mask,blob_array,is_tumor_list,fig5_path)

    fig6_path = os.path.join(figure_output_path, "segmentation_results.png")
    visualize_segmentation(total_segmentation_mask, gt_mask, brain_mask, fig6_path)


    eval_start = time.perf_counter()
    eval_dataset(output_directory='evaluation/metrics', model_name='single_gaussian')
    eval_end = time.perf_counter()
    print(f'Evaluation Runtime: {(eval_end-eval_start):.3f}')