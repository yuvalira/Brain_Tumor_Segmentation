import numpy as np

from config import *

from statistical_models.healthy_likelihood import healthy_gmm_joint_likelihood
from statistical_models.tumor_likelihoods import tumor_joint_likelihood

from image_processing.visualizations          import visualize_probability,visualize_entropy,visualize_sobel_edges, visualize_contours, visualize_expansion, visualize_segmentation
from image_processing.compute_entropy         import compute_entropy
from image_processing.edge_detection          import sobel_edge_detection
from image_processing.contour_detection       import contour_detection
from image_processing.contour_classification  import contour_classification
from image_processing.seed_expansion          import expansion_loop

from utilities.utils import load_and_normalize_slice


def eval_vol(vol_num,
             target_row=None,
             diagnostic_figures = False,
             verbose = False,
             symmetric = False,

             lambda_val             = LAMBDA,

             min_pixels_per_blob    = MIN_NUM_PIXELS_PER_BLOB_DEFAULT,
             allow_internal       = ALLOW_INTERNAL_CONTOURS,
             binarization_factor   = SOBEL_BINARIZATION_OTSU_FACTOR,

             blob_class_threshold = WEIGHTED_POSTERIOR_MEAN_THRESHOLD,

             entropy_thresh       = ENTROPY_THRESHOLD_DEFAULT,
             posterior_min        = POSTERIOR_THRESHOLD_DEFAULT,
             max_expansion_diameter= MAX_EXPANSION_DIAMETER_DEFAULT
             ):



    slice_im, brain_mask, gt_mask, _ = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=True)
    healthy_probabilities = healthy_gmm_joint_likelihood(vol_num,
                                                         lambda_val = lambda_val,
                                                         symmetric = symmetric)
    tumor_probabilities = tumor_joint_likelihood(vol_num, symmetric = symmetric)

    stacked_probabilities = np.dstack([healthy_probabilities, tumor_probabilities])
    total_evidence        = np.sum(stacked_probabilities, axis=-1, keepdims=True)
    posteriors            = np.divide(stacked_probabilities,total_evidence,out=np.zeros_like(stacked_probabilities),where=total_evidence > 0)

    entropy_map                     = compute_entropy(posteriors, brain_mask)
    sobel_map                       = sobel_edge_detection(posteriors, brain_mask)



    blob_array                      = contour_detection(sobel_map,
                                                        brain_mask          = brain_mask,

                                                        min_pixels_per_blob = min_pixels_per_blob,
                                                        allow_internal      = allow_internal,
                                                        binarization_factor = binarization_factor)

    classified_blobs, is_tumor_list = contour_classification(blob_array,
                                                             posteriors,
                                                             entropy_map,

                                                             blob_class_threshold = blob_class_threshold)

    total_segmentation_mask         = expansion_loop(classified_blobs,
                                                     entropy_map,
                                                     posteriors,
                                                     brain_mask,

                                                     entropy_thresh = entropy_thresh,
                                                     posterior_min = posterior_min,
                                                     max_expansion_diameter = max_expansion_diameter)



    gt_mask_1D = np.sum(gt_mask, axis=-1) > 0

    pred_mask = (total_segmentation_mask > 0) & brain_mask.astype(bool)

    # Slice pixel counts
    pred_AND_gt = np.sum(pred_mask & gt_mask_1D)
    pred_OR_gt  = np.sum(pred_mask | gt_mask_1D)
    pred        = np.sum(pred_mask)
    gt          = np.sum(gt_mask_1D)


    # Volumetric metrics calculation
    dice = (2.0 * pred_AND_gt) / (pred + gt + 1e-12)
    iou  = pred_AND_gt / (pred_OR_gt + 1e-12)

    if verbose:
        print(f'Dice: {dice:.3f}, IOU: {iou:.3f}\n')

    if diagnostic_figures:

        figure_output_path = os.path.join(PROJECT_ROOT, 'output', 'diagnostic_figures', f'{MODEL}_vol{vol_num}_row{target_row}')
        os.makedirs(figure_output_path, exist_ok=True)

        fig1_path = os.path.join(figure_output_path, "probability.png")
        visualize_probability(slice_im, posteriors, brain_mask, gt_mask, fig1_path)

        fig2_path = os.path.join(figure_output_path, "entropy.png")
        visualize_entropy(entropy_map, brain_mask, fig2_path)

        fig3_path = os.path.join(figure_output_path, "edges.png")
        visualize_sobel_edges(sobel_map, brain_mask, fig3_path)

        fig4_path = os.path.join(figure_output_path, "contours.png")
        visualize_contours(slice_im, posteriors, sobel_map, brain_mask, gt_mask, blob_array, is_tumor_list, fig4_path, )

        fig5_path = os.path.join(figure_output_path, "seed_expansion.png")
        visualize_expansion(total_segmentation_mask, slice_im, posteriors, brain_mask, gt_mask, blob_array,
                            is_tumor_list, fig5_path)

        fig6_path = os.path.join(figure_output_path, "segmentation_results.png")
        visualize_segmentation(total_segmentation_mask, gt_mask, brain_mask, fig6_path)

    return dice, iou, pred_AND_gt, pred_OR_gt, pred, gt