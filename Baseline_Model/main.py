import os
from inference import (
    compute_entropy,
    posterior_inference,
    sobel_edge_detection,
    sum_tumor_posterior,
)

# Import custom pipeline modules
from utils import load_and_normalize_slice, load_volume_stats
from segmentation import (
    contour_classification,
    contour_detection,
    expansion_loop,
)
from visualization import (
    visualize_entropy,
    visualize_expansion,
    visualize_modalities,
    visualize_probability,
    visualize_row_analysis,
    visualize_shape_detection,
    visualize_gt_vs_prob,
    visualize_evaluation
)

if __name__ == "__main__":
    # ==============================================================================
    # PIPELINE CONFIGURATION & PARAMETERS
    # ==============================================================================
    scan_data_path = "MRI_2026_datasets/Brats/BraTS2020_training_data/content/data"
    z_score_stats_path = "data"
    generative_model_parameters_path = (
        "generative_model_parameters/gmm_6class_volwise_zscore_parameters.npz"
    )
    figure_output_path = "output_figures/vol_{volume_num}"

    volume_num = 320  # Volume ID (1..369)
    slice_num = 90  # 2D Slice Index (0..154)
    target_row = 120  # Selected row for 1D posterior analysis

    figure_output_path = f"output_figures/vol{volume_num}_slice{slice_num}_row{target_row}"

    min_pixels_per_blob = 150  # Area cutoff for noise filtering
    blob_class_threshold = 0.2  # Probability sum threshold to keep blob
    max_expansion_diameter = 10  # Maximum pixel dilation radius

    os.makedirs(figure_output_path, exist_ok=True)

    # ==============================================================================
    # 1. LOAD & NORMALIZE MULTI-MODAL MRI SLICE
    # ==============================================================================
    print(f"--- Loading Volume {volume_num}, Slice {slice_num} ---")
    volume_means, volume_stds = load_volume_stats(z_score_stats_path)
    slice_im, brain_mask, gt_mask = load_and_normalize_slice(
        scan_data_path, volume_num, slice_num, volume_means, volume_stds
    )



    # ==============================================================================
    # 2. GMM POSTERIOR INFERENCE & CONFIDENCE EVALUATION
    # ==============================================================================
    print("--- Running 6-Class GMM Posterior Inference ---")
    posteriors_6d, class_names = posterior_inference(
        slice_im, brain_mask, generative_model_parameters_path
    )
    entropy_map = compute_entropy(posteriors_6d, brain_mask)
    tumor_posterior_map = sum_tumor_posterior(posteriors_6d)

    # ==============================================================================
    # 3. SOBEL GRADIENT & BLOB EXTRACTION
    # ==============================================================================
    print("--- Extracting Sobel Boundaries & Contours ---")
    sobel_map = sobel_edge_detection(tumor_posterior_map, brain_mask)
    blob_array = contour_detection(
        sobel_map, min_pixels_per_blob=min_pixels_per_blob
    )

    # ==============================================================================
    # 4. ENTROPY-WEIGHTED BLOB CLASSIFICATION
    # ==============================================================================
    print("--- Classifying Contours via Entropy-Weighted Scoring ---")
    classified_blobs, is_tumor_list = contour_classification(
        blob_array,
        posteriors_6d,
        entropy_map,
        blob_class_threshold=blob_class_threshold,
    )

    # ==============================================================================
    # 5. AMBIGUOUS SPACE EXPANSION LOOP
    # ==============================================================================
    print("--- Expanding Tumor Blobs into Ambiguous Entropy Space ---")
    total_segmentation_mask = expansion_loop(
        classified_blobs,
        entropy_map,
        posteriors_6d,
        brain_mask,
        max_expansion_diameter=max_expansion_diameter,
    )

# ==============================================================================
    # 6. GENERATE & SAVE ALL DIAGNOSTIC FIGURES (600 DPI)
    # ==============================================================================
    print("--- Generating High-Resolution Diagnostic Figures ---")

    fig1_path = os.path.join(figure_output_path, "1_modalities.png")
    visualize_modalities(slice_im, brain_mask, fig1_path)

    fig2_path = os.path.join(figure_output_path, "2_probability.png")
    visualize_probability(
        slice_im, posteriors_6d, brain_mask, gt_mask, fig2_path
    )

    fig3_path = os.path.join(figure_output_path, "3_maximum_likelihood.png")
    visualize_gt_vs_prob(
        slice_im, posteriors_6d, brain_mask, gt_mask, fig3_path
    )

    fig4_path = os.path.join(figure_output_path, "4_shape_detection.png")
    visualize_shape_detection(
        slice_im,
        sobel_map,
        blob_array,
        is_tumor_list,
        classified_blobs,
        brain_mask,
        gt_mask,
        fig4_path,
    )

    fig5_path = os.path.join(figure_output_path, "5_expansion.png")
    visualize_expansion(
        slice_im,
        entropy_map,
        classified_blobs,
        total_segmentation_mask,
        posteriors_6d,
        brain_mask,
        gt_mask,
        fig5_path,
    )

    fig6_path = os.path.join(figure_output_path, "6_row_analysis.png")
    visualize_row_analysis(
        posteriors_6d, entropy_map, brain_mask, gt_mask, target_row, fig6_path
    )

    fig7_path = os.path.join(figure_output_path, "7_entropy.png")
    visualize_entropy(entropy_map, gt_mask, fig7_path)

    fig8_path = os.path.join(figure_output_path, "8_segmentation_evaluation.png")
    visualize_evaluation(
        slice_im,
        total_segmentation_mask,
        brain_mask,
        gt_mask,
        fig8_path,
    )


    print(
        f"\nPipeline Execution Complete! Figures saved in: '{figure_output_path}'"
    )