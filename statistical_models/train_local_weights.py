import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from config import *
from utilities.utils import load_and_normalize_slice

# Set dark background for medical image contrast
plt.style.use('dark_background')


def compute_and_save_local_weights(symmetric: bool = False):
    """
    Accumulates spatial GMM likelihoods across training volumes to derive coordinate-dependent
    local component mixture weights \\pi_k(x, y).

    :param symmetric: If True, uses 8D symmetric GMM parameters and saves to 'local_weights_symmetric.npz'.
                      If False, uses 4D base GMM parameters and saves to 'local_weights.npz'.
    :param run_diagnostics: If True, generates diagnostic plots comparing local weights to global baselines.
    """
    # 1. Parameter & Path Configurations
    num_features = 8 if symmetric else 4
    gmm_filename = 'healthy_gmm_symmetric.npz' if symmetric else 'healthy_gmm.npz'
    weights_filename = 'local_weights_symmetric.npz' if symmetric else 'local_weights.npz'

    models_dir = os.path.join(PROJECT_ROOT, 'saved_parameters', 'statistical_models')
    gmm_path = os.path.join(models_dir, gmm_filename)
    output_path = os.path.join(models_dir, weights_filename)
    os.makedirs(models_dir, exist_ok=True)

    # Load Pre-trained Healthy GMM Parameters
    gmm_data = np.load(gmm_path)
    global_weights = gmm_data['weights']      # Shape: (K,)
    means = gmm_data['means']                  # Shape: (K, num_features)
    raw_covariances = gmm_data['covariances']  # Shape: (K, num_features, num_features)

    num_components = len(global_weights)
    H, W = IMAGE_PIXEL_LENGTH, IMAGE_PIXEL_LENGTH

    # Pre-regularize covariance matrices BEFORE evaluating PDFs to prevent BLAS/Cholesky overflow
    REG_FACTOR = 1e-3
    covariances_reg = np.array([
        raw_covariances[k] + REG_FACTOR * np.eye(num_features, dtype=np.float64)
        for k in range(num_components)
    ])

    # 2. Accumulate Spatial GMM Likelihoods
    local_likelihood_cumulator = np.zeros((H, W, num_components), dtype=np.float64)
    local_pixel_counter = np.zeros((H, W), dtype=np.float64)

    print(f"Accumulating spatial GMM likelihoods across volumes 1 to {MAX_TRAINING_VOLUME} (Symmetric: {symmetric})...")

    for vol_num in range(1, MAX_TRAINING_VOLUME + 1):
        D8_image, brain_mask, tumor_masks, _ = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=True)
        features_image = D8_image if symmetric else D8_image[:, :, :4]

        binary_gt_mask = np.any(tumor_masks > 0, axis=-1) if tumor_masks.ndim == 3 else (tumor_masks > 0)
        healthy_mask = brain_mask & (~binary_gt_mask)

        if np.any(healthy_mask):
            local_pixel_counter += healthy_mask.astype(np.float64)
            X_healthy = features_image[healthy_mask].astype(np.float64)

            for k in range(num_components):
                pdf_k = multivariate_normal.pdf(
                    X_healthy,
                    mean=means[k],
                    cov=covariances_reg[k],
                    allow_singular=True
                )
                weighted_likelihood_k = global_weights[k] * pdf_k
                local_likelihood_cumulator[:, :, k][healthy_mask] += weighted_likelihood_k

    # 3. Normalize Across Components to Derive Spatial Weights \pi_k(x, y)
    likelihood_sum = np.sum(local_likelihood_cumulator, axis=-1, keepdims=True)
    likelihood_sum_safe = np.where(likelihood_sum == 0, 1e-12, likelihood_sum)

    local_weights = local_likelihood_cumulator / likelihood_sum_safe

    # Fallback to global prior for unvisited background pixels
    unvisited_pixels = (local_pixel_counter == 0)
    local_weights[unvisited_pixels] = global_weights

    print("\n--- Accumulation Summary ---")
    print(f"Total training volumes processed: {MAX_TRAINING_VOLUME}")
    print(f"Max healthy observations per coordinate: {np.max(local_pixel_counter):.0f}")
    print(f"Local Weights Matrix Shape: {local_weights.shape}")

    # 4. Save Weight Matrix
    np.savez(
        output_path,
        local_pixel_counter=local_pixel_counter,
        weights=local_weights,
        global_weights=global_weights
    )
    print(f"Local weight map successfully saved to '{output_path}'\n")



if __name__ == "__main__":
    compute_and_save_local_weights(symmetric=False, run_diagnostics=True)
    compute_and_save_local_weights(symmetric=True, run_diagnostics=True)