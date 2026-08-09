import os
import numpy as np
from sklearn.mixture import GaussianMixture
from config import *
from utilities.utils import load_and_normalize_slice


def fit_and_save_healthy_gmm(symmetric: bool = False):
    """
    Fits a Gaussian Mixture Model on healthy brain tissue voxels and saves the model parameters.

    :param symmetric: If True, uses all 8 feature channels and saves to 'healthy_gmm_symmetric.npz'.
                      If False, uses the first 4 base modalities and saves to 'healthy_gmm.npz'.
    """
    # 1. Parameter & Path Configuration based on mode
    num_features = 8 if symmetric else 4
    n_components = GMM_SYMMETRIC_COMPONENTS if symmetric else GMM_REGULAR_COMPONENTS
    filename = 'healthy_gmm_symmetric.npz' if symmetric else 'healthy_gmm.npz'

    output_dir = os.path.join(PROJECT_ROOT, 'saved_parameters', 'statistical_models')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)

    # Initialize memory array for GMM training features: shape (num_features, N)
    healthy_pixels = np.zeros((num_features, NUM_HEALTHY_TRAINING_SAMPLES), dtype=np.float64)
    sampled_pixel_counter = 0

    # Dataset-wide counts across ALL training volumes
    total_dataset_healthy_pixels = 0
    total_dataset_brain_pixels = 0

    print(f"Processing training volumes 1 to {MAX_TRAINING_VOLUME} (Symmetric Mode: {symmetric})...")

    for vol_num in range(1, MAX_TRAINING_VOLUME + 1):
        # Load 8D slice and masks
        D8_image, brain_mask, mask, _ = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=True)

        # Select target feature channels (4D vs 8D)
        features_image = D8_image if symmetric else D8_image[:, :, :4]

        # Binary ground truth mask
        binary_gt_mask = np.any(mask > 0, axis=-1) if mask.ndim == 3 else (mask > 0)

        # Valid healthy mask
        healthy_mask = brain_mask & (~binary_gt_mask)

        # Count dataset-wide totals for prior estimation
        n_brain = np.sum(brain_mask)
        n_healthy = np.sum(healthy_mask)

        total_dataset_brain_pixels += n_brain
        total_dataset_healthy_pixels += n_healthy

        # Feature sampling for GMM buffer
        if (sampled_pixel_counter < NUM_HEALTHY_TRAINING_SAMPLES) and (n_healthy > 0):
            valid_features = features_image[healthy_mask]
            num_valid = valid_features.shape[0]
            remaining_capacity = NUM_HEALTHY_TRAINING_SAMPLES - sampled_pixel_counter

            if num_valid > remaining_capacity:
                valid_features = valid_features[:remaining_capacity]
                num_valid = remaining_capacity

            healthy_pixels[:, sampled_pixel_counter: sampled_pixel_counter + num_valid] = valid_features.T
            sampled_pixel_counter += num_valid

            print(
                f"Vol {vol_num}/{MAX_TRAINING_VOLUME} | Feature Buffer: {sampled_pixel_counter}/{NUM_HEALTHY_TRAINING_SAMPLES}"
            )

    # Trim unused array capacity
    healthy_pixels = healthy_pixels[:, :sampled_pixel_counter]

    # Compute dataset-wide prior P(Healthy)
    healthy_prior = (
        total_dataset_healthy_pixels / total_dataset_brain_pixels
        if total_dataset_brain_pixels > 0
        else 1.0
    )

    print("\n--- Summary ---")
    print(f"Total Dataset Brain Voxels ({MAX_TRAINING_VOLUME} vols): {total_dataset_brain_pixels}")
    print(f"Total Dataset Healthy Voxels ({MAX_TRAINING_VOLUME} vols): {total_dataset_healthy_pixels}")
    print(f"Dataset Healthy Prior P(Healthy): {healthy_prior:.6f}")
    print(f"GMM Training Matrix Shape: {healthy_pixels.T.shape}")

    print("\nFitting GMM model...")

    # 2. Fit GMM on Transposed Matrix: (N_samples, num_features)
    X_train = healthy_pixels.T

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=200,
        reg_covar=1e-5,
        init_params="random_from_data",
        random_state=RANDOM_SEED,
        n_init=5,
    )
    gmm.fit(X_train)

    # 3. Save parameters and prior
    np.savez(
        output_file,
        prior=healthy_prior,
        weights=gmm.weights_,
        means=gmm.means_,
        covariances=gmm.covariances_,
    )
    print(f"GMM parameters successfully saved to '{output_file}'")


if __name__ == "__main__":
    # Example function calls
    fit_and_save_healthy_gmm(symmetric=False)
    fit_and_save_healthy_gmm(symmetric=True)