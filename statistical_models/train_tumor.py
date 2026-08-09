import os
import numpy as np
from config import *
from utilities.utils import load_and_normalize_slice


def fit_and_save_tumor_gaussian(symmetric: bool = False):
    """
    Computes class-conditional Gaussian parameters (means, covariances, priors)
    for tumor sub-classes across training volumes.

    :param symmetric: If True, uses all 8 feature channels and saves to 'tumor_gaussian_symmetric.npz'.
                      If False, uses the first 4 raw modalities and saves to 'tumor_gaussian.npz'.
    """
    num_classes = 3
    num_features = 8 if symmetric else 4
    filename = 'tumor_gaussian_symmetric.npz' if symmetric else 'tumor_gaussian.npz'

    output_dir = os.path.join(PROJECT_ROOT, 'saved_parameters', 'statistical_models')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)

    tumor_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_dataset_brain_pixels = 0

    # Accumulators initialized strictly as float64 C-contiguous matrices
    sum_x = np.zeros((num_classes, num_features), dtype=np.float64)
    sum_xxT = np.zeros((num_classes, num_features, num_features), dtype=np.float64)

    print(f"Processing training volumes 1 to {MAX_TRAINING_VOLUME} (Symmetric Mode: {symmetric})...")

    for vol_num in range(1, MAX_TRAINING_VOLUME + 1):
        D8_image, brain_mask, tumor_masks, _ = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=True)

        # Select target feature channels (4D vs 8D)
        features_image = D8_image if symmetric else D8_image[:, :, :4]

        total_dataset_brain_pixels += np.sum(brain_mask)

        for k in range(num_classes):
            class_mask = (tumor_masks[:, :, k] > 0) & brain_mask
            n_pixels = np.sum(class_mask)

            if n_pixels > 0:
                # Force C-contiguous float64 memory block for (N, num_features) features
                X_k = np.ascontiguousarray(features_image[class_mask], dtype=np.float64)

                tumor_pixel_counts[k] += n_pixels
                sum_x[k] += np.sum(X_k, axis=0)

                # Direct C-level double-precision accumulation: (num_features, N) @ (N, num_features)
                sum_xxT[k] += np.dot(X_k.T, X_k)

    # 2. Compute Class Means, Covariances, and Priors
    means = np.zeros((num_classes, num_features), dtype=np.float64)
    covariances = np.zeros((num_classes, num_features, num_features), dtype=np.float64)
    priors = np.zeros(num_classes, dtype=np.float64)

    for k in range(num_classes):
        N_k = tumor_pixel_counts[k]
        if N_k > 0:
            means[k] = sum_x[k] / N_k

            # Sample Covariance: E[X X^T] - \mu \mu^T + regularization
            cov_k = (sum_xxT[k] / N_k) - np.outer(means[k], means[k])
            covariances[k] = cov_k + 1e-5 * np.eye(num_features, dtype=np.float64)

            priors[k] = N_k / total_dataset_brain_pixels if total_dataset_brain_pixels > 0 else 0.0

    print("\n--- Accumulation Summary ---")
    for k in range(num_classes):
        print(f"Class {k}: {tumor_pixel_counts[k]} pixels | Prior: {priors[k]:.6f}")

    # 3. Save parameters and dataset priors
    np.savez(
        output_file,
        priors=priors,
        means=means,
        covariances=covariances,
        pixel_counts=tumor_pixel_counts,
        total_brain_pixels=total_dataset_brain_pixels
    )
    print(f"\nGaussian parameters successfully saved to '{output_file}'")


if __name__ == "__main__":
    fit_and_save_tumor_gaussian(symmetric=False)
    fit_and_save_tumor_gaussian(symmetric=True)