import os
import numpy as np
from pathlib import Path
import sys


# Add project root ('Brain_Tumor_Segmentation') to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Now your import will resolve cleanly!
from utilities.utils import load_and_normalize_slice
from config_parameters import *



def estimate_healthy_gaussian_parameters(
    dataset_base_path='MRI_2026_datasets/Brats/BraTS2020_training_data',
    max_train_vol=MAX_TRAINING_VOLUME,
    total_slices=MAX_SLICE + 1,
):
    """Estimates single multivariate Gaussian parameters (mean, covariance)

    and prior for healthy brain voxels (within brain_mask, outside tumor mask)
    using maximum likelihood estimation.
    """

    print("\nEstimating healthy single Gaussian parameters...")

    num_dims = 4

    total_brain_voxels = np.uint64(0)
    running_healthy_count = np.uint64(0)

    running_sum = np.zeros(num_dims, dtype=np.float64)
    running_sum_squares = np.zeros((num_dims, num_dims), dtype=np.float64)

    for vol_num in range(1, max_train_vol + 1):
        print(f'volume: {vol_num}/{MAX_TRAINING_VOLUME}')
        for slice_num in range(total_slices):
            (
                norm_slice,
                brain_mask,
                mask_slice,
            ) = load_and_normalize_slice(
                vol_num,
                slice_num
            )

            if not np.any(brain_mask):
                continue

            image_flat = norm_slice.reshape(-1, num_dims)
            mask_flat = mask_slice.reshape(-1, 3)
            brain_flat = brain_mask.ravel()

            # Track total brain voxels for prior denominator
            num_brain_voxels = np.count_nonzero(brain_flat)
            total_brain_voxels += np.uint64(num_brain_voxels)

            # Healthy voxels: inside brain mask AND sum across all 3 tumor channels is 0
            healthy_mask = brain_flat & (np.sum(mask_flat, axis=1) == 0)

            if not np.any(healthy_mask):
                continue

            pixels = image_flat[healthy_mask]

            running_healthy_count += np.uint64(pixels.shape[0])
            running_sum += np.sum(pixels, axis=0)
            running_sum_squares += pixels.T @ pixels

        if vol_num % 50 == 0 or vol_num == max_train_vol:
            print(f"Processed {vol_num}/{max_train_vol}")

    # Compute Healthy Class Prior
    healthy_prior = float(running_healthy_count) / float(total_brain_voxels)

    # Compute Single Mean & Covariance via MLE
    healthy_mean = running_sum / float(running_healthy_count)
    healthy_covariance = (
        running_sum_squares / float(running_healthy_count)
    ) - np.outer(healthy_mean, healthy_mean)


    np.savez(
        "Brain_Tumor_Segmentation/statistical_models/healthy_single_gaussian/healthy_single_gaussian_parameters.npz",
        healthy_prior=healthy_prior,
        healthy_mean=healthy_mean,
        healthy_covariance=healthy_covariance,
        healthy_count=running_healthy_count,
        total_brain_count=total_brain_voxels,
        modalities=np.array(["T1", "T1ce", "T2", "FLAIR"]),
    )

    print("\nHealthy Gaussian parameters saved to:")
    print("\nHealthy Class Summary")
    print("-----------------------------------")
    print(f"Healthy Voxels     : {running_healthy_count:,}")
    print(f"Total Brain Voxels : {total_brain_voxels:,}")
    print(f"Healthy Prior      : {healthy_prior:.6f}")
    print("Mean Vector        :", healthy_mean)

    return healthy_prior, healthy_mean, healthy_covariance


if __name__ == "__main__":

    healthy_prior, healthy_mean, healthy_covariance = (
        estimate_healthy_gaussian_parameters(
            max_train_vol=MAX_TRAINING_VOLUME,
            total_slices=MAX_SLICE + 1,
        )
    )

    print("\nFinished estimating single healthy Gaussian parameters.")