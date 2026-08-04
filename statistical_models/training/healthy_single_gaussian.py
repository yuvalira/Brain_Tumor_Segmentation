import os
import numpy as np
from pathlib import Path

from config_parameters import *
from utils import load_and_normalize_slice

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def estimate_healthy_gaussian_parameters(
    dataset_base_path,
    volume_means,
    volume_stds,
    max_train_vol=MAX_TRAINING_VOLUME,
    total_slices=MAX_SLICE + 1,
    output_path=os.path.join(
        PARAMS_OUTPUT_PATH,
        "healthy_single_gaussian_parameters.npz",
    ),
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
        for slice_num in range(total_slices):
            (
                norm_slice,
                brain_mask,
                mask_slice,
            ) = load_and_normalize_slice(
                dataset_base_path,
                vol_num,
                slice_num,
                volume_means,
                volume_stds,
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez(
        output_path,
        healthy_prior=healthy_prior,
        healthy_mean=healthy_mean,
        healthy_covariance=healthy_covariance,
        healthy_count=running_healthy_count,
        total_brain_count=total_brain_voxels,
        modalities=np.array(["T1", "T1ce", "T2", "FLAIR"]),
    )

    print("\nHealthy Gaussian parameters saved to:")
    print(output_path)

    print("\nHealthy Class Summary")
    print("-----------------------------------")
    print(f"Healthy Voxels     : {running_healthy_count:,}")
    print(f"Total Brain Voxels : {total_brain_voxels:,}")
    print(f"Healthy Prior      : {healthy_prior:.6f}")
    print("Mean Vector        :", healthy_mean)

    return healthy_prior, healthy_mean, healthy_covariance


if __name__ == "__main__":
    from pathlib import Path
    from utils import load_volume_stats

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    stats_path = PROJECT_ROOT / PARAMS_OUTPUT_PATH
    dataset_path = PROJECT_ROOT / DATASET_PATH
    output_path = stats_path / "healthy_single_gaussian_parameters.npz"

    print("Project root:", PROJECT_ROOT)
    print("Statistics path:", stats_path)
    print("Dataset path:", dataset_path)
    print("Dataset exists:", dataset_path.exists())

    volume_means, volume_stds = load_volume_stats(stats_path)

    healthy_prior, healthy_mean, healthy_covariance = (
        estimate_healthy_gaussian_parameters(
            dataset_base_path=dataset_path,
            volume_means=volume_means,
            volume_stds=volume_stds,
            max_train_vol=MAX_TRAINING_VOLUME,
            total_slices=MAX_SLICE + 1,
            output_path=output_path,
        )
    )

    print("\nFinished estimating single healthy Gaussian parameters.")