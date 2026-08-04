import os
import h5py
import numpy as np
from config_parameters import *


def compute_dataset_volume_stats(
    dataset_base_path,
    total_volumes=369,
    total_slices=155,
    save_dir=PARAMS_OUTPUT_PATH,
):
    """Pre-computes, saves, and returns arrays of shape (total_volumes + 1, 4)

    for volume-wide means and standard deviations across non-zero brain voxels.
    """
    print(
        f"Pre-computing Volume-Wise Z-Score Statistics for {total_volumes} volumes..."
    )

    volume_means = np.zeros((total_volumes + 1, 4), dtype=np.float64)
    volume_stds = np.ones((total_volumes + 1, 4), dtype=np.float64)

    for vol_num in range(1, total_volumes + 1):
        running_count = 0
        running_sum = np.zeros(4, dtype=np.float64)
        running_sum_sq = np.zeros(4, dtype=np.float64)

        for slice_num in range(total_slices):
            file_path = os.path.join(
                dataset_base_path, f"volume_{vol_num}_slice_{slice_num}.h5"
            )
            if not os.path.exists(file_path):
                continue
            try:
                with h5py.File(file_path, "r") as f:
                    image = f["image"][:].astype(np.float64)
            except Exception:
                continue

            # brain mask change --- start ----------------------
            brain_mask = np.any(image > 1e-8, axis=-1)
            # brain mask change --- end ---------------------
            if np.any(brain_mask):
                pixels = image[brain_mask]  # (N_pixels, 4)
                running_count += pixels.shape[0]
                running_sum += np.sum(pixels, axis=0)
                running_sum_sq += np.sum(pixels**2, axis=0)

        # Calculate volume-level statistics across accumulated brain voxels
        if running_count > 0:
            mu = running_sum / running_count
            variance = (running_sum_sq / running_count) - (mu**2)
            # Guard against tiny negative floats due to precision limits
            sigma = np.sqrt(np.maximum(variance, 1e-8))

            volume_means[vol_num] = mu
            volume_stds[vol_num] = sigma

        if vol_num % 50 == 0 or vol_num == total_volumes:
            print(f"  Calculated volume stats: {vol_num}/{total_volumes}...")

    # Ensure output directory exists before saving
    os.makedirs(save_dir, exist_ok=True)

    # Save finalized statistics to disk
    np.save(os.path.join(save_dir, "volume_means.npy"), volume_means)
    np.save(os.path.join(save_dir, "volume_stds.npy"), volume_stds)

    print(
        f"Volume statistics successfully saved to '{os.path.abspath(save_dir)}'!"
    )

    return volume_means, volume_stds


if __name__ == "__main__":
    compute_dataset_volume_stats(DATASET_PATH)