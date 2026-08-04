import os
import h5py
import numpy as np
from config_parameters import *


def load_volume_stats(stats_dir=PARAMS_OUTPUT_PATH):
    """Loads pre-computed volume means and standard deviations."""
    means_path = os.path.join(stats_dir, "volume_means.npy")
    stds_path = os.path.join(stats_dir, "volume_stds.npy")

    if not os.path.exists(means_path) or not os.path.exists(stds_path):
        raise FileNotFoundError(
            f"Pre-computed volume statistics not found in '{stats_dir}'. "
            "Please run compute_dataset_volume_stats() first."
        )

    volume_means = np.load(means_path)
    volume_stds = np.load(stds_path)
    return volume_means, volume_stds


def load_and_normalize_slice(
    dataset_base_path, vol_num, slice_num, volume_means, volume_stds
):
    """Loads a single slice and applies volume-wise Z-score normalization in milliseconds using pre-computed statistics.

    Returns:
        norm_slice: (H, W, 4) float64 array
        brain_mask: (H, W) boolean array
        mask_slice: (H, W, 3) float32 mask array
    """
    file_path = os.path.join(
        dataset_base_path, f"volume_{vol_num}_slice_{slice_num}.h5"
    )

    if not os.path.exists(file_path):
        image = np.zeros((240, 240, 4), dtype=np.float64)
        mask = np.zeros((240, 240, 3), dtype=np.float32)
        return image, image[:, :, 0] > 0, mask

    try:
        with h5py.File(file_path, "r") as f:
            image = f["image"][:].astype(np.float64)
            mask = f["mask"][:].astype(np.float32)
    except Exception:
        image = np.zeros((240, 240, 4), dtype=np.float64)
        mask = np.zeros((240, 240, 3), dtype=np.float32)
        return image, image[:, :, 0] > 0, mask

    # brain mask change --- start ----------------------
    brain_mask = np.any(image > 1e-8, axis=-1)
    # brain mask change --- end ---------------------
    norm_slice = np.zeros_like(image)

    if np.any(brain_mask):
        mu = volume_means[vol_num]
        sigma = volume_stds[vol_num]
        norm_slice[brain_mask] = (image[brain_mask] - mu) / sigma
        norm_slice[brain_mask] = np.clip(norm_slice[brain_mask], -6.0, 6.0)

    return norm_slice, brain_mask, mask
