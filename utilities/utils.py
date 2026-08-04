import os
import h5py
import numpy as np
from config_parameters import *


def load_and_normalize_slice(vol_num, slice_num):
    """Loads a single slice and applies volume-wise Z-score normalization in milliseconds using pre-computed statistics.

    Returns:
        norm_slice: (H, W, 4) float64 array
        brain_mask: (H, W) boolean array
        mask_slice: (H, W, 3) float32 mask array
    """
    file_path = os.path.join(
        'MRI_2026_datasets/Brats/BraTS2020_training_data/content/data/',
         f"volume_{vol_num}_slice_{slice_num}.h5")

    

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

    volume_means = np.load('Brain_Tumor_Segmentation/utilities/volume_means.npy')
    volume_stds = np.load('Brain_Tumor_Segmentation/utilities/volume_stds.npy')


    if np.any(brain_mask):
        mu = volume_means[vol_num]
        sigma = volume_stds[vol_num]
        norm_slice[brain_mask] = (image[brain_mask] - mu) / sigma
        norm_slice[brain_mask] = np.clip(norm_slice[brain_mask], -6.0, 6.0)

    return norm_slice, brain_mask, mask
