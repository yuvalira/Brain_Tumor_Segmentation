import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_erosion
from config import *

# Set dark style for crisp medical image contrast
plt.style.use('dark_background')


import os
import h5py
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
from config import PROJECT_ROOT, DATA_ROOT

# 1. Define paths relative to project root
DATASET_DIR = os.path.join(DATA_ROOT, 'MRI_2026_datasets', 'Brats', 'BraTS2020_training_data', 'content', 'data')
print(f"dataset director: ", DATASET_DIR)
UTILITIES_DIR = os.path.join(PROJECT_ROOT, 'utilities')

# 2. Pre-load normalization stats once at module level to avoid repeated disk reads
VOLUME_MEANS = np.load(os.path.join(UTILITIES_DIR, 'volume_means.npy'))
VOLUME_STDS = np.load(os.path.join(UTILITIES_DIR, 'volume_stds.npy'))


def load_and_normalize_slice(vol_num, slice_num, symmetric=False, blur_sigma=2.0):
    """
    Loads an HDF5 slice, applies dataset-level z-score normalization, and optionally
    computes 4D spatial symmetry feature channels (NDI) to return an 8D image array.
    """
    file_path = os.path.join(DATASET_DIR, f"volume_{vol_num}_slice_{slice_num}.h5")

    with h5py.File(file_path, "r") as f:
        image = f["image"][:].astype(np.float64)
        mask = f["mask"][:].astype(np.float32)

    brain_mask = np.any(image > 1e-8, axis=-1)
    eroded_mask = binary_erosion(brain_mask, iterations=2)
    norm_slice = np.zeros_like(image)

    # 1. Base Z-score normalization across brain pixels
    if np.any(eroded_mask):
        mu = VOLUME_MEANS[vol_num]
        sigma = VOLUME_STDS[vol_num]
        sigma = np.where(sigma == 0, 1e-8, sigma)
        norm_slice[eroded_mask] = (image[eroded_mask] - mu) / sigma

    # 2. Return 4D slice if symmetry features are not requested
    if not symmetric:
        return norm_slice, brain_mask, mask

    # 3. Compute 4D Normalized Difference Index (NDI) for Symmetry (8D Mode)

    mirrored_eroded_mask = np.flipud(eroded_mask)
    symmetric_brain_mask = eroded_mask & mirrored_eroded_mask

    blurred_raw = np.zeros_like(image)
    for c in range(4):
        blurred_raw[:, :, c] = gaussian_filter(image[:, :, c], sigma=blur_sigma)

    mirrored_raw = np.flipud(blurred_raw)
    denom = np.maximum(blurred_raw + mirrored_raw, 1e-3)
    ndi_raw = np.clip((blurred_raw - mirrored_raw) / denom, -1.0, 1.0)

    norm_ndi = np.zeros_like(ndi_raw)
    if np.any(symmetric_brain_mask):
        for c in range(4):
            valid_pixels = ndi_raw[:, :, c][symmetric_brain_mask]
            if len(valid_pixels) > 0 and np.std(valid_pixels) > 0:
                m_ndi = np.mean(valid_pixels)
                s_ndi = np.std(valid_pixels)
                norm_ndi[:, :, c][symmetric_brain_mask] = (valid_pixels - m_ndi) / s_ndi

    D8_image = np.dstack([norm_slice, norm_ndi])

    return D8_image, brain_mask, mask, symmetric_brain_mask