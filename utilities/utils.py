import os
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from config import DATA_ROOT, PROJECT_ROOT

DATASET_DIR = os.path.join(
    DATA_ROOT,
    "MRI_2026_datasets",
    "Brats",
    "BraTS2020_training_data",
    "content",
    "data",
)
UTILITIES_DIR = os.path.join(PROJECT_ROOT, "utilities")

VOLUME_MEANS = np.load(os.path.join(UTILITIES_DIR, "volume_means.npy"))
VOLUME_STDS = np.load(os.path.join(UTILITIES_DIR, "volume_stds.npy"))


def load_and_normalize_slice(vol_num, slice_num, blur_sigma=4.0):
    """
    Loads an HDF5 slice and applies 9D z-score normalization.
    Zeros out symmetry channels (4-7) for pixels lacking a contralateral partner.
    """
    file_path = os.path.join(DATASET_DIR, f"volume_{vol_num}_slice_{slice_num}.h5")

    with h5py.File(file_path, "r") as f:
        image = f["image"][:].astype(np.float64)
        mask = f["mask"][:].astype(np.float32)

    brain_mask = np.any(image > 1e-8, axis=-1)
    mirrored_brain_mask = np.flipud(brain_mask)
    symmetric_brain_mask = brain_mask & mirrored_brain_mask

    # 1. Symmetry features (NDI)
    blurred_raw = gaussian_filter(image, sigma=(blur_sigma, blur_sigma, 0.0))
    mirrored_raw = np.flipud(blurred_raw)
    denom = np.abs(blurred_raw) + np.abs(mirrored_raw) + 1e-4
    ndi_raw = np.clip((blurred_raw - mirrored_raw) / denom, -1.0, 1.0)

    # 2. Boundary distance feature
    dist_map = distance_transform_edt(brain_mask)
    max_depth = np.max(dist_map)
    normalized_depth = dist_map / max_depth if max_depth > 0 else np.zeros_like(dist_map)

    # 3. Stack all 9 unnormalized channels: (H, W, 9)
    raw_9d = np.dstack([image, ndi_raw, normalized_depth[:, :, np.newaxis]])

    # 4. Vectorized Z-Score Normalization
    multimodal_image = np.zeros_like(raw_9d)
    if np.any(brain_mask):
        mu = VOLUME_MEANS[vol_num]
        sigma = VOLUME_STDS[vol_num]

        # Standardize entire brain
        multimodal_image[brain_mask] = (raw_9d[brain_mask] - mu) / sigma

        # Explicitly zero out symmetry channels (4-7) where no bilateral mirror exists
        non_symmetric_brain = brain_mask & (~symmetric_brain_mask)
        multimodal_image[non_symmetric_brain, 4:8] = 0.0

    return multimodal_image, brain_mask, mask, symmetric_brain_mask