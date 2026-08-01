import os
import h5py
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

# Activate automatic NumPy <-> R array conversion
numpy2ri.activate()

# Import R's 'sn' package
try:
    sn = importr('sn')
except Exception as e:
    raise ImportError(
        "R package 'sn' is required. Please install it in your R environment using: "
        "install.packages('sn')"
    ) from e

# Import custom utility function
from utils import load_volume_stats

# --- Parameters & File Paths ---
DATASET_PATH = "MRI_2026_datasets/Brats/BraTS2020_training_data/content/data"
STATS_DIR = "data"
OUTPUT_PARAM_FILE = "generative_model_parameters/skew_t_4class_parameters.npz"
os.makedirs(os.path.dirname(OUTPUT_PARAM_FILE), exist_ok=True)

VOLUMES_RANGE = range(1, 241)  # Volumes 1 to 240
SLICES_PER_VOLUME = 155        # Slices 0 to 154
MAX_VOXELS_PER_CLASS = 150000  # Subsample limit per class to ensure fast MLE fitting

# Target Classes (including Background / Healthy brain)
# Adjust channel indexing based on your target mapping
TARGET_CLASSES = {
    "Background": "background", # Voxels inside brain mask but outside tumor
    "NCR_NET": 0,               # Non-Enhancing Core / Necrotic
    "Edema": 1,                 # Peritumoral Edema
    "ET": 2                     # Enhancing Tumor
}

# 1. Load Pre-computed Means and Stds
volume_means, volume_stds = load_volume_stats(STATS_DIR)

# 2. Voxel Accumulation Containers
accumulated_voxels = {c_name: [] for c_name in TARGET_CLASSES}

print("Accumulating 4D voxels across volumes 1 to 240...")

# 3. Extract Normalized Voxels Across Scans
for vol_idx in VOLUMES_RANGE:
    vol_mean = volume_means[vol_idx]
    vol_std = volume_stds[vol_idx]
    vol_std = np.where(vol_std == 0, 1e-8, vol_std)

    for slice_num in range(SLICES_PER_VOLUME):
        h5_filename = f"volume_{vol_idx}_slice_{slice_num}.h5"
        h5_path = os.path.join(DATASET_PATH, h5_filename)
        
        if not os.path.exists(h5_path):
            continue

        with h5py.File(h5_path, 'r') as hf:
            image_data = hf['image'][:]  # (240, 240, 4)
            mask_data = hf['mask'][:]    # (240, 240, 3)

        # Z-score normalize 4D slice
        norm_slice = (image_data - vol_mean) / vol_std

        # Create combined tumor mask to isolate healthy tissue
        any_tumor_mask = np.any(mask_data > 0, axis=-1)

        for c_name, target_ch in TARGET_CLASSES.items():
            if c_name == "Background":
                # Voxels where there is brain signal but NO tumor mask
                # Assuming background non-brain is zero in image
                brain_mask = np.any(image_data != 0, axis=-1)
                class_mask = brain_mask & (~any_tumor_mask)
            else:
                class_mask = (mask_data[..., target_ch] > 0)

            if np.any(class_mask):
                # Extract (N_voxels, 4) matrix of modalities
                voxels = norm_slice[class_mask]
                accumulated_voxels[c_name].append(voxels)

# 4. Fit 4D Multivariate Skew-t per Class
skew_t_params = {}

print("\nFitting 4D Multivariate Skew-t models via Maximum Likelihood Estimation...\n")

for c_name in TARGET_CLASSES:
    chunks = accumulated_voxels[c_name]
    if not chunks:
        print(f"Warning: No voxels accumulated for class '{c_name}'. Skipping...")
        continue

    # Merge into a single 2D array of shape (N_voxels, 4)
    all_voxels = np.concatenate(chunks, axis=0)
    total_found = len(all_voxels)

    # Subsample voxels for computational efficiency in MLE fitting if dataset is huge
    if total_found > MAX_VOXELS_PER_CLASS:
        idx = np.random.choice(total_found, size=MAX_VOXELS_PER_CLASS, replace=False)
        X_fit = all_voxels[idx]
    else:
        X_fit = all_voxels

    print(f"[{c_name}] Fitting model on {len(X_fit):,} voxels (out of {total_found:,} total)...")

    # Call R's 'sn' package: Fit Multivariate Skew-t using Maximum Penalized Likelihood (mst_mple)
    try:
        r_fit = sn.mst_mple(y=X_fit)
        dp = r_fit.rx2('dp')  # Extract direct parameter list

        # Extract parameters into NumPy arrays
        xi = np.array(dp.rx2('beta')).flatten()     # Location vector (4,)
        Omega = np.array(dp.rx2('Omega'))           # Dispersion/Scale matrix (4, 4)
        alpha = np.array(dp.rx2('alpha')).flatten()  # Skewness shape vector (4,)
        nu = float(np.array(dp.rx2('df'))[0])       # Degrees of Freedom (Kurtosis scalar)

        # Save parameters into dictionary
        skew_t_params[f"{c_name}_xi"] = xi
        skew_t_params[f"{c_name}_Omega"] = Omega
        skew_t_params[f"{c_name}_alpha"] = alpha
        skew_t_params[f"{c_name}_nu"] = nu

        print(f"  -> Location (xi) : {np.round(xi, 3)}")
        print(f"  -> Skewness (alpha): {np.round(alpha, 3)}")
        print(f"  -> DoF / Kurtosis (nu): {nu:.2f}\n")

    except Exception as e:
        print(f"  [ERROR] Fitting failed for class '{c_name}': {e}\n")

# 5. Save all parameters to a compressed .npz archive
np.savez_compressed(OUTPUT_PARAM_FILE, **skew_t_params)
print(f"Successfully saved all 4D Skew-t class parameters to:\n  '{OUTPUT_PARAM_FILE}'")