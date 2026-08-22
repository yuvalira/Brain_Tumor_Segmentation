import os
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from config import DATA_ROOT, PROJECT_ROOT, NUMBER_OF_VOLUMES, NUMBER_OF_SLICES


def compute_z_score_stats(blur_sigma=2.0):
    dataset_dir = os.path.join(
        DATA_ROOT,
        "MRI_2026_datasets",
        "Brats",
        "BraTS2020_training_data",
        "content",
        "data",
    )
    utilities_dir = os.path.join(PROJECT_ROOT, "utilities")
    os.makedirs(utilities_dir, exist_ok=True)

    means = np.zeros((NUMBER_OF_VOLUMES + 1, 9), dtype=np.float64)
    stds = np.zeros((NUMBER_OF_VOLUMES + 1, 9), dtype=np.float64)

    print(f"Computing 3D z-score stats across {NUMBER_OF_VOLUMES} volumes...")

    for vol_num in range(1, NUMBER_OF_VOLUMES + 1):
        vol_raw_and_dist_voxels = []
        vol_symmetry_voxels = []

        for slice_num in range(NUMBER_OF_SLICES):
            file_path = os.path.join(dataset_dir, f"volume_{vol_num}_slice_{slice_num}.h5")
            if not os.path.exists(file_path):
                continue

            with h5py.File(file_path, "r") as f:
                image = f["image"][:].astype(np.float64)

            brain_mask = np.any(image > 1e-8, axis=-1)
            if not np.any(brain_mask):
                continue

            # 1. Symmetry NDI & Mask
            mirrored_brain_mask = np.flipud(brain_mask)
            symmetric_brain_mask = brain_mask & mirrored_brain_mask

            blurred_raw = gaussian_filter(image, sigma=(blur_sigma, blur_sigma, 0.0))
            mirrored_raw = np.flipud(blurred_raw)
            denom = np.abs(blurred_raw) + np.abs(mirrored_raw) + 1e-4
            ndi_raw = np.clip((blurred_raw - mirrored_raw) / denom, -1.0, 1.0)

            # 2. Boundary Distance
            dist_map = distance_transform_edt(brain_mask)
            max_depth = np.max(dist_map)
            normalized_depth = dist_map / max_depth if max_depth > 0 else np.zeros_like(dist_map)

            # Raw (0-3) + Distance (8) pool across full brain_mask
            raw_and_dist = np.dstack([image, normalized_depth[:, :, np.newaxis]])
            vol_raw_and_dist_voxels.append(raw_and_dist[brain_mask])

            # Symmetry channels (4-7) pool STRICTLY within symmetric_brain_mask
            if np.any(symmetric_brain_mask):
                vol_symmetry_voxels.append(ndi_raw[symmetric_brain_mask])

        # Compute volume-level statistics
        if len(vol_raw_and_dist_voxels) > 0:
            flat_raw_dist = np.vstack(vol_raw_and_dist_voxels)  # Shape: (N, 5) -> Channels [0, 1, 2, 3, 8]

            # Channels 0-3 (Raw Modalities)
            means[vol_num, :4] = np.mean(flat_raw_dist[:, :4], axis=0)
            stds[vol_num, :4] = np.std(flat_raw_dist[:, :4], axis=0)

            # Channel 8 (Boundary Distance)
            means[vol_num, 8] = np.mean(flat_raw_dist[:, 4])
            stds[vol_num, 8] = np.std(flat_raw_dist[:, 4])

        if len(vol_symmetry_voxels) > 0:
            flat_sym = np.vstack(vol_symmetry_voxels)  # Shape: (M, 4) -> Channels [4, 5, 6, 7]
            means[vol_num, 4:8] = np.mean(flat_sym, axis=0)
            stds[vol_num, 4:8] = np.std(flat_sym, axis=0)

        # Variance zero-guard
        stds[vol_num] = np.where(stds[vol_num] < 1e-6, 1.0, stds[vol_num])

        if vol_num % 25 == 0 or vol_num == NUMBER_OF_VOLUMES:
            print(f"Processed Volume {vol_num}/{NUMBER_OF_VOLUMES}")

    np.save(os.path.join(utilities_dir, "volume_means.npy"), means)
    np.save(os.path.join(utilities_dir, "volume_stds.npy"), stds)
    print(f"Saved 9D volume statistics successfully to {utilities_dir}")


if __name__ == "__main__":
    compute_z_score_stats()