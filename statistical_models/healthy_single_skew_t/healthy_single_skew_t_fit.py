import os
from pathlib import Path
import numpy as np

import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from config_parameters import *
from utils import load_and_normalize_slice, load_volume_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sn = importr("sn")


def fit_multivariate_skew_t(data):
    """Fits one multivariate skew-t distribution using R's sn::mst.mple()."""
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, received shape {data.shape}.")

    if data.shape[0] == 0:
        raise ValueError("Cannot fit a skew-t distribution to an empty array.")

    if not np.all(np.isfinite(data)):
        raise ValueError("Skew-t fitting data contain NaN or infinite values.")

    design = np.ones((data.shape[0], 1), dtype=np.float64)

    with localconverter(default_converter + numpy2ri.converter):
        r_design = ro.conversion.py2rpy(design)
        r_data = ro.conversion.py2rpy(data)

    fit = sn.mst_mple(x=r_design, y=r_data)
    dp = fit.rx2("dp")

    beta_r = dp.rx2("beta")
    omega_r = dp.rx2("Omega")
    alpha_r = dp.rx2("alpha")
    nu_r = dp.rx2("nu")
    log_likelihood_r = fit.rx2("logL")

    with localconverter(default_converter + numpy2ri.converter):
        location = np.asarray(
            ro.conversion.rpy2py(beta_r), dtype=np.float64
        ).reshape(-1)

        dispersion = np.asarray(
            ro.conversion.rpy2py(omega_r), dtype=np.float64
        )

        skewness = np.asarray(
            ro.conversion.rpy2py(alpha_r), dtype=np.float64
        ).reshape(-1)

        degrees_of_freedom = float(
            np.asarray(ro.conversion.rpy2py(nu_r), dtype=np.float64).reshape(-1)[0]
        )

        log_likelihood = float(
            np.asarray(
                ro.conversion.rpy2py(log_likelihood_r), dtype=np.float64
            ).reshape(-1)[0]
        )

    return location, dispersion, skewness, degrees_of_freedom, log_likelihood


def estimate_healthy_skew_t_parameters(
    dataset_base_path,
    volume_means,
    volume_stds,
    max_train_vol=MAX_TRAINING_VOLUME,
    total_slices=MAX_SLICE + 1,
    max_fit_samples=3_000_000,
    random_seed=42,
    clip_min=-6.0,
    clip_max=6.0,
    output_path=os.path.join(
        PARAMS_OUTPUT_PATH,
        "healthy_skew_t_parameters.npz",
    ),
):
    """Accumulates healthy voxels across training volumes by picking 1 random slice per volume

    sequential from 1 to max_train_vol until max_fit_samples is reached.
    Fits a multivariate skew-t distribution and saves parameters to an .npz file.
    """
    print("\nEstimating Healthy Skew-t parameters...")

    num_dims = 4
    modalities = np.array(["T1", "T1ce", "T2", "FLAIR"])
    rng = np.random.default_rng(random_seed)

    total_brain_voxels = np.uint64(0)
    total_healthy_voxels = np.uint64(0)
    healthy_pixel_pool = []
    retained_sample_count = 0
    volumes_used_count = 0

    # Sequential iteration starting at volume 1
    for vol_num in range(1, max_train_vol + 1):
        if retained_sample_count >= max_fit_samples:
            break

        # Pick one random slice from the current volume
        random_slice = rng.integers(0, total_slices)

        (
            norm_slice,
            brain_mask,
            mask_slice,
        ) = load_and_normalize_slice(
            dataset_base_path,
            vol_num,
            random_slice,
            volume_means,
            volume_stds,
        )

        if not np.any(brain_mask):
            continue

        image_flat = norm_slice.reshape(-1, num_dims)
        mask_flat = mask_slice.reshape(-1, 3)
        brain_flat = brain_mask.ravel()

        # Track total brain voxels for prior denominator
        num_brain_pixels = np.count_nonzero(brain_flat)
        total_brain_voxels += np.uint64(num_brain_pixels)

        # Healthy mask: Inside brain_mask AND sum of tumor classes == 0
        healthy_mask_flat = brain_flat & (np.sum(mask_flat, axis=-1) == 0)

        if not np.any(healthy_mask_flat):
            continue

        pixels = image_flat[healthy_mask_flat]

        # Filter finite values
        finite_rows = np.all(np.isfinite(pixels), axis=1)
        pixels = pixels[finite_rows]

        if pixels.shape[0] == 0:
            continue

        total_healthy_voxels += np.uint64(pixels.shape[0])

        remaining_capacity = max_fit_samples - retained_sample_count

        if pixels.shape[0] > remaining_capacity:
            selected_indices = rng.choice(
                pixels.shape[0],
                size=remaining_capacity,
                replace=False,
            )
            pixels = pixels[selected_indices]

        healthy_pixel_pool.append(pixels)
        retained_sample_count += pixels.shape[0]
        volumes_used_count += 1

        if vol_num % 50 == 0 or vol_num == max_train_vol:
            print(
                f"Processed volume {vol_num}/{max_train_vol} "
                f"| Accumulated: {retained_sample_count:,}/{max_fit_samples:,} samples"
            )

    print(
        f"\n[+] Pixel sampling complete. Used {volumes_used_count} volumes to reach {retained_sample_count:,} pixels."
    )

    if total_brain_voxels == 0:
        raise RuntimeError("No brain voxels were loaded.")

    if total_healthy_voxels == 0 or retained_sample_count == 0:
        raise RuntimeError("No healthy voxels were accumulated for fitting.")

    # Calculate Healthy Prior
    healthy_prior = float(total_healthy_voxels) / float(total_brain_voxels)

    # Combine sampled pixels and clip outliers
    healthy_pixels_all = np.vstack(healthy_pixel_pool)
    healthy_pixels_all = np.clip(healthy_pixels_all, clip_min, clip_max).astype(
        np.float64
    )

    print(
        f"\nFitting Healthy Skew-t Distribution on {healthy_pixels_all.shape[0]:,} samples..."
    )

    (
        location,
        dispersion,
        skewness,
        degrees_of_freedom,
        log_likelihood,
    ) = fit_multivariate_skew_t(healthy_pixels_all)

    print("\nFit Results:")
    print(f"  Location          : {location}")
    print(f"  Skewness          : {skewness}")
    print(f"  Degrees of Freedom: {degrees_of_freedom:.6f}")
    print(f"  Log-Likelihood    : {log_likelihood:.6f}")

    # Output path resolution
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        healthy_prior=np.float64(healthy_prior),
        location=location,
        dispersion=dispersion,
        skewness=skewness,
        degrees_of_freedom=np.float64(degrees_of_freedom),
        log_likelihood=np.float64(log_likelihood),
        healthy_voxel_count=total_healthy_voxels,
        total_brain_voxels=total_brain_voxels,
        fit_sample_count=np.int64(retained_sample_count),
        volumes_used=np.int64(volumes_used_count),
        modalities=modalities,
        clip_min=np.float64(clip_min),
        clip_max=np.float64(clip_max),
    )

    print(f"\nHealthy Skew-t parameters saved to: {output_path}")

    print("\nSummary")
    print("-----------------------------------")
    print(f"Volumes Used         : {volumes_used_count} / {max_train_vol}")
    print(f"Total Brain Voxels   : {total_brain_voxels:,}")
    print(f"Total Healthy Voxels : {total_healthy_voxels:,}")
    print(f"Retained Fit Samples : {retained_sample_count:,}")
    print(f"Healthy Prior        : {healthy_prior:.6f}")

    return (
        healthy_prior,
        location,
        dispersion,
        skewness,
        degrees_of_freedom,
        log_likelihood,
    )


if __name__ == "__main__":
    stats_path = PROJECT_ROOT / PARAMS_OUTPUT_PATH
    dataset_path = PROJECT_ROOT / DATASET_PATH
    output_path = stats_path / "healthy_skew_t_parameters.npz"

    print("Project root:", PROJECT_ROOT)
    print("Statistics path:", stats_path)
    print("Dataset path:", dataset_path)
    print("Dataset exists:", dataset_path.exists())

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found at '{dataset_path}'."
        )

    volume_means, volume_stds = load_volume_stats(stats_path)

    (
        healthy_prior,
        location,
        dispersion,
        skewness,
        degrees_of_freedom,
        log_likelihood,
    ) = estimate_healthy_skew_t_parameters(
        dataset_base_path=dataset_path,
        volume_means=volume_means,
        volume_stds=volume_stds,
        max_train_vol=MAX_TRAINING_VOLUME,
        total_slices=MAX_SLICE + 1,
        max_fit_samples=3_000_000,
        random_seed=42,
        clip_min=-6.0,
        clip_max=6.0,
        output_path=output_path,
    )

    print("\nFinished estimating healthy skew-t parameters.")