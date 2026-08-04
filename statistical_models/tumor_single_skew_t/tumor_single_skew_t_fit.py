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
    """
    Fits one multivariate skew-t distribution using R's sn::mst.mple().
    """

    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, received shape {data.shape}."
        )

    if data.shape[0] == 0:
        raise ValueError(
            "Cannot fit a skew-t distribution to an empty array."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "Skew-t fitting data contain NaN or infinite values."
        )

    design = np.ones(
        (data.shape[0], 1),
        dtype=np.float64,
    )

    with localconverter(
        default_converter + numpy2ri.converter
    ):
        r_design = ro.conversion.py2rpy(design)
        r_data = ro.conversion.py2rpy(data)

    fit = sn.mst_mple(
        x=r_design,
        y=r_data,
    )

    dp = fit.rx2("dp")

    beta_r = dp.rx2("beta")
    omega_r = dp.rx2("Omega")
    alpha_r = dp.rx2("alpha")
    nu_r = dp.rx2("nu")
    log_likelihood_r = fit.rx2("logL")

    with localconverter(
        default_converter + numpy2ri.converter
    ):
        location = np.asarray(
            ro.conversion.rpy2py(beta_r),
            dtype=np.float64,
        ).reshape(-1)

        dispersion = np.asarray(
            ro.conversion.rpy2py(omega_r),
            dtype=np.float64,
        )

        skewness = np.asarray(
            ro.conversion.rpy2py(alpha_r),
            dtype=np.float64,
        ).reshape(-1)

        degrees_of_freedom = float(
            np.asarray(
                ro.conversion.rpy2py(nu_r),
                dtype=np.float64,
            ).reshape(-1)[0]
        )

        log_likelihood = float(
            np.asarray(
                ro.conversion.rpy2py(
                    log_likelihood_r
                ),
                dtype=np.float64,
            ).reshape(-1)[0]
        )

    return (
        location,
        dispersion,
        skewness,
        degrees_of_freedom,
        log_likelihood,
    )


def estimate_tumor_skew_t_parameters(
    dataset_base_path,
    volume_means,
    volume_stds,
    max_train_vol=MAX_TRAINING_VOLUME,
    total_slices=MAX_SLICE + 1,
    max_fit_samples=100_000,
    random_seed=42,
    clip_min=-6.0,
    clip_max=6.0,
    output_path=os.path.join(
        PARAMS_OUTPUT_PATH,
        "tumor_skew_t_parameters.npz",
    ),
):
    """
    Fits one multivariate skew-t distribution to each tumor class:

        NCR/NET
        ED
        ET

    Priors are calculated relative to all brain voxels.

    Because fitting millions of voxels in R is impractical, at most
    max_fit_samples voxels from each tumor class are retained for fitting.
    """

    print("\nEstimating tumor skew-t parameters...")

    num_classes = 3
    num_dims = 4

    class_names = np.array(
        [
            "NCR_NET",
            "ED",
            "ET",
        ]
    )

    modalities = np.array(
        [
            "T1",
            "T1ce",
            "T2",
            "FLAIR",
        ]
    )

    rng = np.random.default_rng(random_seed)

    total_brain_voxels = np.uint64(0)

    running_count = np.zeros(
        num_classes,
        dtype=np.uint64,
    )

    tumor_pixel_pools = [
        [],
        [],
        [],
    ]

    retained_sample_counts = np.zeros(
        num_classes,
        dtype=np.int64,
    )

    volume_sequence = np.arange(
        1,
        max_train_vol + 1,
    )

    rng.shuffle(volume_sequence)

    for processed_idx, vol_num in enumerate(
        volume_sequence,
        start=1,
    ):
        for slice_num in range(total_slices):
            (
                norm_slice,
                brain_mask,
                mask_slice,
            ) = load_and_normalize_slice(
                dataset_base_path,
                int(vol_num),
                slice_num,
                volume_means,
                volume_stds,
            )

            if not np.any(brain_mask):
                continue

            image_flat = norm_slice.reshape(
                -1,
                num_dims,
            )

            mask_flat = mask_slice.reshape(
                -1,
                num_classes,
            )

            brain_flat = brain_mask.ravel()

            total_brain_voxels += np.count_nonzero(
                brain_flat
            )

            class_masks = [
                (mask_flat[:, 0] > 0) & brain_flat,
                (mask_flat[:, 1] > 0) & brain_flat,
                (mask_flat[:, 2] > 0) & brain_flat,
            ]

            for class_idx, class_mask in enumerate(
                class_masks
            ):
                if not np.any(class_mask):
                    continue

                pixels = image_flat[class_mask]

                finite_rows = np.all(
                    np.isfinite(pixels),
                    axis=1,
                )

                pixels = pixels[finite_rows]

                if pixels.shape[0] == 0:
                    continue

                running_count[class_idx] += (
                    pixels.shape[0]
                )

                remaining_capacity = (
                    max_fit_samples
                    - retained_sample_counts[class_idx]
                )

                if remaining_capacity <= 0:
                    continue

                if pixels.shape[0] > remaining_capacity:
                    selected_indices = rng.choice(
                        pixels.shape[0],
                        size=remaining_capacity,
                        replace=False,
                    )

                    pixels = pixels[selected_indices]

                tumor_pixel_pools[class_idx].append(
                    pixels
                )

                retained_sample_counts[class_idx] += (
                    pixels.shape[0]
                )

        if (
            processed_idx % 50 == 0
            or processed_idx == max_train_vol
        ):
            print(
                f"Processed "
                f"{processed_idx}/{max_train_vol} volumes"
            )

    if total_brain_voxels == 0:
        raise RuntimeError(
            "No brain voxels were loaded. Check the dataset path "
            "and HDF5 file naming."
        )

    if np.any(running_count == 0):
        raise RuntimeError(
            "One or more tumor classes contain no voxels. "
            f"Counts: {running_count}"
        )

    if np.any(retained_sample_counts == 0):
        raise RuntimeError(
            "One or more tumor classes have no retained fitting samples. "
            f"Retained counts: {retained_sample_counts}"
        )

    tumor_priors = (
        running_count.astype(np.float64)
        / float(total_brain_voxels)
    )

    tumor_locations = np.zeros(
        (num_classes, num_dims),
        dtype=np.float64,
    )

    tumor_dispersions = np.zeros(
        (num_classes, num_dims, num_dims),
        dtype=np.float64,
    )

    tumor_skewness = np.zeros(
        (num_classes, num_dims),
        dtype=np.float64,
    )

    tumor_degrees_of_freedom = np.zeros(
        num_classes,
        dtype=np.float64,
    )

    tumor_log_likelihoods = np.zeros(
        num_classes,
        dtype=np.float64,
    )

    print("\nStarting R skew-t fits...")

    for class_idx, class_name in enumerate(
        class_names
    ):
        class_pixels = np.vstack(
            tumor_pixel_pools[class_idx]
        )

        class_pixels = np.clip(
            class_pixels,
            clip_min,
            clip_max,
        ).astype(np.float64)

        print(
            f"\nFitting {class_name}: "
            f"{class_pixels.shape[0]} samples"
        )

        (
            tumor_locations[class_idx],
            tumor_dispersions[class_idx],
            tumor_skewness[class_idx],
            tumor_degrees_of_freedom[class_idx],
            tumor_log_likelihoods[class_idx],
        ) = fit_multivariate_skew_t(
            class_pixels
        )

        print(
            f"  Location: "
            f"{tumor_locations[class_idx]}"
        )

        print(
            f"  Skewness: "
            f"{tumor_skewness[class_idx]}"
        )

        print(
            f"  Degrees of freedom: "
            f"{tumor_degrees_of_freedom[class_idx]:.6f}"
        )

        print(
            f"  Log-likelihood: "
            f"{tumor_log_likelihoods[class_idx]:.6f}"
        )

    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_path,
        priors=tumor_priors,
        locations=tumor_locations,
        dispersions=tumor_dispersions,
        skewness=tumor_skewness,
        degrees_of_freedom=(
            tumor_degrees_of_freedom
        ),
        log_likelihoods=tumor_log_likelihoods,
        counts=running_count,
        fit_sample_counts=retained_sample_counts,
        total_brain_voxels=total_brain_voxels,
        class_names=class_names,
        modalities=modalities,
        clip_min=np.float64(clip_min),
        clip_max=np.float64(clip_max),
    )

    print("\nTumor skew-t parameters saved to")
    print(output_path)

    print("\nTumor priors")
    print("-----------------------")

    for class_idx, class_name in enumerate(
        class_names
    ):
        print(
            f"{class_name:8s}: "
            f"{tumor_priors[class_idx]:.6f}"
        )

    print(
        f"Sum      : "
        f"{tumor_priors.sum():.6f}"
    )

    print("\nVoxel counts")
    print("-----------------------")

    for class_idx, class_name in enumerate(
        class_names
    ):
        print(
            f"{class_name:8s}: "
            f"{running_count[class_idx]}"
        )

    print("\nFitting sample counts")
    print("-----------------------")

    for class_idx, class_name in enumerate(
        class_names
    ):
        print(
            f"{class_name:8s}: "
            f"{retained_sample_counts[class_idx]}"
        )

    return (
        tumor_priors,
        tumor_locations,
        tumor_dispersions,
        tumor_skewness,
        tumor_degrees_of_freedom,
        tumor_log_likelihoods,
    )


if __name__ == "__main__":
    stats_path = PROJECT_ROOT / PARAMS_OUTPUT_PATH
    dataset_path = PROJECT_ROOT / DATASET_PATH

    output_path = (
        stats_path
        / "tumor_skew_t_parameters.npz"
    )

    print("Project root:", PROJECT_ROOT)
    print("Statistics path:", stats_path)
    print("Dataset path:", dataset_path)
    print("Dataset exists:", dataset_path.exists())

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found at '{dataset_path}'."
        )

    volume_means, volume_stds = load_volume_stats(
        stats_path
    )

    (
        tumor_priors,
        tumor_locations,
        tumor_dispersions,
        tumor_skewness,
        tumor_degrees_of_freedom,
        tumor_log_likelihoods,
    ) = estimate_tumor_skew_t_parameters(
        dataset_base_path=dataset_path,
        volume_means=volume_means,
        volume_stds=volume_stds,
        max_train_vol=MAX_TRAINING_VOLUME,
        total_slices=MAX_SLICE + 1,
        max_fit_samples=100_000,
        random_seed=42,
        clip_min=-6.0,
        clip_max=6.0,
        output_path=output_path,
    )

    print(
        "\nFinished estimating tumor skew-t parameters."
    )