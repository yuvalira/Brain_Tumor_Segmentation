import csv
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from sklearn.mixture import GaussianMixture


# =============================================================================
# ADD PROJECT ROOT TO PYTHON PATH
# =============================================================================

# Current file:
# Brain_Tumor_Segmentation/statistical_models/GMM_components_check.py
#
# Project root:
# Brain_Tumor_Segmentation/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make config_parameters.py and utilities importable.
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config_parameters.py"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"config_parameters.py was not found at: {CONFIG_PATH}"
    )


from config_parameters import (
    BRAIN_MASK_THRESHOLD,
    DATASET_PATH,
    FIRST_VOLUME,
    GMM_COVARIANCE_REGULARIZATION,
    GMM_HEALTHY_MAX_COMPONENTS,
    GMM_HEALTHY_MIN_COMPONENTS,
    GMM_HEALTHY_SAMPLES_PER_VOLUME,
    GMM_MAX_ITERATIONS,
    GMM_MAX_TRAIN_SAMPLES_HEALTHY,
    GMM_MAX_TRAIN_SAMPLES_TUMOR,
    GMM_MAX_VALIDATION_SAMPLES_HEALTHY,
    GMM_MAX_VALIDATION_SAMPLES_TUMOR,
    GMM_MIN_COMPONENT_WEIGHT,
    GMM_N_INITIALIZATIONS,
    GMM_SELECTION_REPEATS,
    GMM_TRAIN_FRACTION,
    GMM_TUMOR_MAX_COMPONENTS,
    GMM_TUMOR_MIN_COMPONENTS,
    GMM_TUMOR_SAMPLES_PER_VOLUME,
    MAX_TRAINING_VOLUME,
    NUM_MODALITIES,
    NUM_SLICES,
    RANDOM_SEED,
    VOLUME_MEANS_PATH,
    VOLUME_STDS_PATH,
    ZSCORE_CLIP_MAX,
    ZSCORE_CLIP_MIN,
)

# =============================================================================
# CLASS DEFINITIONS
# =============================================================================

CLASS_NAMES = (
    "HEALTHY",
    "NCR_NET",
    "ED",
    "ET",
)

NUM_CLASSES = len(CLASS_NAMES)

HEALTHY_CLASS_INDEX = 0
TUMOR_CLASS_INDICES = (1, 2, 3)

RESULTS_PATH = (
    Path(__file__).resolve().parent
    / "gmm_component_selection_results.csv"
)


# =============================================================================
# PATH RESOLUTION
# =============================================================================

def resolve_path(path):
    """
    Resolves a configured path.

    Absolute paths are returned unchanged. Relative dataset paths are first
    interpreted relative to the workspace containing the repository.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    workspace_candidate = PROJECT_ROOT.parent / path

    if workspace_candidate.exists():
        return workspace_candidate

    return PROJECT_ROOT / path


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_normalize_slice_for_gmm(
    volume_num,
    slice_num,
    dataset_path,
    volume_means,
    volume_stds,
):
    """
    Loads and normalizes one MRI slice.

    Returns
    -------
    normalized_image : np.ndarray
        Shape (H, W, 4).

    brain_mask : np.ndarray
        Shape (H, W).

    tumor_mask : np.ndarray
        Shape (H, W, 3).

    If a file is missing or invalid, returns (None, None, None).
    """
    file_path = (
        dataset_path
        / f"volume_{volume_num}_slice_{slice_num}.h5"
    )

    if not file_path.exists():
        return None, None, None

    try:
        with h5py.File(file_path, "r") as file:
            image = file["image"][:].astype(np.float64)
            tumor_mask = file["mask"][:].astype(np.float32)

    except (OSError, KeyError) as error:
        print(
            f"Could not load {file_path}: {error}"
        )
        return None, None, None

    brain_mask = np.any(
        image > BRAIN_MASK_THRESHOLD,
        axis=-1,
    )

    normalized_image = np.zeros_like(
        image,
        dtype=np.float64,
    )

    if np.any(brain_mask):
        mean = volume_means[volume_num]
        std = volume_stds[volume_num]

        normalized_image[brain_mask] = (
            image[brain_mask] - mean
        ) / std

        normalized_image[brain_mask] = np.clip(
            normalized_image[brain_mask],
            ZSCORE_CLIP_MIN,
            ZSCORE_CLIP_MAX,
        )

    return (
        normalized_image,
        brain_mask,
        tumor_mask,
    )


# =============================================================================
# SAMPLING
# =============================================================================

def randomly_limit_rows(
    data,
    maximum_rows,
    rng,
):
    """Randomly limits a 2D array to at most maximum_rows."""
    if data.shape[0] <= maximum_rows:
        return data

    selected_indices = rng.choice(
        data.shape[0],
        size=maximum_rows,
        replace=False,
    )

    return data[selected_indices]


def extract_class_pixels(
    normalized_image,
    brain_mask,
    tumor_mask,
):
    """
    Extracts pixels for:

        0: Healthy
        1: NCR/NET
        2: ED
        3: ET
    """
    healthy_mask = (
        brain_mask
        & ~np.any(tumor_mask > 0, axis=-1)
    )

    class_masks = (
        healthy_mask,
        brain_mask & (tumor_mask[:, :, 0] > 0),
        brain_mask & (tumor_mask[:, :, 1] > 0),
        brain_mask & (tumor_mask[:, :, 2] > 0),
    )

    class_pixels = []

    for class_mask in class_masks:
        pixels = normalized_image[class_mask]

        if pixels.size > 0:
            finite_rows = np.all(
                np.isfinite(pixels),
                axis=1,
            )
            pixels = pixels[finite_rows]

        class_pixels.append(pixels)

    return class_pixels


def collect_volume_samples(
    volume_num,
    dataset_path,
    volume_means,
    volume_stds,
    rng,
):
    """
    Collects capped samples for each tissue class from one volume.

    A small sample is first retained from each slice so that one large slice
    does not dominate the entire volume.
    """
    per_slice_cap = 500

    class_buffers = [
        [] for _ in range(NUM_CLASSES)
    ]

    for slice_num in range(NUM_SLICES):
        (
            normalized_image,
            brain_mask,
            tumor_mask,
        ) = load_and_normalize_slice_for_gmm(
            volume_num=volume_num,
            slice_num=slice_num,
            dataset_path=dataset_path,
            volume_means=volume_means,
            volume_stds=volume_stds,
        )

        if normalized_image is None:
            continue

        if not np.any(brain_mask):
            continue

        class_pixels = extract_class_pixels(
            normalized_image=normalized_image,
            brain_mask=brain_mask,
            tumor_mask=tumor_mask,
        )

        for class_index, pixels in enumerate(
            class_pixels
        ):
            if pixels.shape[0] == 0:
                continue

            pixels = randomly_limit_rows(
                data=pixels,
                maximum_rows=per_slice_cap,
                rng=rng,
            )

            class_buffers[class_index].append(
                pixels
            )

    volume_samples = []

    for class_index, buffers in enumerate(
        class_buffers
    ):
        if not buffers:
            volume_samples.append(
                np.empty(
                    (0, NUM_MODALITIES),
                    dtype=np.float64,
                )
            )
            continue

        pixels = np.vstack(buffers)

        if class_index == HEALTHY_CLASS_INDEX:
            maximum_rows = (
                GMM_HEALTHY_SAMPLES_PER_VOLUME
            )
        else:
            maximum_rows = (
                GMM_TUMOR_SAMPLES_PER_VOLUME
            )

        pixels = randomly_limit_rows(
            data=pixels,
            maximum_rows=maximum_rows,
            rng=rng,
        )

        volume_samples.append(pixels)

    return volume_samples


def collect_all_training_samples(
    dataset_path,
    volume_means,
    volume_stds,
):
    """
    Samples pixels from every training volume once.

    Returns
    -------
    samples_by_volume : dict
        samples_by_volume[volume_num][class_index] gives an array of shape
        (N, 4).
    """
    samples_by_volume = {}

    for volume_num in range(
        FIRST_VOLUME,
        MAX_TRAINING_VOLUME + 1,
    ):
        print(
            f"Sampling volume "
            f"{volume_num}/{MAX_TRAINING_VOLUME}"
        )

        volume_rng = np.random.default_rng(
            RANDOM_SEED + volume_num
        )

        volume_samples = collect_volume_samples(
            volume_num=volume_num,
            dataset_path=dataset_path,
            volume_means=volume_means,
            volume_stds=volume_stds,
            rng=volume_rng,
        )

        samples_by_volume[volume_num] = (
            volume_samples
        )

        summary = {
            CLASS_NAMES[class_index]: (
                volume_samples[class_index].shape[0]
            )
            for class_index in range(NUM_CLASSES)
        }

        print("  Samples:", summary)

    return samples_by_volume


def combine_volume_samples(
    samples_by_volume,
    volume_ids,
    class_index,
    maximum_rows,
    rng,
):
    """Combines samples from selected volumes for one class."""
    arrays = []

    for volume_num in volume_ids:
        pixels = samples_by_volume[
            int(volume_num)
        ][class_index]

        if pixels.shape[0] > 0:
            arrays.append(pixels)

    if not arrays:
        return np.empty(
            (0, NUM_MODALITIES),
            dtype=np.float64,
        )

    combined = np.vstack(arrays)

    return randomly_limit_rows(
        data=combined,
        maximum_rows=maximum_rows,
        rng=rng,
    )


# =============================================================================
# GMM MODEL SELECTION
# =============================================================================

def get_component_candidates(class_index):
    """Returns candidate component counts for a tissue class."""
    if class_index == HEALTHY_CLASS_INDEX:
        return range(
            GMM_HEALTHY_MIN_COMPONENTS,
            GMM_HEALTHY_MAX_COMPONENTS + 1,
        )

    return range(
        GMM_TUMOR_MIN_COMPONENTS,
        GMM_TUMOR_MAX_COMPONENTS + 1,
    )


def get_maximum_sample_counts(class_index):
    """Returns train and validation sample limits."""
    if class_index == HEALTHY_CLASS_INDEX:
        return (
            GMM_MAX_TRAIN_SAMPLES_HEALTHY,
            GMM_MAX_VALIDATION_SAMPLES_HEALTHY,
        )

    return (
        GMM_MAX_TRAIN_SAMPLES_TUMOR,
        GMM_MAX_VALIDATION_SAMPLES_TUMOR,
    )


def evaluate_gmm_candidates(
    training_pixels,
    validation_pixels,
    class_name,
    class_index,
    repeat_index,
):
    """Fits and evaluates all candidate GMM sizes for one class."""
    results = []

    for num_components in get_component_candidates(
        class_index
    ):
        if training_pixels.shape[0] < num_components:
            continue

        print(
            f"  {class_name}: fitting "
            f"K={num_components}"
        )

        model = GaussianMixture(
            n_components=num_components,
            covariance_type="full",
            tol=1e-3,
            reg_covar=(
                GMM_COVARIANCE_REGULARIZATION
            ),
            max_iter=GMM_MAX_ITERATIONS,
            n_init=GMM_N_INITIALIZATIONS,
            init_params="kmeans",
            random_state=(
                RANDOM_SEED
                + repeat_index * 100
                + class_index * 10
                + num_components
            ),
        )

        model.fit(training_pixels)

        result = {
            "repeat": repeat_index,
            "class_name": class_name,
            "class_index": class_index,
            "num_components": num_components,
            "num_training_samples": (
                training_pixels.shape[0]
            ),
            "num_validation_samples": (
                validation_pixels.shape[0]
            ),
            "bic": model.bic(training_pixels),
            "aic": model.aic(training_pixels),
            "training_log_likelihood": (
                model.score(training_pixels)
            ),
            "validation_log_likelihood": (
                model.score(validation_pixels)
            ),
            "converged": bool(model.converged_),
            "num_iterations": int(model.n_iter_),
            "minimum_component_weight": float(
                np.min(model.weights_)
            ),
        }

        results.append(result)

        print(
            f"    BIC={result['bic']:.2f}, "
            f"validation LL="
            f"{result['validation_log_likelihood']:.6f}, "
            f"min weight="
            f"{result['minimum_component_weight']:.6f}, "
            f"converged={result['converged']}"
        )

    return results


# =============================================================================
# RESULT SUMMARIZATION
# =============================================================================

def summarize_results(results):
    """Aggregates results across repeated patient-level splits."""
    grouped_results = defaultdict(list)

    for result in results:
        key = (
            result["class_name"],
            result["class_index"],
            result["num_components"],
        )
        grouped_results[key].append(result)

    summaries = []

    for key, rows in grouped_results.items():
        class_name, class_index, num_components = key

        bic_values = np.array(
            [row["bic"] for row in rows]
        )

        validation_values = np.array(
            [
                row["validation_log_likelihood"]
                for row in rows
            ]
        )

        minimum_weights = np.array(
            [
                row["minimum_component_weight"]
                for row in rows
            ]
        )

        converged_values = np.array(
            [
                float(row["converged"])
                for row in rows
            ]
        )

        ddof = 1 if len(rows) > 1 else 0

        summaries.append(
            {
                "class_name": class_name,
                "class_index": class_index,
                "num_components": num_components,
                "mean_bic": float(
                    np.mean(bic_values)
                ),
                "std_bic": float(
                    np.std(bic_values, ddof=ddof)
                ),
                "mean_validation_log_likelihood": float(
                    np.mean(validation_values)
                ),
                "std_validation_log_likelihood": float(
                    np.std(
                        validation_values,
                        ddof=ddof,
                    )
                ),
                "mean_minimum_component_weight": float(
                    np.mean(minimum_weights)
                ),
                "convergence_rate": float(
                    np.mean(converged_values)
                ),
                "num_repeats": len(rows),
            }
        )

    return summaries


def recommend_component_counts(summaries):
    """
    Suggests one component count per tissue class.

    It first removes unstable solutions. It then finds models whose validation
    likelihood is within one standard error of the best model. Among those,
    it prefers the BIC-optimal model when possible; otherwise, it selects the
    smallest near-best model.
    """
    recommendations = {}

    for class_name in CLASS_NAMES:
        class_rows = [
            row
            for row in summaries
            if row["class_name"] == class_name
        ]

        stable_rows = [
            row
            for row in class_rows
            if (
                row["convergence_rate"] >= 2.0 / 3.0
                and row[
                    "mean_minimum_component_weight"
                ] >= GMM_MIN_COMPONENT_WEIGHT
            )
        ]

        if not stable_rows:
            stable_rows = class_rows
            print(
                f"Warning: no fully stable candidate "
                f"was found for {class_name}."
            )

        best_validation_row = max(
            stable_rows,
            key=lambda row: row[
                "mean_validation_log_likelihood"
            ],
        )

        num_repeats = max(
            best_validation_row["num_repeats"],
            1,
        )

        validation_standard_error = (
            best_validation_row[
                "std_validation_log_likelihood"
            ]
            / np.sqrt(num_repeats)
        )

        validation_limit = (
            best_validation_row[
                "mean_validation_log_likelihood"
            ]
            - validation_standard_error
        )

        near_best_rows = [
            row
            for row in stable_rows
            if row[
                "mean_validation_log_likelihood"
            ] >= validation_limit
        ]

        bic_best_row = min(
            stable_rows,
            key=lambda row: row["mean_bic"],
        )

        if bic_best_row in near_best_rows:
            selected_row = bic_best_row
        else:
            selected_row = min(
                near_best_rows,
                key=lambda row: row[
                    "num_components"
                ],
            )

        recommendations[class_name] = selected_row

    return recommendations


# =============================================================================
# SAVING
# =============================================================================

def save_results(results, results_path):
    """Saves all repeat-level results to CSV."""
    if not results:
        raise RuntimeError(
            "No GMM selection results were generated."
        )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        results_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nResults saved to: {results_path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    dataset_path = resolve_path(DATASET_PATH)
    means_path = resolve_path(VOLUME_MEANS_PATH)
    stds_path = resolve_path(VOLUME_STDS_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{dataset_path}"
        )

    if not means_path.exists():
        raise FileNotFoundError(
            f"Volume means not found: {means_path}"
        )

    if not stds_path.exists():
        raise FileNotFoundError(
            f"Volume standard deviations not found: "
            f"{stds_path}"
        )

    volume_means = np.load(means_path)
    volume_stds = np.load(stds_path)

    print("Dataset path:", dataset_path)
    print("Collecting training samples...")

    samples_by_volume = (
        collect_all_training_samples(
            dataset_path=dataset_path,
            volume_means=volume_means,
            volume_stds=volume_stds,
        )
    )

    training_volume_ids = np.arange(
        FIRST_VOLUME,
        MAX_TRAINING_VOLUME + 1,
    )

    all_results = []

    for repeat_index in range(
        GMM_SELECTION_REPEATS
    ):
        print(
            "\n"
            + "=" * 70
            + f"\nPATIENT SPLIT REPEAT "
            f"{repeat_index + 1}/"
            f"{GMM_SELECTION_REPEATS}"
            + "\n"
            + "=" * 70
        )

        split_rng = np.random.default_rng(
            RANDOM_SEED + repeat_index
        )

        shuffled_volumes = (
            training_volume_ids.copy()
        )
        split_rng.shuffle(shuffled_volumes)

        split_index = int(
            GMM_TRAIN_FRACTION
            * shuffled_volumes.shape[0]
        )

        fit_volume_ids = (
            shuffled_volumes[:split_index]
        )

        validation_volume_ids = (
            shuffled_volumes[split_index:]
        )

        for class_index, class_name in enumerate(
            CLASS_NAMES
        ):
            (
                maximum_training_samples,
                maximum_validation_samples,
            ) = get_maximum_sample_counts(
                class_index
            )

            training_pixels = (
                combine_volume_samples(
                    samples_by_volume=(
                        samples_by_volume
                    ),
                    volume_ids=fit_volume_ids,
                    class_index=class_index,
                    maximum_rows=(
                        maximum_training_samples
                    ),
                    rng=split_rng,
                )
            )

            validation_pixels = (
                combine_volume_samples(
                    samples_by_volume=(
                        samples_by_volume
                    ),
                    volume_ids=(
                        validation_volume_ids
                    ),
                    class_index=class_index,
                    maximum_rows=(
                        maximum_validation_samples
                    ),
                    rng=split_rng,
                )
            )

            print(
                f"\n{class_name}: "
                f"{training_pixels.shape[0]:,} "
                f"training samples, "
                f"{validation_pixels.shape[0]:,} "
                f"validation samples"
            )

            if training_pixels.shape[0] == 0:
                print(
                    f"Skipping {class_name}: "
                    "no training samples."
                )
                continue

            if validation_pixels.shape[0] == 0:
                print(
                    f"Skipping {class_name}: "
                    "no validation samples."
                )
                continue

            class_results = (
                evaluate_gmm_candidates(
                    training_pixels=training_pixels,
                    validation_pixels=(
                        validation_pixels
                    ),
                    class_name=class_name,
                    class_index=class_index,
                    repeat_index=repeat_index,
                )
            )

            all_results.extend(class_results)

    save_results(
        results=all_results,
        results_path=RESULTS_PATH,
    )

    summaries = summarize_results(all_results)

    recommendations = (
        recommend_component_counts(summaries)
    )

    print(
        "\n"
        + "=" * 70
        + "\nRECOMMENDED GMM COMPONENT COUNTS"
        + "\n"
        + "=" * 70
    )

    for class_name, result in (
        recommendations.items()
    ):
        print(
            f"{class_name:8s}: "
            f"K={result['num_components']} | "
            f"mean BIC={result['mean_bic']:.2f} | "
            f"validation LL="
            f"{result['mean_validation_log_likelihood']:.6f} | "
            f"convergence="
            f"{result['convergence_rate']:.2f} | "
            f"mean minimum weight="
            f"{result['mean_minimum_component_weight']:.6f}"
        )


if __name__ == "__main__":
    main()