import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.utils import load_and_normalize_slice


# Fitting configuration
MAX_TRAINING_VOLUME = 300
TOTAL_SLICES = 155
NUM_COMPONENTS = 4
SAMPLES_PER_SLICE = 200
SAMPLES_PER_VOLUME = 1_500
MAX_FIT_SAMPLES_PER_CLASS = 100_000
N_INITIALIZATIONS = 5
MAX_ITERATIONS = 300
REG_COVARIANCE = 1e-5
RANDOM_SEED = 42

CLASS_NAMES = np.array(["NCR_NET", "ED", "ET"])
MODALITIES = np.array(["T1", "T1ce", "T2", "FLAIR"])
OUTPUT_PATH = Path(__file__).resolve().parent / "tumor_gmm_parameters.npz"


def limit_samples(data, max_samples, rng):
    """Randomly limits the number of rows without replacement."""
    if len(data) <= max_samples:
        return data

    indices = rng.choice(len(data), size=max_samples, replace=False)
    return data[indices]


def estimate_tumor_gmm_parameters():
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_pixels = [[] for _ in CLASS_NAMES]
    class_counts = np.zeros(len(CLASS_NAMES), dtype=np.uint64)
    total_brain_voxels = np.uint64(0)

    print(f"Collecting tumor samples from {MAX_TRAINING_VOLUME} volumes...")

    for volume_num in range(1, MAX_TRAINING_VOLUME + 1):
        volume_rng = np.random.default_rng(RANDOM_SEED + volume_num)
        volume_samples = [[] for _ in CLASS_NAMES]
        volume_counts = np.zeros(len(CLASS_NAMES), dtype=np.uint64)

        for slice_num in range(TOTAL_SLICES):
            image, brain_mask, tumor_mask = load_and_normalize_slice(volume_num, slice_num)

            if not np.any(brain_mask):
                continue

            brain_mask = brain_mask.astype(bool)
            total_brain_voxels += np.count_nonzero(brain_mask)

            for class_index in range(len(CLASS_NAMES)):
                class_mask = brain_mask & (tumor_mask[:, :, class_index] > 0)
                volume_counts[class_index] += np.count_nonzero(class_mask)

                pixels = image[class_mask]
                pixels = pixels[np.all(np.isfinite(pixels), axis=1)]

                if len(pixels) > 0:
                    pixels = limit_samples(pixels, SAMPLES_PER_SLICE, volume_rng)
                    volume_samples[class_index].append(pixels)

        class_counts += volume_counts
        retained_counts = []

        for class_index in range(len(CLASS_NAMES)):
            if volume_samples[class_index]:
                pixels = np.vstack(volume_samples[class_index])
                pixels = limit_samples(pixels, SAMPLES_PER_VOLUME, volume_rng)
                sampled_pixels[class_index].append(pixels)
                retained_counts.append(len(pixels))
            else:
                retained_counts.append(0)

        print(
            f"Volume {volume_num}/{MAX_TRAINING_VOLUME} | "
            f"voxels={dict(zip(CLASS_NAMES, volume_counts))} | "
            f"retained={dict(zip(CLASS_NAMES, retained_counts))}"
        )

    if total_brain_voxels == 0:
        raise RuntimeError("No brain voxels were loaded.")

    priors = class_counts.astype(np.float64) / float(total_brain_voxels)
    fit_samples = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        if not sampled_pixels[class_index]:
            raise RuntimeError(f"No samples were collected for {class_name}.")

        pixels = np.vstack(sampled_pixels[class_index])
        pixels = limit_samples(pixels, MAX_FIT_SAMPLES_PER_CLASS, rng)

        if len(pixels) < NUM_COMPONENTS:
            raise RuntimeError(f"Not enough samples to fit {class_name}.")

        fit_samples.append(pixels)

    weights = np.zeros((len(CLASS_NAMES), NUM_COMPONENTS))
    means = np.zeros((len(CLASS_NAMES), NUM_COMPONENTS, len(MODALITIES)))
    covariances = np.zeros(
        (len(CLASS_NAMES), NUM_COMPONENTS, len(MODALITIES), len(MODALITIES))
    )
    converged = np.zeros(len(CLASS_NAMES), dtype=bool)
    iterations = np.zeros(len(CLASS_NAMES), dtype=int)
    lower_bounds = np.zeros(len(CLASS_NAMES))

    for class_index, class_name in enumerate(CLASS_NAMES):
        print(
            f"\nFitting {class_name} GMM with {NUM_COMPONENTS} components "
            f"using {len(fit_samples[class_index]):,} samples..."
        )

        gmm = GaussianMixture(
            n_components=NUM_COMPONENTS,
            covariance_type="full",
            n_init=N_INITIALIZATIONS,
            max_iter=MAX_ITERATIONS,
            reg_covar=REG_COVARIANCE,
            random_state=RANDOM_SEED + class_index,
        )

        gmm.fit(fit_samples[class_index])

        if not gmm.converged_:
            raise RuntimeError(f"{class_name} GMM did not converge.")

        weights[class_index] = gmm.weights_
        means[class_index] = gmm.means_
        covariances[class_index] = gmm.covariances_
        converged[class_index] = gmm.converged_
        iterations[class_index] = gmm.n_iter_
        lower_bounds[class_index] = gmm.lower_bound_

        print(f"Prior: {priors[class_index]:.8f}")
        print(f"Weights: {gmm.weights_}")
        print(f"Converged in {gmm.n_iter_} iterations")

    np.savez(
        OUTPUT_PATH,
        priors=priors,
        weights=weights,
        means=means,
        covariances=covariances,
        num_components=np.full(len(CLASS_NAMES), NUM_COMPONENTS),
        counts=class_counts,
        total_brain_voxels=total_brain_voxels,
        fit_sample_counts=np.array([len(samples) for samples in fit_samples]),
        converged=converged,
        num_iterations=iterations,
        lower_bounds=lower_bounds,
        class_names=CLASS_NAMES,
        modalities=MODALITIES,
    )

    print(f"\nParameters saved to: {OUTPUT_PATH}")
    print("Tumor priors:", dict(zip(CLASS_NAMES, priors)))

    return priors, weights, means, covariances


if __name__ == "__main__":
    estimate_tumor_gmm_parameters()