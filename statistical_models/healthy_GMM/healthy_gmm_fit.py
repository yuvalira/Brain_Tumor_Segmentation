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
NUM_COMPONENTS = 9
SAMPLES_PER_SLICE = 200
SAMPLES_PER_VOLUME = 3_000
MAX_FIT_SAMPLES = 300_000
N_INITIALIZATIONS = 5
MAX_ITERATIONS = 300
REG_COVARIANCE = 1e-5
RANDOM_SEED = 42

MODALITIES = np.array(["T1", "T1ce", "T2", "FLAIR"])
OUTPUT_PATH = Path(__file__).resolve().parent / "healthy_gmm_parameters.npz"


def limit_samples(data, max_samples, rng):
    """Randomly limits the number of rows without replacement."""
    if len(data) <= max_samples:
        return data

    indices = rng.choice(len(data), size=max_samples, replace=False)
    return data[indices]


def estimate_healthy_gmm_parameters():
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_pixels = []
    total_brain_voxels = 0
    total_healthy_voxels = 0

    print(f"Collecting healthy samples from {MAX_TRAINING_VOLUME} volumes...")

    for volume_num in range(1, MAX_TRAINING_VOLUME + 1):
        volume_rng = np.random.default_rng(RANDOM_SEED + volume_num)
        volume_samples = []
        volume_brain_count = 0
        volume_healthy_count = 0

        for slice_num in range(TOTAL_SLICES):
            image, brain_mask, tumor_mask = load_and_normalize_slice(volume_num, slice_num)

            if not np.any(brain_mask):
                continue

            brain_mask = brain_mask.astype(bool)
            healthy_mask = brain_mask & ~np.any(tumor_mask > 0, axis=-1)

            volume_brain_count += np.count_nonzero(brain_mask)
            volume_healthy_count += np.count_nonzero(healthy_mask)

            pixels = image[healthy_mask]
            pixels = pixels[np.all(np.isfinite(pixels), axis=1)]

            if len(pixels) > 0:
                pixels = limit_samples(pixels, SAMPLES_PER_SLICE, volume_rng)
                volume_samples.append(pixels)

        total_brain_voxels += volume_brain_count
        total_healthy_voxels += volume_healthy_count

        if volume_samples:
            volume_samples = np.vstack(volume_samples)
            volume_samples = limit_samples(volume_samples, SAMPLES_PER_VOLUME, volume_rng)
            sampled_pixels.append(volume_samples)
            retained = len(volume_samples)
        else:
            retained = 0

        print(
            f"Volume {volume_num}/{MAX_TRAINING_VOLUME} | "
            f"healthy={volume_healthy_count:,} | retained={retained:,}"
        )

    if total_brain_voxels == 0 or not sampled_pixels:
        raise RuntimeError("No healthy brain pixels were loaded.")

    healthy_prior = total_healthy_voxels / total_brain_voxels
    sampled_pixels = np.vstack(sampled_pixels)
    sampled_pixels = limit_samples(sampled_pixels, MAX_FIT_SAMPLES, rng)

    print(f"\nFitting {NUM_COMPONENTS}-component healthy GMM...")
    print(f"Fit samples: {len(sampled_pixels):,}")
    print(f"Healthy prior: {healthy_prior:.8f}")

    gmm = GaussianMixture(
        n_components=NUM_COMPONENTS,
        covariance_type="full",
        n_init=N_INITIALIZATIONS,
        max_iter=MAX_ITERATIONS,
        reg_covar=REG_COVARIANCE,
        random_state=RANDOM_SEED,
    )

    gmm.fit(sampled_pixels)

    if not gmm.converged_:
        raise RuntimeError("Healthy GMM did not converge.")

    np.savez(
        OUTPUT_PATH,
        healthy_prior=healthy_prior,
        weights=gmm.weights_,
        means=gmm.means_,
        covariances=gmm.covariances_,
        num_components=NUM_COMPONENTS,
        healthy_voxel_count=total_healthy_voxels,
        total_brain_voxels=total_brain_voxels,
        fit_sample_count=len(sampled_pixels),
        converged=gmm.converged_,
        num_iterations=gmm.n_iter_,
        lower_bound=gmm.lower_bound_,
        modalities=MODALITIES,
    )

    print(f"\nParameters saved to: {OUTPUT_PATH}")
    print(f"Converged in {gmm.n_iter_} iterations")
    print("Component weights:", gmm.weights_)

    return healthy_prior, gmm.weights_, gmm.means_, gmm.covariances_


if __name__ == "__main__":
    estimate_healthy_gmm_parameters()