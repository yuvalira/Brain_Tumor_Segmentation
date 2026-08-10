import os

import numpy as np
from sklearn.mixture import GaussianMixture

from config import *
from utilities.utils import load_and_normalize_slice


def fit_and_save_tumor_gmm(symmetric: bool = False):
    """Fit a four-component GMM independently for each tumor tissue."""
    num_classes = 3
    num_components = 4
    num_features = 8 if symmetric else 4
    filename = "tumor_gmm_symmetric.npz" if symmetric else "tumor_gmm.npz"
    output_dir = os.path.join(PROJECT_ROOT, "saved_parameters", "statistical_models")
    os.makedirs(output_dir, exist_ok=True)

    class_pixels = [[] for _ in range(num_classes)]
    total_brain_pixels = 0
    print(
        f"Collecting tumor pixels from volumes 1-{MAX_TRAINING_VOLUME} "
        f"(symmetric={symmetric})..."
    )

    for vol_num in range(1, MAX_TRAINING_VOLUME + 1):
        slice_output = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=symmetric)
        features_image, brain_mask, tumor_masks = slice_output[:3]
        total_brain_pixels += int(np.sum(brain_mask))

        for class_index in range(num_classes):
            class_mask = (tumor_masks[:, :, class_index] > 0) & brain_mask
            if np.any(class_mask):
                class_pixels[class_index].append(
                    np.asarray(features_image[class_mask], dtype=np.float64)
                )

    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    priors = np.zeros(num_classes, dtype=np.float64)
    weights = np.zeros((num_classes, num_components), dtype=np.float64)
    means = np.zeros((num_classes, num_components, num_features), dtype=np.float64)
    covariances = np.zeros(
        (num_classes, num_components, num_features, num_features),
        dtype=np.float64,
    )

    for class_index in range(num_classes):
        if not class_pixels[class_index]:
            raise ValueError(f"No training pixels found for tumor class {class_index}.")

        X_class = np.concatenate(class_pixels[class_index], axis=0)
        pixel_counts[class_index] = len(X_class)
        priors[class_index] = len(X_class) / total_brain_pixels
        if len(X_class) < num_components:
            raise ValueError(
                f"Tumor class {class_index} has only {len(X_class)} pixels; "
                f"at least {num_components} are required."
            )

        print(
            f"Fitting class {class_index}: {len(X_class):,} pixels, "
            f"prior={priors[class_index]:.6f}"
        )
        gmm = GaussianMixture(
            n_components=num_components,
            covariance_type="full",
            reg_covar=1e-5,
            max_iter=200,
            n_init=5,
            random_state=RANDOM_SEED,
        )
        gmm.fit(X_class)
        weights[class_index] = gmm.weights_
        means[class_index] = gmm.means_
        covariances[class_index] = gmm.covariances_
        print(f"  converged={gmm.converged_}, iterations={gmm.n_iter_}")

    output_path = os.path.join(output_dir, filename)
    np.savez(
        output_path,
        priors=priors,
        weights=weights,
        means=means,
        covariances=covariances,
        pixel_counts=pixel_counts,
        total_brain_pixels=total_brain_pixels,
    )
    print(f"Tumor GMM parameters saved to '{output_path}'")


if __name__ == "__main__":
    fit_and_save_tumor_gmm(symmetric=False)
    fit_and_save_tumor_gmm(symmetric=True)
