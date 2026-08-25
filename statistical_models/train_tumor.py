import os
import numpy as np
from sklearn.mixture import GaussianMixture
from config import PROJECT_ROOT, MAX_TRAINING_VOLUME, SLICE_NUM, RANDOM_SEED
from utilities.utils import load_and_normalize_slice


def fit_and_save_tumor_gmm(
        num_components,
        filename,
        channel_indices,
        num_classes,
        training_volumes=None,
):
    """
    Fits a Gaussian Mixture Model independently for each tumor tissue class across
    an arbitrary subset of feature channels and saves the learned parameters.

    :param num_components: Number of mixture components (K) per class.
    :param filename: Target output filename (e.g., 'tumor_gmm_raw.npz').
    :param channel_indices: Channel indices to slice from the 9D feature tensor.
    :param num_classes: Number of distinct tumor classes in the mask (default: 3).
    """
    training_volumes = np.asarray(
        list(training_volumes) if training_volumes is not None
        else range(1, MAX_TRAINING_VOLUME + 1),
        dtype=int,
    )
    output_dir = os.path.join(PROJECT_ROOT, "saved_parameters", "statistical_models")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    num_features = len(channel_indices)
    class_pixels = [[] for _ in range(num_classes)]
    total_brain_pixels = 0

    print(
        f"Collecting tumor pixels from {len(training_volumes)} training volumes | "
        f"Channels: {channel_indices} | K={num_components}..."
    )

    for vol_num in training_volumes:
        image, brain_mask, tumor_masks, _ = load_and_normalize_slice(vol_num, SLICE_NUM)

        # Select target feature channels
        features_image = image[:, :, channel_indices]
        total_brain_pixels += int(np.sum(brain_mask))

        for class_index in range(num_classes):
            # Extract ground truth mask for this specific tumor sub-region
            if tumor_masks.ndim == 3:
                class_mask = (tumor_masks[:, :, class_index] > 0) & brain_mask
            else:
                class_mask = (tumor_masks == (class_index + 1)) & brain_mask

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
        priors[class_index] = len(X_class) / total_brain_pixels if total_brain_pixels > 0 else 0.0

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
        print(f"  Class {class_index} converged: {gmm.converged_}, iters: {gmm.n_iter_}")

    np.savez(
        output_path,
        priors=priors,
        weights=weights,
        means=means,
        covariances=covariances,
        pixel_counts=pixel_counts,
        total_brain_pixels=total_brain_pixels,
        channel_indices=np.array(channel_indices),
        training_volumes=training_volumes,
    )
    print(f"Tumor GMM parameters successfully saved to '{output_path}'\n")
