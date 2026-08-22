import os
import numpy as np
from scipy.stats import multivariate_normal
from config import PROJECT_ROOT, SLICE_NUM
from utilities.utils import load_and_normalize_slice


def tumor_gmm_joint_likelihood(
    vol_num: int,
    filename: str,
    slice_num: int = SLICE_NUM,
):
    """
    Computes the joint tumor GMM likelihood map P(x, Tumor_c) for each tumor class c in a slice.
    Reads channel indices and parameters directly from the saved tumor GMM .npz file.

    :param vol_num: Training/evaluation volume index.
    :param filename: Saved tumor GMM filename (e.g., 'tumor_gmm_raw.npz').
    :param slice_num: Axial slice index.
    :return: 3D array of shape (H, W, num_classes) representing joint likelihoods per tumor class,
             or 2D array of the total tumor joint likelihood sum across classes.
    """
    models_dir = os.path.join(PROJECT_ROOT, "saved_parameters", "statistical_models")
    gmm_path = os.path.join(models_dir, filename)

    # 1. Load Pre-trained Tumor GMM Parameters
    gmm = np.load(gmm_path)
    priors = gmm["priors"]            # Shape: (num_classes,)
    weights = gmm["weights"]          # Shape: (num_classes, K)
    means = gmm["means"]              # Shape: (num_classes, K, D)
    covariances = gmm["covariances"]  # Shape: (num_classes, K, D, D)

    # Extract channel indices stored during training
    if "channel_indices" in gmm:
        channel_indices = gmm["channel_indices"]
    else:
        channel_indices = list(range(means.shape[-1]))

    num_classes, num_components, D = means.shape

    # 2. Load & Slice Multimodal Features
    image, brain_mask, _, _ = load_and_normalize_slice(vol_num, slice_num)
    features_image = image[:, :, channel_indices]  # Shape: (H, W, D)
    H, W, _ = features_image.shape

    # 3. Evaluate Multivariate Gaussian Densities per Class
    X_flat = features_image.reshape(-1, D).astype(np.float64)
    tumor_joint_likelihood_map = np.zeros((H * W, num_classes), dtype=np.float64)

    for c in range(num_classes):
        class_pdf_sum = np.zeros(H * W, dtype=np.float64)
        for k in range(num_components):
            pdf_k = multivariate_normal.pdf(
                X_flat,
                mean=means[c, k],
                cov=covariances[c, k],
                allow_singular=True,
            )
            class_pdf_sum += weights[c, k] * pdf_k

        # Multiply by class prior: P(Tumor_c) * p(x | Tumor_c)
        tumor_joint_likelihood_map[:, c] = priors[c] * class_pdf_sum

    # 4. Spatial Masking & Reshaping
    tumor_joint_likelihood_map = tumor_joint_likelihood_map.reshape(H, W, num_classes)
    tumor_joint_likelihood_map *= brain_mask[:, :, np.newaxis]

    return tumor_joint_likelihood_map