import os
import numpy as np
from scipy.stats import multivariate_normal
from config import PROJECT_ROOT, SLICE_NUM
from utilities.utils import load_and_normalize_slice


def healthy_gmm_joint_likelihood(
    vol_num: int,
    filename: str,
    slice_num: int = SLICE_NUM,
):
    """
    Computes the joint healthy GMM likelihood map P(x, Healthy) for a slice.
    Reads channel indices directly from the saved GMM .npz file.

    :param vol_num: Training/evaluation volume index.
    :param filename: Saved GMM model filename (e.g., 'healthy_gmm_raw.npz').
    :param slice_num: Axial slice index.
    :return: 2D array of joint healthy likelihood values across the slice grid (H, W).
    """
    models_dir = os.path.join(PROJECT_ROOT, "saved_parameters", "statistical_models")
    gmm_path = os.path.join(models_dir, filename)

    # 1. Load Pre-trained GMM Parameters
    gmm = np.load(gmm_path)
    weights = gmm["weights"]          # Shape: (K,)
    means = gmm["means"]              # Shape: (K, D)
    covariances = gmm["covariances"]  # Shape: (K, D, D)
    prior_healthy = float(gmm["prior"])

    # Extract channel indices stored during training
    if "channel_indices" in gmm:
        channel_indices = gmm["channel_indices"]
    else:
        channel_indices = list(range(means.shape[1]))

    num_components = len(weights)

    # 2. Load & Slice Multimodal Features
    image, brain_mask, _, _ = load_and_normalize_slice(vol_num, slice_num)
    features_image = image[:, :, channel_indices]  # Shape: (H, W, D)
    H, W, D = features_image.shape

    # 3. Evaluate Multivariate Gaussian Densities
    X_flat = features_image.reshape(-1, D).astype(np.float64)
    joint_likelihoods = np.zeros((H * W, num_components), dtype=np.float64)

    for k in range(num_components):
        pdf_k = multivariate_normal.pdf(
            X_flat,
            mean=means[k],
            cov=covariances[k],
            allow_singular=True,
        )
        joint_likelihoods[:, k] = weights[k] * pdf_k

    # 4. Spatial Masking & Joint Prior Multiplication
    joint_likelihoods = joint_likelihoods.reshape(H, W, num_components)
    joint_likelihoods *= brain_mask[:, :, np.newaxis]

    # Sum over all mixture components and scale by dataset prior P(Healthy)
    healthy_joint_likelihood_map = prior_healthy * np.sum(joint_likelihoods, axis=-1)

    return healthy_joint_likelihood_map