import os
import numpy as np
from scipy.stats import multivariate_normal
from config import *
from utilities.utils import load_and_normalize_slice


def healthy_gmm_joint_likelihood(vol_num: int, lambda_val: float = LAMBDA, symmetric: bool = False):
    """
    Computes the joint healthy GMM likelihood map P(x, Healthy) for a slice.

    :param vol_num: Training/evaluation volume index.
    :param lambda_val: Linear interpolation weight between global prior and local spatial weights.
    :param symmetric: If True, uses 8D features (4 raw + 4 symmetry/NDI channels).
                      If False, uses 4D base raw modality features.
    :return: 2D array of joint healthy likelihood values across the slice grid (H, W).
    """
    # 1. Determine filenames and feature dimensions based on mode
    gmm_filename = 'healthy_gmm_symmetric.npz' if symmetric else 'healthy_gmm.npz'
    weights_filename = 'local_weights_symmetric.npz' if symmetric else 'local_weights.npz'

    models_dir = os.path.join(PROJECT_ROOT, 'saved_parameters', 'statistical_models')
    gmm_path = os.path.join(models_dir, gmm_filename)
    local_weights_path = os.path.join(models_dir, weights_filename)

    # Load Pre-trained GMM Parameters & Spatial Weights Map
    gmm = np.load(gmm_path)
    global_weights = gmm['weights']  # Shape: (K,)
    means = gmm['means']  # Shape: (K, 4) or (K, 8)
    covariances = gmm['covariances']  # Shape: (K, 4, 4) or (K, 8, 8)
    prior_healthy = gmm['prior']  # Dataset prior P(Healthy)

    num_components = len(global_weights)

    # 2. Select global baseline weights or interpolate spatial weights.
    global_weights_3d = global_weights[np.newaxis, np.newaxis, :]
    if lambda_val == 0:
        weights = np.broadcast_to(
            global_weights_3d,
            (IMAGE_PIXEL_LENGTH, IMAGE_PIXEL_LENGTH, num_components),
        )
    else:
        local_weights = np.load(local_weights_path)['weights']
        weights = (1.0 - lambda_val) * global_weights_3d + lambda_val * local_weights

    # 3. Load Target Slice Features
    slice_output = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=symmetric)
    features_image = slice_output[0]  # Shape: (H, W, 4) or (H, W, 8)
    brain_mask = slice_output[1]

    H, W, C = features_image.shape

    # Flatten spatial dimensions for vectorized PDF calculation: shape (H*W, C)
    X_flat = features_image.reshape(-1, C).astype(np.float64)

    # Un-normalized joint likelihood buffer: shape (H*W, K)
    joint_likelihoods = np.zeros((H * W, num_components), dtype=np.float64)

    # 4. Evaluate Multivariate Gaussian Densities
    for k in range(num_components):
        pdf_k = multivariate_normal.pdf(
            X_flat,
            mean=means[k],
            cov=covariances[k],
            allow_singular=True
        )

        w_k_flat = weights[:, :, k].reshape(-1)

        # Component joint likelihood: w_k(x, y) * N(x | mu_k, Sigma_k)
        joint_likelihoods[:, k] = w_k_flat * pdf_k

    # Reshape back to 3D spatial grid: shape (H, W, K)
    joint_likelihoods = joint_likelihoods.reshape(H, W, num_components)

    # Zero out background non-brain pixels via broadcasting
    joint_likelihoods *= brain_mask[:, :, np.newaxis]

    # Sum across GMM components and multiply by global prior
    healthy_joint_likelihood_map = np.sum(prior_healthy * joint_likelihoods, axis=-1)

    return healthy_joint_likelihood_map
