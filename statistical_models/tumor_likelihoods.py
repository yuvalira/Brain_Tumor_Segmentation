import os
import numpy as np
from scipy.stats import multivariate_normal
from config import *
from utilities.utils import load_and_normalize_slice


def tumor_joint_likelihood(vol_num: int, symmetric: bool = False):
    """
    Computes joint likelihood maps P(x, Tumor_c) for each tumor class across a slice.

    :param vol_num: Training/evaluation volume index.
    :param symmetric: If True, uses 8D features (4 raw + 4 symmetry/NDI channels).
                      If False, uses 4D base raw modality features.
    :return: 3D array of joint likelihoods across the slice grid with shape (H, W, num_tumor_classes).
    """
    # 1. Load Pre-trained Multi-Class Tumor Parameters
    filename = 'tumor_gaussian_symmetric.npz' if symmetric else 'tumor_gaussian.npz'
    gaussian_path = os.path.join(PROJECT_ROOT, 'saved_parameters', 'statistical_models', filename)
    gaussian = np.load(gaussian_path)

    priors = gaussian['priors']  # Shape: (3,) - Priors for each tumor class
    means = gaussian['means']  # Shape: (3, 4) or (3, 8) - Mean vectors per class
    covariances = gaussian['covariances']  # Shape: (3, 4, 4) or (3, 8, 8) - Covariance matrices per class

    num_tumor_classes = len(priors)  # 3 Classes

    # 2. Load Target Slice Features
    slice_output = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=symmetric)
    features_image = slice_output[0]  # Shape: (H, W, 4) or (H, W, 8)
    brain_mask = slice_output[1]  # Shape: (H, W)

    H, W, C = features_image.shape

    # Flatten spatial dimensions for vectorized PDF calculation: shape (H*W, C)
    X_flat = features_image.reshape(-1, C).astype(np.float64)

    # 3. Compute Joint Likelihood for Each Tumor Class: shape (H*W, 3)
    joint_likelihoods = np.zeros((H * W, num_tumor_classes), dtype=np.float64)

    for c in range(num_tumor_classes):
        pdf_c = multivariate_normal.pdf(
            X_flat,
            mean=means[c],
            cov=covariances[c],
            allow_singular=True
        )

        # Joint Likelihood P(x, Tumor_c) = P(Tumor_c) * N(x | mu_c, Sigma_c)
        joint_likelihoods[:, c] = priors[c] * pdf_c

    # 4. Reshape back to 3D spatial grid: shape (H, W, num_tumor_classes)
    joint_likelihoods = joint_likelihoods.reshape(H, W, num_tumor_classes)

    # Zero out background non-brain pixels via broadcasting
    joint_likelihoods *= brain_mask[:, :, np.newaxis]

    return joint_likelihoods