import os

import numpy as np
from scipy.stats import multivariate_normal

from config import *
from utilities.utils import load_and_normalize_slice


def tumor_joint_likelihood(vol_num: int, symmetric: bool = False):
    """Calculate one joint likelihood map for each four-component tumor GMM."""
    filename = "tumor_gmm_symmetric.npz" if symmetric else "tumor_gmm.npz"
    parameters = np.load(
        os.path.join(PROJECT_ROOT, "saved_parameters", "statistical_models", filename)
    )
    priors = parameters["priors"]
    weights = parameters["weights"]
    means = parameters["means"]
    covariances = parameters["covariances"]

    slice_output = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=symmetric)
    features_image, brain_mask = slice_output[:2]
    height, width, num_features = features_image.shape
    X_flat = features_image.reshape(-1, num_features).astype(np.float64)
    joint_likelihoods = np.zeros((height * width, len(priors)), dtype=np.float64)

    for class_index in range(len(priors)):
        class_likelihood = np.zeros(height * width, dtype=np.float64)
        for component_index in range(weights.shape[1]):
            class_likelihood += weights[class_index, component_index] * multivariate_normal.pdf(
                X_flat,
                mean=means[class_index, component_index],
                cov=covariances[class_index, component_index],
                allow_singular=True,
            )
        joint_likelihoods[:, class_index] = priors[class_index] * class_likelihood

    joint_likelihoods = joint_likelihoods.reshape(height, width, len(priors))
    joint_likelihoods *= brain_mask[:, :, np.newaxis]
    return joint_likelihoods
