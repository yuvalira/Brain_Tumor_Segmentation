from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


PARAMETERS_PATH = Path(__file__).resolve().parent / "healthy_gmm_parameters.npz"


def gmm_logpdf(pixels, weights, means, covariances):
    """Calculates log p(x) for a Gaussian mixture."""
    component_scores = []

    for weight, mean, covariance in zip(weights, means, covariances):
        component_scores.append(
            np.log(max(weight, 1e-300))
            + multivariate_normal.logpdf(pixels, mean=mean, cov=covariance)
        )

    return logsumexp(np.column_stack(component_scores), axis=1)


def healthy_gmm_stat_inference(image, brain_mask, parameters_path=PARAMETERS_PATH):
    """Calculates p(x | Healthy) * P(Healthy) for every brain pixel."""
    brain_mask = np.asarray(brain_mask, dtype=bool)
    output = np.zeros(image.shape[:2], dtype=np.float64)
    brain_pixels = image[brain_mask]

    if brain_pixels.size == 0:
        return output

    with np.load(parameters_path) as parameters:
        prior = float(parameters["healthy_prior"])
        weights = parameters["weights"]
        means = parameters["means"]
        covariances = parameters["covariances"]

    log_likelihood = gmm_logpdf(brain_pixels, weights, means, covariances)
    scores = np.exp(log_likelihood + np.log(prior))
    output[brain_mask] = np.nan_to_num(scores)

    return output