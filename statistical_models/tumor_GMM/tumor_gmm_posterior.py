from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


PARAMETERS_PATH = Path(__file__).resolve().parent / "tumor_gmm_parameters.npz"


def gmm_logpdf(pixels, weights, means, covariances):
    """Calculates log p(x) for a Gaussian mixture."""
    component_scores = []

    for weight, mean, covariance in zip(weights, means, covariances):
        component_scores.append(
            np.log(max(weight, 1e-300))
            + multivariate_normal.logpdf(pixels, mean=mean, cov=covariance)
        )

    return logsumexp(np.column_stack(component_scores), axis=1)


def tumor_gmm_stat_inference(image, brain_mask, parameters_path=PARAMETERS_PATH):
    """
    Calculates p(x | C_k) * P(C_k) for NCR/NET, ED and ET.

    Returns an array with shape (H, W, 3).
    """
    brain_mask = np.asarray(brain_mask, dtype=bool)
    brain_pixels = image[brain_mask]

    with np.load(parameters_path) as parameters:
        priors = parameters["priors"]
        weights = parameters["weights"]
        means = parameters["means"]
        covariances = parameters["covariances"]

    num_classes = len(priors)
    output = np.zeros((*image.shape[:2], num_classes), dtype=np.float64)

    if brain_pixels.size == 0:
        return output

    for class_index in range(num_classes):
        log_likelihood = gmm_logpdf(
            brain_pixels,
            weights[class_index],
            means[class_index],
            covariances[class_index],
        )

        scores = np.exp(log_likelihood + np.log(priors[class_index]))
        output[brain_mask, class_index] = np.nan_to_num(scores)

    return output