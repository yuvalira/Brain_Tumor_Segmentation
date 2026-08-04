import numpy as np
from scipy.stats import multivariate_normal

def tumor_single_gaussian_stat_inference(image, brain_mask):
    data = np.load("Brain_Tumor_Segmentation/statistical_models/tumor_single_gaussian/tumor_single_gaussian_parameters.npz")
    priors, means, covariances = data["priors"], data["means"], data["covariances"]

    num_classes = priors.shape[0]
    height, width, _ = image.shape

    # Output shape: (Height, Width, Classes)
    unnormalized_posteriors = np.zeros((height, width, num_classes))

    brain_pixels = image[brain_mask > 0]

    if brain_pixels.size == 0:
        return unnormalized_posteriors

    for i in range(num_classes):
        likelihood = multivariate_normal.pdf(brain_pixels, mean=means[i], cov=covariances[i])
        # Index spatial mask first, then the class index along the last axis
        unnormalized_posteriors[brain_mask > 0, i] = likelihood * priors[i]

    return unnormalized_posteriors