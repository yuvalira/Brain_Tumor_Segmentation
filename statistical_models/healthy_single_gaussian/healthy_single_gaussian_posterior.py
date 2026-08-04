import numpy as np
from scipy.stats import multivariate_normal


def healthy_single_gaussian_stat_inference(image, brain_mask):
    data = np.load("Brain_Tumor_Segmentation/statistical_models/healthy_single_gaussian/healthy_single_gaussian_parameters.npz")
    
    # Extract scalars and 1D/2D arrays
    prior = float(data["healthy_prior"])         # Scalar prior
    mean = data["healthy_mean"]                 # Shape: (4,)
    covariance = data["healthy_covariance"]     # Shape: (4, 4)

    height, width, channels = image.shape

    # Output container initialized to zero (2D array instead of 3D)
    unnormalized_posterior = np.zeros((height, width))

    # Extract pixel features inside brain mask -> Shape: (N_mask_pixels, 4)
    brain_pixels = image[brain_mask > 0]  

    if brain_pixels.size == 0:
        return unnormalized_posterior

    # Compute PDF for the single healthy class and scale by prior
    likelihood = multivariate_normal.pdf(brain_pixels, mean=mean, cov=covariance)
    unnormalized_posterior[brain_mask > 0] = likelihood * prior

    return unnormalized_posterior