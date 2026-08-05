from pathlib import Path

import numpy as np
from scipy.stats import multivariate_t, t


PARAMETERS_PATH = (
    Path(__file__).resolve().parent
    / "healthy_single_skew_t_parameters.npz"
)


def multivariate_skew_t_logpdf(
    pixels,
    location,
    dispersion,
    skewness,
    degrees_of_freedom,
):
    """
    Computes the log-PDF of the multivariate skew-t distribution.

    Parameters
    ----------
    pixels : np.ndarray
        Array of shape (N, D). Each row is one multimodal MRI pixel.

    location : np.ndarray
        Skew-t location vector xi, shape (D,).

    dispersion : np.ndarray
        Skew-t dispersion matrix Omega, shape (D, D).

    skewness : np.ndarray
        Skewness/slant vector alpha, shape (D,).

    degrees_of_freedom : float
        Degrees of freedom nu.

    Returns
    -------
    log_density : np.ndarray
        Log skew-t density for every pixel, shape (N,).
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    location = np.asarray(location, dtype=np.float64).reshape(-1)
    dispersion = np.asarray(dispersion, dtype=np.float64)
    skewness = np.asarray(skewness, dtype=np.float64).reshape(-1)
    degrees_of_freedom = float(degrees_of_freedom)

    if pixels.ndim != 2:
        raise ValueError(
            f"Expected pixels with shape (N, D), received {pixels.shape}."
        )

    num_dimensions = pixels.shape[1]

    if location.shape != (num_dimensions,):
        raise ValueError(
            f"Location must have shape ({num_dimensions},), "
            f"received {location.shape}."
        )

    if skewness.shape != (num_dimensions,):
        raise ValueError(
            f"Skewness must have shape ({num_dimensions},), "
            f"received {skewness.shape}."
        )

    if dispersion.shape != (num_dimensions, num_dimensions):
        raise ValueError(
            "Dispersion must have shape "
            f"({num_dimensions}, {num_dimensions}), "
            f"received {dispersion.shape}."
        )

    if degrees_of_freedom <= 0:
        raise ValueError(
            "Degrees of freedom must be positive."
        )

    if not np.all(np.isfinite(pixels)):
        raise ValueError(
            "Input pixels contain NaN or infinite values."
        )

    if not np.all(np.isfinite(location)):
        raise ValueError(
            "Location contains NaN or infinite values."
        )

    if not np.all(np.isfinite(dispersion)):
        raise ValueError(
            "Dispersion contains NaN or infinite values."
        )

    if not np.all(np.isfinite(skewness)):
        raise ValueError(
            "Skewness contains NaN or infinite values."
        )

    # Numerical symmetry in case the saved matrix contains very small
    # floating-point asymmetries.
    dispersion = 0.5 * (dispersion + dispersion.T)

    diagonal = np.diag(dispersion)

    if np.any(diagonal <= 0):
        raise ValueError(
            "The diagonal of the dispersion matrix must be positive."
        )

    # Center every pixel around the fitted location xi.
    centered_pixels = pixels - location

    # Mahalanobis quantity:
    #
    # Q = (x - xi)^T Omega^(-1) (x - xi)
    #
    # np.linalg.solve is more stable than explicitly calculating
    # the inverse of Omega.
    solved_pixels = np.linalg.solve(
        dispersion,
        centered_pixels.T,
    ).T

    mahalanobis_squared = np.einsum(
        "ij,ij->i",
        centered_pixels,
        solved_pixels,
    )

    # Protect against extremely small negative values caused by
    # floating-point precision.
    mahalanobis_squared = np.maximum(
        mahalanobis_squared,
        0.0,
    )

    # Standardize using the marginal scales of Omega:
    #
    # z = omega^(-1) (x - xi)
    #
    marginal_scales = np.sqrt(diagonal)
    standardized_pixels = centered_pixels / marginal_scales

    # alpha^T z for every pixel.
    skew_projection = standardized_pixels @ skewness

    # Argument of the univariate Student-t CDF.
    skew_argument = (
        skew_projection
        * np.sqrt(
            (degrees_of_freedom + num_dimensions)
            / (
                degrees_of_freedom
                + mahalanobis_squared
            )
        )
    )

    # Symmetric multivariate Student-t component.
    log_symmetric_t = multivariate_t.logpdf(
        pixels,
        loc=location,
        shape=dispersion,
        df=degrees_of_freedom,
    )

    # Skewness correction:
    #
    # T_(nu + D)(skew_argument)
    #
    log_skew_correction = t.logcdf(
        skew_argument,
        df=degrees_of_freedom + num_dimensions,
    )

    # Complete multivariate skew-t density:
    #
    # f_ST(x) = 2 * t_D(x) * T_(nu+D)(...)
    #
    log_density = (
        np.log(2.0)
        + log_symmetric_t
        + log_skew_correction
    )

    return log_density


def healthy_single_skew_t_stat_inference(
    image,
    brain_mask,
    parameters_path=PARAMETERS_PATH,
):
    """
    Computes the unnormalized healthy skew-t posterior:

        p(x | Healthy) * P(Healthy)

    for every pixel inside the brain mask.

    Parameters
    ----------
    image : np.ndarray
        Normalized multimodal MRI slice, shape (H, W, 4).

    brain_mask : np.ndarray
        Boolean brain mask, shape (H, W).

    parameters_path : str or pathlib.Path
        Path to healthy_single_skew_t_parameters.npz.

    Returns
    -------
    unnormalized_posterior : np.ndarray
        Unnormalized healthy posterior map, shape (H, W).
        Pixels outside the brain mask are zero.
    """
    image = np.asarray(image, dtype=np.float64)
    brain_mask = np.asarray(brain_mask, dtype=bool)

    if image.ndim != 3:
        raise ValueError(
            "Image must have shape (H, W, D)."
        )

    if brain_mask.shape != image.shape[:2]:
        raise ValueError(
            "Brain mask shape must match the first two "
            "dimensions of the image."
        )

    parameters_path = Path(parameters_path)

    if not parameters_path.exists():
        raise FileNotFoundError(
            f"Skew-t parameter file was not found: "
            f"{parameters_path}"
        )

    with np.load(parameters_path) as parameters:
        healthy_prior = float(
            parameters["healthy_prior"]
        )

        location = parameters["location"]
        dispersion = parameters["dispersion"]
        skewness = parameters["skewness"]

        degrees_of_freedom = float(
            parameters["degrees_of_freedom"]
        )

    if not np.isfinite(healthy_prior):
        raise ValueError(
            "Healthy prior is not finite."
        )

    if healthy_prior <= 0:
        raise ValueError(
            "Healthy prior must be positive."
        )

    height, width, _ = image.shape

    unnormalized_posterior = np.zeros(
        (height, width),
        dtype=np.float64,
    )

    # Extract only brain pixels.
    brain_pixels = image[brain_mask]

    if brain_pixels.size == 0:
        return unnormalized_posterior

    # Calculate log p(x | Healthy).
    log_likelihood = multivariate_skew_t_logpdf(
        pixels=brain_pixels,
        location=location,
        dispersion=dispersion,
        skewness=skewness,
        degrees_of_freedom=degrees_of_freedom,
    )

    # Calculate:
    #
    # log[p(x | Healthy) P(Healthy)]
    #
    log_unnormalized_posterior = (
        log_likelihood
        + np.log(healthy_prior)
    )

    # Return to the normal probability domain so this output matches
    # the existing Gaussian posterior functions and main.py.
    brain_scores = np.exp(
        log_unnormalized_posterior
    )

    # Convert possible numerical NaN values to zero.
    brain_scores = np.nan_to_num(
        brain_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    unnormalized_posterior[brain_mask] = brain_scores

    return unnormalized_posterior