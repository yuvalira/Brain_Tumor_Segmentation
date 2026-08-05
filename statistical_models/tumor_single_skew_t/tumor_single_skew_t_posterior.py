from pathlib import Path

import numpy as np

from statistical_models.healthy_single_skew_t.healthy_single_skew_t_posterior import (
    multivariate_skew_t_logpdf,
)


PARAMETERS_PATH = (
    Path(__file__).resolve().parent
    / "tumor_single_skew_t_parameters.npz"
)


def tumor_single_skew_t_stat_inference(
    image,
    brain_mask,
    parameters_path=PARAMETERS_PATH,
):
    """
    Computes the unnormalized skew-t posterior for each tumor class:

        p(x | C_k) * P(C_k)

    where the tumor classes are:

        0: NCR/NET
        1: ED
        2: ET

    Parameters
    ----------
    image : np.ndarray
        Normalized multimodal MRI slice, shape (H, W, 4).

    brain_mask : np.ndarray
        Boolean brain mask, shape (H, W).

    parameters_path : str or pathlib.Path
        Path to tumor_single_skew_t_parameters.npz.

    Returns
    -------
    unnormalized_posteriors : np.ndarray
        Unnormalized tumor posterior scores, shape (H, W, 3).

        Channel order:
            0: NCR/NET
            1: ED
            2: ET

        Pixels outside the brain mask are zero.
    """
    image = np.asarray(
        image,
        dtype=np.float64,
    )

    brain_mask = np.asarray(
        brain_mask,
        dtype=bool,
    )

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
            f"Tumor skew-t parameter file was not found: "
            f"{parameters_path}"
        )

    # Load the parameters estimated during training.
    with np.load(parameters_path) as parameters:
        priors = np.asarray(
            parameters["priors"],
            dtype=np.float64,
        )

        locations = np.asarray(
            parameters["locations"],
            dtype=np.float64,
        )

        dispersions = np.asarray(
            parameters["dispersions"],
            dtype=np.float64,
        )

        skewness = np.asarray(
            parameters["skewness"],
            dtype=np.float64,
        )

        degrees_of_freedom = np.asarray(
            parameters["degrees_of_freedom"],
            dtype=np.float64,
        )

        # This is optional metadata, but useful for validation.
        class_names = (
            parameters["class_names"]
            if "class_names" in parameters
            else np.array(["NCR_NET", "ED", "ET"])
        )

    num_classes = priors.shape[0]
    num_modalities = image.shape[-1]

    # Validate the stored parameter dimensions.
    if locations.shape != (
        num_classes,
        num_modalities,
    ):
        raise ValueError(
            "Locations must have shape "
            f"({num_classes}, {num_modalities}), "
            f"received {locations.shape}."
        )

    if dispersions.shape != (
        num_classes,
        num_modalities,
        num_modalities,
    ):
        raise ValueError(
            "Dispersions must have shape "
            f"({num_classes}, {num_modalities}, "
            f"{num_modalities}), "
            f"received {dispersions.shape}."
        )

    if skewness.shape != (
        num_classes,
        num_modalities,
    ):
        raise ValueError(
            "Skewness must have shape "
            f"({num_classes}, {num_modalities}), "
            f"received {skewness.shape}."
        )

    if degrees_of_freedom.shape != (num_classes,):
        raise ValueError(
            "Degrees of freedom must have shape "
            f"({num_classes},), "
            f"received {degrees_of_freedom.shape}."
        )

    if not np.all(np.isfinite(priors)):
        raise ValueError(
            "Tumor priors contain NaN or infinite values."
        )

    if np.any(priors <= 0):
        raise ValueError(
            "All tumor priors must be positive."
        )

    if not np.all(np.isfinite(locations)):
        raise ValueError(
            "Tumor locations contain NaN or infinite values."
        )

    if not np.all(np.isfinite(dispersions)):
        raise ValueError(
            "Tumor dispersions contain NaN or infinite values."
        )

    if not np.all(np.isfinite(skewness)):
        raise ValueError(
            "Tumor skewness parameters contain NaN or "
            "infinite values."
        )

    if not np.all(
        np.isfinite(degrees_of_freedom)
    ):
        raise ValueError(
            "Degrees of freedom contain NaN or "
            "infinite values."
        )

    if np.any(degrees_of_freedom <= 0):
        raise ValueError(
            "All degrees of freedom must be positive."
        )

    height, width, _ = image.shape

    # Output shape: (H, W, number of tumor classes).
    unnormalized_posteriors = np.zeros(
        (
            height,
            width,
            num_classes,
        ),
        dtype=np.float64,
    )

    # Work only with pixels inside the brain.
    brain_pixels = image[brain_mask]

    if brain_pixels.size == 0:
        return unnormalized_posteriors

    for class_index in range(num_classes):
        # Calculate:
        #
        # log p(x | C_k)
        #
        log_likelihood = multivariate_skew_t_logpdf(
            pixels=brain_pixels,
            location=locations[class_index],
            dispersion=dispersions[class_index],
            skewness=skewness[class_index],
            degrees_of_freedom=(
                degrees_of_freedom[class_index]
            ),
        )

        # Calculate:
        #
        # log[p(x | C_k) P(C_k)]
        #
        log_unnormalized_posterior = (
            log_likelihood
            + np.log(priors[class_index])
        )

        # Return to the normal probability domain so this output
        # matches the existing Gaussian implementation.
        class_scores = np.exp(
            log_unnormalized_posterior
        )

        class_scores = np.nan_to_num(
            class_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        unnormalized_posteriors[
            brain_mask,
            class_index,
        ] = class_scores

    print(
        "Computed tumor skew-t scores for classes:",
        [str(name) for name in class_names],
    )

    return unnormalized_posteriors