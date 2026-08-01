import numpy as np
from scipy.ndimage import sobel
from scipy.stats import multivariate_normal



def posterior_inference(
    slice_im: np.ndarray, brain_mask: np.ndarray, param_file: str
) -> tuple[np.ndarray, list[str]]:
    """Computes 6-class joint posterior probabilities using exact prior weighting."""
    class_names = [
        "NCR/NET",
        "ED",
        "ET",
        "Healthy WM",
        "Healthy GM",
        "Healthy CSF",
    ]
    H, W, C = slice_im.shape
    posteriors_6d = np.zeros((H, W, 6), dtype=np.float64)

    if not np.any(brain_mask):
        return posteriors_6d, class_names

    # Load parameters with prior structure
    params = np.load(param_file, allow_pickle=True)
    tumor_priors = params["tumor_priors"]
    healthy_prior = params["healthy_prior"]
    healthy_weights = params["healthy_weights"]

    pdfs_tumor = [
        multivariate_normal(
            mean=params["tumor_means"][i], cov=params["tumor_covariances"][i]
        )
        for i in range(3)
    ]
    pdfs_healthy = [
        multivariate_normal(
            mean=params["healthy_means"][m],
            cov=params["healthy_covariances"][m],
        )
        for m in range(3)
    ]

    flat_pixels = slice_im.reshape(-1, C)
    weighted_likelihoods = np.zeros(
        (flat_pixels.shape[0], 6), dtype=np.float64
    )

    brain_flat = brain_mask.ravel()
    brain_pixels = flat_pixels[brain_flat]

    # Evaluate Tumor Classes (0, 1, 2)
    for c in range(3):
        weighted_likelihoods[brain_flat, c] = (
            pdfs_tumor[c].pdf(brain_pixels) * tumor_priors[c]
        )

    # Evaluate Healthy Classes (3, 4, 5)
    for m in range(3):
        weighted_likelihoods[brain_flat, 3 + m] = (
            pdfs_healthy[m].pdf(brain_pixels)
            * healthy_weights[m]
            * healthy_prior
        )

    # Evidence & Normalization
    evidence = np.sum(weighted_likelihoods, axis=1, keepdims=True) + 1e-12
    posteriors = weighted_likelihoods / evidence
    posteriors_6d = posteriors.reshape(H, W, 6)

    return posteriors_6d, class_names




def compute_entropy(
    posteriors_6d: np.ndarray, brain_mask: np.ndarray
) -> np.ndarray:
    """Computes normalized Shannon entropy map [0, 1]."""
    p = np.clip(posteriors_6d, 1e-12, 1.0)
    num_classes = p.shape[-1]
    entropy_map = -np.sum(p * np.log(p), axis=-1) / np.log(num_classes)
    entropy_map[~brain_mask] = 0.0
    return entropy_map


def sum_tumor_posterior(posteriors_6d: np.ndarray) -> np.ndarray:
    """Sums the posterior probabilities of the 3 tumor sub-classes (NCR/NET, ED, ET)."""
    return np.sum(posteriors_6d[:, :, 0:3], axis=-1)


def sobel_edge_detection(
    image_2d: np.ndarray, brain_mask: np.ndarray
) -> np.ndarray:
    """Computes 2D Sobel gradient magnitude map."""
    gx = sobel(image_2d, axis=0)
    gy = sobel(image_2d, axis=1)
    mag = np.hypot(gx, gy)
    mag[~brain_mask] = 0.0
    return mag