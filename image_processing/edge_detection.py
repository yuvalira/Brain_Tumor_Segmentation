import numpy as np
from scipy.ndimage import sobel

# def sobel_edge_detection(
#     image: np.ndarray, brain_mask: np.ndarray
# ) -> np.ndarray:
#     """Computes 2D Sobel gradient magnitude map averaged across channels."""
#     gx = sobel(image, axis=0)
#     gy = sobel(image, axis=1)
    
#     # Gradient magnitude per channel (always non-negative)
#     mag = np.hypot(gx, gy)  # Shape: (H, W, C) if multi-channel, else (H, W)

#     # Compute mean across channel dimension
#     if mag.ndim == 3:
#         mag = np.mean(np.abs(mag), axis=-1)

#     mag[~brain_mask] = 0.0
#     return mag


def sobel_edge_detection(
    posteriors: np.ndarray, brain_mask: np.ndarray
) -> np.ndarray:
    """Sum the posteriors of the final three channels (tumor classes)

    and compute the 2D Sobel gradient magnitude map on the merged layer.
    """
    # Sum the last 3 channels (tumor classes) -> Shape: (H, W)
    if posteriors.ndim == 3 and posteriors.shape[-1] >= 3:
        tumor_merged_prob = np.sum(posteriors[..., -3:], axis=-1)
    else:
        tumor_merged_prob = posteriors.squeeze()

    # Compute horizontal and vertical gradients along 2D spatial axes
    gx = sobel(tumor_merged_prob, axis=0)
    gy = sobel(tumor_merged_prob, axis=1)

    # Compute gradient magnitude
    mag = np.hypot(gx, gy)

    # Zero out background outside brain mask
    mag[~brain_mask] = 0.0

    return mag