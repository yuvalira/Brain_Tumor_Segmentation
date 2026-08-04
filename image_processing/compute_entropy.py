import numpy as np

def compute_entropy(
    posteriors: np.ndarray,
     brain_mask: np.ndarray
) -> np.ndarray:
    """Computes normalized Shannon entropy map [0, 1]."""
    p = np.clip(posteriors, 1e-12, 1.0)
    num_classes = p.shape[-1]
    entropy_map = -np.sum(p * np.log(p), axis=-1) / np.log(num_classes)
    entropy_map[~brain_mask] = 0.0
    return entropy_map