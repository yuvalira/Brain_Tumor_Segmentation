import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from config import *
from utilities.utils import load_and_normalize_slice

plt.style.use("dark_background")


def plot_gmm_component_likelihoods(
    vol_num: int = 11,
    slice_num: int = SLICE_NUM,
    symmetric: bool = False,
    cols: int = 3,
):
    """
    Evaluates a slice using each GMM component separately to construct a
    (240, 240, num_components) likelihood tensor, then plots each component map.
    """
    # 1. Determine filenames based on symmetric parameter
    gmm_filename = "healthy_gmm_symmetric.npz" if symmetric else "healthy_gmm.npz"
    gmm_path = os.path.join(
        PROJECT_ROOT, "saved_parameters", "statistical_models", gmm_filename
    )

    # 2. Load GMM parameters
    gmm = np.load(gmm_path)
    global_weights = gmm["weights"]  # Shape: (K,)
    means = gmm["means"]              # Shape: (K, C)
    covariances = gmm["covariances"]  # Shape: (K, C, C)

    num_components = len(global_weights)

    # 3. Load normalized slice image and brain mask
    slice_output = load_and_normalize_slice(vol_num, slice_num, symmetric=symmetric)
    features_image = slice_output[0]  # Shape: (H, W, C)
    brain_mask = slice_output[1]

    H, W, C = features_image.shape

    # 4. Flatten image for batch multivariate PDF evaluation
    X_flat = features_image.reshape(-1, C).astype(np.float64)

    # Tensor buffer for separate component likelihoods: (240, 240, K)
    component_likelihood_maps = np.zeros((H, W, num_components), dtype=np.float64)

    # 5. Evaluate Multivariate Gaussian Density for each component separately
    for k in range(num_components):
        pdf_k = multivariate_normal.pdf(
            X_flat,
            mean=means[k],
            cov=covariances[k],
            allow_singular=True,
        )

        # Un-weighted density map for component k
        pdf_k_2d = pdf_k.reshape(H, W)

        # Zero out background non-brain pixels and store
        component_likelihood_maps[:, :, k] = np.where(brain_mask, pdf_k_2d, np.nan)

    # 6. Plotting Configuration
    rows = int(np.ceil(num_components / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 4.2 * rows), facecolor="black"
    )
    axes_flat = axes.flatten() if num_components > 1 else [axes]

    inferno_cmap = mpl.colormaps["inferno"].copy()
    inferno_cmap.set_bad(color="black")

    for k in range(num_components):
        im = axes_flat[k].imshow(
            component_likelihood_maps[:, :, k], cmap=inferno_cmap
        )

        axes_flat[k].set_title(
            f"Component {k}\n(Global Weight $\pi_{{{k}}} = {global_weights[k]:.3f}$)",
            fontsize=10,
            color="white",
        )
        axes_flat[k].axis("off")

        cbar = fig.colorbar(im, ax=axes_flat[k], fraction=0.046, pad=0.04)
        cbar.set_label("Likelihood Density", fontsize=8, color="white")
        cbar.ax.tick_params(labelsize=8, colors="white")

    # Hide unused subplots in the layout grid
    for k in range(num_components, len(axes_flat)):
        axes_flat[k].axis("off")

    plt.suptitle(
        f"Individual GMM Component Likelihood Maps (Vol {vol_num}, Slice {slice_num})",
        fontsize=12,
        fontweight="bold",
        color="white",
        y=0.99,
    )
    plt.tight_layout()
    plt.show()

    return component_likelihood_maps


# Example Call
if __name__ == "__main__":
    likelihood_tensor = plot_gmm_component_likelihoods(
        vol_num=11, slice_num=SLICE_NUM, symmetric=False, cols=3
    )