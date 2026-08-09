import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from config import *
from utilities.utils import load_and_normalize_slice

plt.style.use("dark_background")

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from config import *
from utilities.utils import load_and_normalize_slice

plt.style.use("dark_background")


def plot_local_vs_global_weight_changes(
    vol_num: int = 11,
    slice_num: int = SLICE_NUM,
    cols: int = 3,
):
    """
    Loads slice data, GMM priors, and local weight maps to display the percentage
    change of local weights relative to global priors for ALL components in a grid.
    """
    # 1. Load slice data and brain mask
    _, brain_mask, _ = load_and_normalize_slice(vol_num, slice_num)

    # 2. Load GMM parameters and Local Weight map
    gmm_path = os.path.join(
        PROJECT_ROOT, "saved_parameters", "statistical_models", "healthy_gmm.npz"
    )
    weights_path = os.path.join(
        PROJECT_ROOT, "saved_parameters", "statistical_models", "local_weights.npz"
    )

    gmm_data = np.load(gmm_path)
    weights_data = np.load(weights_path)

    global_weights = gmm_data["weights"]  # Shape: (K,)
    local_weights = weights_data["weights"]  # Shape: (H, W, K)

    num_components = len(global_weights)

    # 3. Dynamic Grid Dimensions
    rows = int(np.ceil(num_components / cols))

    # 4. Setup Plotting Parameters
    bwr_cmap = mpl.colormaps["bwr"].copy()
    bwr_cmap.set_bad(color="black")

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows), facecolor="black")
    axes_flat = axes.flatten() if num_components > 1 else [axes]

    for k in range(num_components):
        # Calculate percentage change: ((local - global) / global) * 100
        pct_diff = ((local_weights[:, :, k] - global_weights[k]) / global_weights[k]) * 100.0

        # Mask background pixels outside the brain with NaN
        masked_diff = np.where(brain_mask, pct_diff, np.nan)

        # Compute 95th percentile bound based on valid brain pixels
        valid_values = pct_diff[brain_mask]
        vbound = np.percentile(np.abs(valid_values), 95) if len(valid_values) > 0 else 100.0
        if vbound == 0 or np.isnan(vbound):
            vbound = 100.0

        # Plot symmetric divergence map around 0
        im = axes_flat[k].imshow(
            masked_diff, cmap=bwr_cmap, vmin=-vbound, vmax=vbound
        )

        axes_flat[k].set_title(
            f"Component {k} (Global $\pi_{{{k}}} = {global_weights[k]:.3f}$)\n"
            f"Bounds: $\pm{vbound:.1f}\%$",
            fontsize=10,
            color="white",
        )
        axes_flat[k].axis("off")

        # Colorbar configuration
        cbar = fig.colorbar(im, ax=axes_flat[k], fraction=0.046, pad=0.04)
        cbar.set_label("% Change", fontsize=8, color="white")
        cbar.ax.tick_params(labelsize=8, colors="white")

    # Hide unused grid subplots if total components < rows * cols
    for k in range(num_components, len(axes_flat)):
        axes_flat[k].axis("off")

    plt.tight_layout()
    plt.show()


# Example Call
if __name__ == "__main__":
    plot_local_vs_global_weight_changes(vol_num=11, slice_num=SLICE_NUM, cols=3)