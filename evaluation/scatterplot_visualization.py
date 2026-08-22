import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def plot_score_correlations(
    raw,
    boundary,
    symmetric,
    full,
    metric_name="Dice",
):
    """
    Compare Raw scores against the other feature configurations.

    Each subplot shows:
        - Scatter plot
        - Identity line (y = x)
        - Pearson correlation coefficient and p-value
        - Mean score improvement relative to Raw
        - Percentage of cases improved relative to Raw

    Parameters
    ----------
    raw : array-like
        Scores from the raw-feature method.

    boundary : array-like
        Scores from the boundary-distance method.

    symmetric : array-like
        Scores from the symmetric-feature method.

    full : array-like
        Scores from the full/all-modalities method.

    metric_name : str, optional
        Name of the metric being plotted, e.g. "Dice" or "IoU".
    """

    raw = np.asarray(raw).flatten()
    boundary = np.asarray(boundary).flatten()
    symmetric = np.asarray(symmetric).flatten()
    full = np.asarray(full).flatten()

    comparisons = [
        ("Boundary Distance", boundary),
        ("Symmetric", symmetric),
        ("Full", full),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (method_name, scores) in zip(axes, comparisons):

        # -----------------------------------------------------
        # Remove NaN / inf values pairwise
        # -----------------------------------------------------
        valid = np.isfinite(raw) & np.isfinite(scores)

        x = raw[valid]
        y = scores[valid]

        # -----------------------------------------------------
        # Pearson correlation
        # -----------------------------------------------------
        r, p = pearsonr(x, y)

        # -----------------------------------------------------
        # Improvement relative to Raw
        # -----------------------------------------------------
        delta = y - x

        mean_delta = np.mean(delta)
        median_delta = np.median(delta)

        improved_percent = 100 * np.mean(delta > 0)
        degraded_percent = 100 * np.mean(delta < 0)
        equal_percent = 100 * np.mean(delta == 0)

        # -----------------------------------------------------
        # Scatter plot
        # -----------------------------------------------------
        ax.scatter(
            x,
            y,
            alpha=0.7,
            edgecolors="black",
        )

        # -----------------------------------------------------
        # Identity line: y = x
        # -----------------------------------------------------
        ax.plot(
            [0, 1],
            [0, 1],
            "--",
            linewidth=1.5,
            label="y = x",
        )

        # -----------------------------------------------------
        # Axes
        # -----------------------------------------------------
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_xlabel(f"Raw {metric_name}")
        ax.set_ylabel(f"{method_name} {metric_name}")

        # -----------------------------------------------------
        # Title with statistics
        # -----------------------------------------------------
        ax.set_title(
            f"Raw vs {method_name}\n"
            f"Pearson r = {r:.3f}, p = {p:.2e}\n"
            f"Mean Δ = {mean_delta:+.3f} | "
            f"Improved = {improved_percent:.1f}%"
        )

        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"{metric_name} Score Correlations",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()