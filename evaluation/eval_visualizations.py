from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_histogram(
    title: str,
    file_path: str | Path,
    metric_key: str = "dice_per_volume",
) -> None:
    """Loads a metrics .npz file and plots a histogram with 20 bins between 0 and 1."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    # 1. Open and load the .npz file
    with np.load(file_path) as data:
        if metric_key not in data:
            raise KeyError(
                f"Key '{metric_key}' not found in {file_path}. "
                f"Available keys: {list(data.keys())}"
            )

        scores = data[metric_key]
        model_name = (
            str(data["model"])
            if "model" in data
            else file_path.stem.split("_")[0]
        )

    # 2. Compute mean and std before binning
    mean_val = np.mean(scores)
    std_val = np.std(scores)

    # 3. Define 20 equal sections (bins) between 0 and 1
    bins = np.linspace(0.0, 1.0, 21)  # 21 edge points create exactly 20 bins

    # 4. Build the histogram
    plt.figure(figsize=(8, 5))
    plt.hist(
        scores,
        bins=bins,
        edgecolor="black",
        alpha=0.75,
        color="#2b5c8f",
        rwidth=0.9,
    )

    metric_name = (
        "Dice Score"
        if "dice" in metric_key.lower()
        else "IoU Score"
        if "iou" in metric_key.lower()
        else metric_key
    )

    plt.title(
        f"{title} {metric_name} Distribution (Mean: {mean_val:.3f}, Std: {std_val:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel(f"{metric_name}", fontsize=11)
    plt.ylabel("Number of Volumes", fontsize=11)

    plt.xlim(0.0, 1.0)
    plt.xticks(np.linspace(0.0, 1.0, 11))  # Ticks every 0.1
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    # Save figure next to the metrics file
    output_png = file_path.parent / f"{file_path.stem}_{metric_key}_hist.png"
    plt.savefig(output_png, dpi=300)
    print(f"Histogram saved successfully to: {output_png}")
    plt.close()


# --- Function Calls ---

plot_metrics_histogram(
    "Single Gaussian",
    "evaluation/metrics/single_gaussian_metrics.npz",
    metric_key="dice_per_volume",
)

plot_metrics_histogram(
    "Single Gaussian",
    "evaluation/metrics/single_gaussian_metrics.npz",
    metric_key="iou_per_volume",
)