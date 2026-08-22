import numpy as np
import matplotlib.pyplot as plt

def plot_segmentation_boxplots(dice_scores, iou_scores):
    """
    Plot Dice and IoU box plots for multiple segmentation methods.

    Parameters
    ----------
    dice_scores : dict
        Dictionary mapping method names to arrays of Dice scores.

    iou_scores : dict
        Dictionary mapping method names to arrays of IoU scores.
    """

    # ---------------------------------------------------------
    # Dice
    # ---------------------------------------------------------
    labels = list(dice_scores.keys())

    dice_data = [
        np.asarray(scores).flatten()
        for scores in dice_scores.values()
    ]

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        dice_data,
        labels=labels,
        showmeans=True,
    )

    plt.ylabel("Dice Score")
    plt.xlabel("Feature Configuration")
    plt.title("Dice Score Comparison")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # IoU
    # ---------------------------------------------------------
    labels = list(iou_scores.keys())

    iou_data = [
        np.asarray(scores).flatten()
        for scores in iou_scores.values()
    ]

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        iou_data,
        labels=labels,
        showmeans=True,
    )

    plt.ylabel("IoU Score")
    plt.xlabel("Feature Configuration")
    plt.title("IoU Score Comparison")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()