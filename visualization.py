import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation


def _overlay_gt(ax, gt_mask):
    """Utility to overlay true GT boundary in bright cyan."""
    gt_binary = gt_mask.sum(axis=-1) > 0
    if np.any(gt_binary):
        contour = binary_dilation(gt_binary) ^ gt_binary
        c_disp = np.zeros((*contour.shape, 4))
        c_disp[contour] = [0.0, 1.0, 1.0, 0.9]
        ax.imshow(c_disp, origin="lower", interpolation="nearest")


def _get_masked_modality(slice_im, brain_mask, channel_idx):
    """Masks background pixels to the brain's minimum value for clean black backgrounds."""
    frame = slice_im[:, :, channel_idx].copy()
    if np.any(brain_mask):
        frame[~brain_mask] = np.min(frame[brain_mask])
    return frame


def visualize_modalities(
    slice_im: np.ndarray, brain_mask: np.ndarray, save_path: str
):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = [
        "1. T1 Modality",
        "2. T1ce Modality",
        "3. T2 Modality",
        "4. FLAIR Modality",
    ]

    for i in range(4):
        frame = _get_masked_modality(slice_im, brain_mask, i)
        axes[i].imshow(frame, cmap="gray", origin="lower")
        axes[i].set_title(titles[i], fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()


def visualize_probability(
    slice_im: np.ndarray,
    posteriors_6d: np.ndarray,
    brain_mask: np.ndarray,
    gt_mask: np.ndarray,
    save_path: str,
):
    """Figure 2: Probability maps with GT Tumor Mask in panel [0,3]."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # [0,0] Clean T1ce Scan
    t1ce = _get_masked_modality(slice_im, brain_mask, 1)
    axes[0].imshow(t1ce, cmap="gray", origin="lower")
    axes[0].set_title("[0,0] Structural T1ce Scan", fontsize=10)

    # [0,1] Healthy RGB Posteriors
    healthy_rgb = np.clip(posteriors_6d[:, :, 3:6], 0, 1)
    healthy_rgb[~brain_mask] = 0.0
    axes[1].imshow(healthy_rgb, origin="lower")
    axes[1].set_title("[0,1] Healthy GMM Posteriors (RGB)", fontsize=10)

    # [0,2] Tumor RGB Posteriors
    tumor_rgb = np.clip(posteriors_6d[:, :, 0:3], 0, 1)
    tumor_rgb[~brain_mask] = 0.0
    axes[2].imshow(tumor_rgb, origin="lower")
    axes[2].set_title("[0,2] Tumor GMM Posteriors (RGB)", fontsize=10)

    # [0,3] GT Tumor RGB Mask
    gt_rgb = np.clip(gt_mask, 0, 1)
    axes[3].imshow(gt_rgb, origin="lower")
    axes[3].set_title(
        "[0,3] Ground Truth Tumor Mask\n(Red: NCR/NET | Green: ED | Blue: ET)",
        fontsize=10,
    )

    for ax in axes:
        _overlay_gt(ax, gt_mask)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()


def visualize_gt_vs_prob(
    slice_im: np.ndarray,
    posteriors_6d: np.ndarray,
    brain_mask: np.ndarray,
    gt_mask: np.ndarray,
    save_path: str,
):
    """New Figure: Compares T1ce, GT Mask, Tumor Probability, and ML Map (all with GT boundary)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Structural T1ce Scan
    t1ce = _get_masked_modality(slice_im, brain_mask, 1)
    axes[0].imshow(t1ce, cmap="gray", origin="lower")
    axes[0].set_title("1. Structural T1ce Scan", fontsize=10)

    # 2. RGB GT Mask
    gt_rgb = np.clip(gt_mask, 0, 1)
    axes[1].imshow(gt_rgb, origin="lower")
    axes[1].set_title(
        "2. Ground Truth RGB Mask\n(Red: NCR/NET | Green: ED | Blue: ET)",
        fontsize=10,
    )

    # 3. RGB Tumor Probability Map
    tumor_rgb = np.clip(posteriors_6d[:, :, 0:3], 0, 1)
    tumor_rgb[~brain_mask] = 0.0
    axes[2].imshow(tumor_rgb, origin="lower")
    axes[2].set_title("3. RGB Tumor Probability Map", fontsize=10)

    # 4. Maximum Likelihood Tumor RGB Map
    argmax_c = np.argmax(posteriors_6d, axis=-1)
    ml_rgb = np.zeros((*posteriors_6d.shape[:2], 3), dtype=np.float32)

    tumor_colors = {
        0: [1.0, 0.0, 0.0],  # Red (NCR/NET)
        1: [0.0, 1.0, 0.0],  # Green (ED)
        2: [0.0, 0.0, 1.0],  # Blue (ET)
    }
    for c in range(3):
        ml_rgb[(argmax_c == c) & brain_mask] = tumor_colors[c]

    # Healthy argmax shown in dark gray
    healthy_voxels = (argmax_c >= 3) & brain_mask
    ml_rgb[healthy_voxels] = [0.2, 0.2, 0.2]

    ml_rgb[~brain_mask] = [0.0, 0.0, 0.0]
    axes[3].imshow(ml_rgb, origin="lower")
    axes[3].set_title("4. Maximum Likelihood Tumor RGB Map", fontsize=10)

    # Apply cyan GT boundary overlay to ALL subplots
    for ax in axes:
        _overlay_gt(ax, gt_mask)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()


def visualize_shape_detection(
    slice_im: np.ndarray,
    sobel_map: np.ndarray,
    all_blobs: np.ndarray,
    is_tumor_list: list[bool],
    tumor_blobs: np.ndarray,
    brain_mask: np.ndarray,
    gt_mask: np.ndarray,
    save_path: str,
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # [0,0] Structural T1ce Scan
    t1ce = _get_masked_modality(slice_im, brain_mask, 1)
    axes[0].imshow(t1ce, cmap="gray", origin="lower")
    axes[0].set_title("[0,0] Structural T1ce Scan", fontsize=10)

    # [0,1] Sobel Map
    axes[1].imshow(sobel_map, cmap="magma", origin="lower")
    axes[1].set_title("[0,1] Sobel on Sum of Tumor Posteriors", fontsize=10)

    # [0,2] All Blobs Overlaid
    axes[2].imshow(sobel_map, cmap="magma", origin="lower")
    if all_blobs.shape[-1] > 0:
        for i in range(all_blobs.shape[-1]):
            cnts, _ = cv2.findContours(
                all_blobs[:, :, i].astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            col = "lime" if is_tumor_list[i] else "red"
            for cnt in cnts:
                pts = cnt.squeeze()
                if pts.ndim == 2:
                    axes[2].plot(pts[:, 0], pts[:, 1], color=col, linewidth=1.2)
    axes[2].set_title(
        "[0,2] Classified Blobs (Lime: Tumor | Red: Rejected)", fontsize=10
    )

    # [0,3] Only Retained Tumor Blobs Overlaid
    axes[3].imshow(sobel_map, cmap="magma", origin="lower")
    if tumor_blobs.shape[-1] > 0:
        for i in range(tumor_blobs.shape[-1]):
            cnts, _ = cv2.findContours(
                tumor_blobs[:, :, i].astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for cnt in cnts:
                pts = cnt.squeeze()
                if pts.ndim == 2:
                    axes[3].plot(
                        pts[:, 0], pts[:, 1], color="yellow", linewidth=1.5
                    )
    axes[3].set_title(
        "[0,3] Retained Tumor Blobs (Yellow Overlay)", fontsize=10
    )

    for ax in axes:
        _overlay_gt(ax, gt_mask)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()


def visualize_expansion(
    slice_im: np.ndarray,
    entropy_map: np.ndarray,
    tumor_blobs: np.ndarray,
    expanded_seg: np.ndarray,
    posteriors_6d: np.ndarray,
    brain_mask: np.ndarray,
    gt_mask: np.ndarray,
    save_path: str,
    entropy_thresh: float = 0.25,
    posterior_min: float = 0.05,
):
    """Figure 4: Shows expansion with initial seed blob boundary overlaid distinctly on top of the expanded mask."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # [0,0] Clean T1ce
    t1ce = _get_masked_modality(slice_im, brain_mask, 1)
    axes[0].imshow(t1ce, cmap="gray", origin="lower")
    axes[0].set_title("[0,0] Structural T1ce Scan", fontsize=10)

    # [0,1] Entropy Map
    axes[1].imshow(entropy_map, cmap="inferno", origin="lower", vmin=0, vmax=1)
    axes[1].set_title("[0,1] Normalized Entropy Map", fontsize=10)

    # [0,2] Global Threshold Match Overlaid
    tumor_sum = np.sum(posteriors_6d[:, :, 0:3], axis=-1)
    global_candidates = (
        (entropy_map >= entropy_thresh)
        & (tumor_sum >= posterior_min)
        & brain_mask
    )

    t1ce_norm = (t1ce - t1ce.min()) / (t1ce.max() - t1ce.min() + 1e-8)
    t1ce_norm[~brain_mask] = 0.0

    overlay_global = np.dstack([t1ce_norm] * 3)
    overlay_global[global_candidates] = [1.0, 0.5, 0.0]  # Orange global mask
    axes[2].imshow(overlay_global, origin="lower")
    axes[2].set_title(
        "[0,2] All Pixels Meeting Thresholds (Global)", fontsize=10
    )

    # [0,3] Expanded Mask
    overlay_exp = np.dstack([t1ce_norm] * 3)
    overlay_exp[expanded_seg] = [1.0, 1.0, 0.0]  # Yellow expanded mask
    axes[3].imshow(overlay_exp, origin="lower")

    # OVERLAY INITIAL SEED CONTOUR (High zorder ensures it renders ABOVE the yellow fill)
    if tumor_blobs.shape[-1] > 0:
        for i in range(tumor_blobs.shape[-1]):
            cnts, _ = cv2.findContours(
                tumor_blobs[:, :, i].astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for cnt in cnts:
                pts = cnt.squeeze()
                if pts.ndim == 2:
                    # Draw Red Dashed line over the Yellow Expanded Blob
                    axes[3].plot(
                        pts[:, 0],
                        pts[:, 1],
                        color="red",
                        linestyle="-",
                        linewidth=1.8,
                        zorder=10,
                        label="Initial Seed Blob" if i == 0 else "",
                    )

    axes[3].set_title(
        "[0,3] Blob Expansion (Red: Seed | Yellow: Expanded)",
        fontsize=10,
    )

    # Ground Truth Cyan Boundary Overlay
    for ax in axes:
        _overlay_gt(ax, gt_mask)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()

def visualize_row_analysis(
    posteriors_6d: np.ndarray,
    entropy_map: np.ndarray,
    brain_mask: np.ndarray,
    gt_mask: np.ndarray,
    target_row: int,
    save_path: str,
):
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 4)

    # [0,0] RGB Tumor Map
    ax0 = fig.add_subplot(gs[0, 0])
    tumor_rgb = np.clip(posteriors_6d[:, :, 0:3], 0, 1)
    tumor_rgb[~brain_mask] = 0.0
    ax0.imshow(tumor_rgb, origin="lower")
    ax0.axhline(
        y=target_row, color="yellow", linestyle="--", linewidth=1.2, alpha=0.85
    )
    _overlay_gt(ax0, gt_mask)
    ax0.set_title(
        f"[0,0] RGB Tumor Map\n(Yellow Line: Row {target_row})", fontsize=10
    )
    ax0.axis("off")

    # [0,1..0,3] 1D Posterior Profile Plot
    ax_profile = fig.add_subplot(gs[0, 1:])
    W = posteriors_6d.shape[1]
    x_coords = np.arange(W)
    row_posteriors = posteriors_6d[target_row, :, :]
    row_entropy = entropy_map[target_row, :]

    colors = ["red", "lime", "blue", "#404040", "#808080", "#C0C0C0"]
    labels = [
        "NCR/NET",
        "ED",
        "ET",
        "Healthy WM",
        "Healthy GM",
        "Healthy CSF",
    ]

    ax_profile.stackplot(
        x_coords, row_posteriors.T, colors=colors, labels=labels, alpha=0.75
    )
    ax_profile.plot(
        x_coords,
        row_entropy,
        color="#FF00FF",
        linewidth=2.0,
        label="Entropy (Uncertainty)",
    )

    # Cyan Dashed Lines for Ground Truth Intersections
    gt_binary = gt_mask.sum(axis=-1) > 0
    if np.any(gt_binary):
        gt_contour = binary_dilation(gt_binary) ^ gt_binary
        gt_intersections = np.where(gt_contour[target_row, :])[0]
        for i, x_gt in enumerate(gt_intersections):
            ax_profile.axvline(
                x=x_gt,
                color="cyan",
                linestyle="--",
                linewidth=1.8,
                label="True GT Boundary" if i == 0 else "",
            )

    ax_profile.set_xlim(0, W - 1)
    ax_profile.set_ylim(0, 1.05)
    ax_profile.set_xlabel("Pixel Column Index (X)", fontsize=10)
    ax_profile.set_ylabel("Probability / Entropy", fontsize=10)
    ax_profile.set_title(
        f"[0,1..0,3] 1D Posterior Profile & Entropy Overlay (Row y={target_row})",
        fontsize=10,
    )
    ax_profile.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()


def visualize_entropy(
    entropy_map: np.ndarray, gt_mask: np.ndarray, save_path: str
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(
        entropy_map, cmap="inferno", origin="lower", vmin=0.0, vmax=1.0
    )
    _overlay_gt(ax, gt_mask)
    ax.set_title("Normalized Entropy Map with True GT Boundary", fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()


def visualize_evaluation(
    slice_im: np.ndarray,
    expanded_seg: np.ndarray,
    brain_mask: np.ndarray,
    gt_mask: np.ndarray,
    save_path: str,
):
    """Generates a single-plot evaluation figure displaying confusion matrix classification

    overlaid on the T1ce scan:
      - True Positives (TP): Green
      - False Negatives (FN): Red (Missed tumor pixels)
      - False Positives (FP): Blue (Over-segmented pixels)
      - True Negatives (TN): Grayscale T1ce background/tissue
    """
    # 1. Prepare base structural T1ce image in grayscale
    t1ce = _get_masked_modality(slice_im, brain_mask, 1)
    t1ce_norm = (t1ce - t1ce.min()) / (t1ce.max() - t1ce.min() + 1e-8)
    t1ce_norm[~brain_mask] = 0.0

    # Initialize 3-channel RGB background using grayscale T1ce
    eval_rgb = np.dstack([t1ce_norm] * 3)

    # 2. Extract ground truth binary tumor mask and predicted segmentation mask
    gt_binary = (gt_mask.sum(axis=-1) > 0) & brain_mask
    pred_binary = expanded_seg & brain_mask

    # 3. Compute Confusion Matrix Masks
    tp_mask = pred_binary & gt_binary  # True Positives
    fn_mask = (~pred_binary) & gt_binary  # False Negatives
    fp_mask = pred_binary & (~gt_binary)  # False Positives

    # 4. Color Assignment
    eval_rgb[tp_mask] = [0.0, 1.0, 0.0]  # Green for True Positives
    eval_rgb[fn_mask] = [1.0, 0.0, 0.0]  # Red for False Negatives
    eval_rgb[fp_mask] = [0.0, 0.0, 1.0]  # Blue for False Positives

    # 5. Plotting single figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(eval_rgb, origin="lower")

    # Add GT Cyan boundary for reference
    _overlay_gt(ax, gt_mask)

    ax.set_title(
        "Segmentation Evaluation Map vs. GT\n"
        "(Green: TP | Red: FN | Blue: FP | Grayscale: TN)",
        fontsize=11,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()