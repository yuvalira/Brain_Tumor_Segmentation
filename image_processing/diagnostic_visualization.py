import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def produce_diagnostic_figure(
    image,
    posteriors,
    brain_mask,
    gt_mask,
    entropy_map,
    edge_map,
    blob_array,
    is_tumor_list,
    final_segmentation,
    save_path=None,
):
    """Generates a 2x4 diagnostic grid combining raw MRI, ground truth, posterior maps,

    entropy, Sobel edges, classified blob contours, seed expansion, and the final segmentation.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # -------------------------------------------------------------------------
    # Helper: Build Composite Posterior Image
    # -------------------------------------------------------------------------
    tumor_rgb = posteriors[..., -3:]  # RGB channels for last 3 tumor classes
    num_healthy = posteriors.shape[-1] - 3
    if num_healthy > 0:
        grayscale_weights = np.linspace(0.2, 0.8, num_healthy)
        healthy_gray = np.sum(
            posteriors[..., :num_healthy] * grayscale_weights,
            axis=-1,
            keepdims=True,
        )
        prob_composite = np.clip(healthy_gray + tumor_rgb, 0.0, 1.0)
    else:
        prob_composite = np.clip(tumor_rgb, 0.0, 1.0)
    prob_composite[brain_mask == 0] = 0.0

    # Helper: Normalize Ground Truth Mask to RGB if needed
    if gt_mask.ndim == 3 and gt_mask.shape[-1] == 3:
        gt_rgb = gt_mask.copy()
    else:
        gt_mask_2d = gt_mask[..., 0] if gt_mask.ndim > 2 else gt_mask
        color_map = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        gt_clean = np.clip(gt_mask_2d.astype(int), 0, len(color_map) - 1)
        gt_rgb = color_map[gt_clean]
    gt_rgb[brain_mask == 0] = 0.0

    # Helper: Final Segmentation to RGB
    if final_segmentation.ndim == 3 and final_segmentation.shape[-1] == 3:
        seg_rgb = final_segmentation.copy()
    else:
        seg_2d = (
            final_segmentation[..., 0]
            if final_segmentation.ndim > 2
            else final_segmentation
        )
        color_map = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        seg_clean = np.clip(seg_2d.astype(int), 0, len(color_map) - 1)
        seg_rgb = color_map[seg_clean]
    seg_rgb[brain_mask == 0] = 0.0

    # -------------------------------------------------------------------------
    # Contour Extraction
    # -------------------------------------------------------------------------
    num_blobs = blob_array.shape[-1] if blob_array.ndim == 3 else 0
    tumor_contours, non_tumor_contours = [], []
    seed_mask_combined = np.zeros(brain_mask.shape, dtype=bool)

    for i in range(num_blobs):
        mask_i = blob_array[:, :, i].astype(np.uint8)
        cnts, _ = cv2.findContours(
            mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if is_tumor_list[i]:
            tumor_contours.extend(cnts)
            seed_mask_combined |= mask_i.astype(bool)
        else:
            non_tumor_contours.extend(cnts)

    # Post-expansion binary mask
    if final_segmentation.ndim > 2 and final_segmentation.shape[-1] == 3:
        expanded_mask = (np.sum(final_segmentation, axis=-1) > 0) & (brain_mask > 0)
    else:
        expanded_mask = (final_segmentation > 0) & (brain_mask > 0)

    # Difference between final expanded region and original tumor seeds
    expansion_only_mask = expanded_mask & (~seed_mask_combined)

    # =========================================================================
    # ROW 0
    # =========================================================================
    # (0, 0) Raw Modality (T1ce)
    t1ce_channel = 1 if image.shape[-1] > 1 else 0
    raw_im = image[..., t1ce_channel].copy()
    raw_im[brain_mask == 0] = 0.0
    axes[0, 0].imshow(raw_im, cmap="gray")
    axes[0, 0].set_title("Raw Modality (T1ce)")
    axes[0, 0].axis("off")

    # (0, 1) RGB Ground Truth Mask
    axes[0, 1].imshow(gt_rgb)
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis("off")

    # (0, 2) Posterior Probability Map
    axes[0, 2].imshow(prob_composite)
    axes[0, 2].set_title("Posterior Map")
    axes[0, 2].axis("off")

    # (0, 3) Entropy Map
    entropy_masked = np.copy(entropy_map)
    entropy_masked[brain_mask == 0] = np.nan
    cmap_entropy = plt.cm.magma.copy()
    cmap_entropy.set_bad(color="black")
    im_ent = axes[0, 3].imshow(entropy_masked, cmap=cmap_entropy)
    axes[0, 3].set_title("Entropy Map")
    axes[0, 3].axis("off")
    fig.colorbar(
        im_ent, ax=axes[0, 3], fraction=0.046, pad=0.04, label="Bits"
    )

    # -------------------------------------------------------------------------
    # Build Pixel-wise TP / FN / FP Error Map
    # -------------------------------------------------------------------------
    pred_bin = (
        np.sum(final_segmentation, axis=-1) > 0
        if (final_segmentation.ndim > 2 and final_segmentation.shape[-1] == 3)
        else (final_segmentation > 0)
    )

    gt_bin = (
        np.sum(gt_mask, axis=-1) > 0 if gt_mask.ndim > 2 else (gt_mask > 0)
    )
    brain_bin = brain_mask.astype(bool)

    tp = pred_bin & gt_bin & brain_bin
    fn = (~pred_bin) & gt_bin & brain_bin
    fp = pred_bin & (~gt_bin) & brain_bin

    error_map = np.zeros((*brain_mask.shape, 3), dtype=np.float32)
    error_map[tp] = [0.0, 1.0, 0.0]  # Green: True Positive
    error_map[fn] = [1.0, 0.0, 0.0]  # Red:   False Negative
    error_map[fp] = [0.0, 0.0, 1.0]  # Blue:  False Positive

    # =========================================================================
    # ROW 1
    # =========================================================================
    # (1, 0) Sobel Edge Map
    sobel_masked = np.copy(edge_map)
    sobel_masked[brain_mask == 0] = np.nan
    cmap_sobel = plt.cm.viridis.copy()
    cmap_sobel.set_bad(color="black")
    im_sob = axes[1, 0].imshow(sobel_masked, cmap=cmap_sobel)
    axes[1, 0].set_title("Sobel Edge Map")
    axes[1, 0].axis("off")
    fig.colorbar(
        im_sob, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Gradient"
    )

    # (1, 1) Classified Contours overlaid on Posterior Map
    axes[1, 1].imshow(prob_composite)
    for cnt in tumor_contours:
        pts = cnt.squeeze(axis=1)
        if pts.ndim == 2 and len(pts) > 1:
            pts = np.vstack([pts, pts[0]])
            axes[1, 1].plot(
                pts[:, 0], pts[:, 1], color="red", linewidth=1.5, label="Tumor"
            )
    for cnt in non_tumor_contours:
        pts = cnt.squeeze(axis=1)
        if pts.ndim == 2 and len(pts) > 1:
            pts = np.vstack([pts, pts[0]])
            axes[1, 1].plot(
                pts[:, 0],
                pts[:, 1],
                color="lime",
                linewidth=1.5,
                label="Non-Tumor",
            )
    axes[1, 1].set_title("Classified Contours")
    axes[1, 1].axis("off")

    # (1, 2) Posterior + Tumor Contours (Cyan) + Seed Expansion (Yellow RGBA)
    axes[1, 2].imshow(prob_composite)
    # Semitransparent yellow overlay for expanded area (R=1, G=1, B=0, A=0.35)
    yellow_rgba = np.zeros((*brain_mask.shape, 4), dtype=np.float32)
    yellow_rgba[expansion_only_mask] = [1.0, 1.0, 0.0, 0.35]
    axes[1, 2].imshow(yellow_rgba)

    # Tumor contours in Cyan
    for cnt in tumor_contours:
        pts = cnt.squeeze(axis=1)
        if pts.ndim == 2 and len(pts) > 1:
            pts = np.vstack([pts, pts[0]])
            axes[1, 2].plot(pts[:, 0], pts[:, 1], color="cyan", linewidth=1.5)
    axes[1, 2].set_title("Seed Expansion (Yellow) & Tumor Seeds (Cyan)")
    axes[1, 2].axis("off")

    # (1, 3) Final Segmentation Result
    axes[1, 3].imshow(error_map)
    axes[1, 3].set_title("Final Segmentation")
    axes[1, 3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)