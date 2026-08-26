import cv2
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
    """
    Generates a 2x4 diagnostic grid.

    All displayed images are rotated 90 degrees counterclockwise.
    The raw MRI panel displays channel 0 (FLAIR), with pixels outside
    the brain mask set to black.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # -------------------------------------------------------------------------
    # Helper: rotate image 90 degrees counterclockwise
    # -------------------------------------------------------------------------
    def rotate_ccw(arr):
        return np.rot90(arr, k=1)

    brain_bin = brain_mask.astype(bool)

    # -------------------------------------------------------------------------
    # Build Composite Posterior Image
    # -------------------------------------------------------------------------
    tumor_rgb = posteriors[..., -3:]
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

    prob_composite[~brain_bin] = 0.0

    # -------------------------------------------------------------------------
    # Ground Truth -> RGB
    # -------------------------------------------------------------------------
    if gt_mask.ndim == 3 and gt_mask.shape[-1] == 3:
        gt_rgb = gt_mask.copy()
    else:
        gt_mask_2d = gt_mask[..., 0] if gt_mask.ndim > 2 else gt_mask

        color_map = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        gt_clean = np.clip(
            gt_mask_2d.astype(int),
            0,
            len(color_map) - 1,
        )

        gt_rgb = color_map[gt_clean]

    gt_rgb[~brain_bin] = 0.0

    # -------------------------------------------------------------------------
    # Final Segmentation -> RGB
    # -------------------------------------------------------------------------
    if final_segmentation.ndim == 3 and final_segmentation.shape[-1] == 3:
        seg_rgb = final_segmentation.copy()
    else:
        seg_2d = (
            final_segmentation[..., 0]
            if final_segmentation.ndim > 2
            else final_segmentation
        )

        color_map = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        seg_clean = np.clip(
            seg_2d.astype(int),
            0,
            len(color_map) - 1,
        )

        seg_rgb = color_map[seg_clean]

    seg_rgb[~brain_bin] = 0.0

    # -------------------------------------------------------------------------
    # Blob masks
    # -------------------------------------------------------------------------
    num_blobs = blob_array.shape[-1] if blob_array.ndim == 3 else 0

    tumor_mask_combined = np.zeros(brain_mask.shape, dtype=np.uint8)
    non_tumor_mask_combined = np.zeros(brain_mask.shape, dtype=np.uint8)
    seed_mask_combined = np.zeros(brain_mask.shape, dtype=bool)

    for i in range(num_blobs):
        mask_i = blob_array[:, :, i].astype(np.uint8)

        if is_tumor_list[i]:
            tumor_mask_combined |= mask_i
            seed_mask_combined |= mask_i.astype(bool)
        else:
            non_tumor_mask_combined |= mask_i

    # -------------------------------------------------------------------------
    # Expanded segmentation mask
    # -------------------------------------------------------------------------
    if final_segmentation.ndim > 2 and final_segmentation.shape[-1] == 3:
        expanded_mask = (
            np.sum(final_segmentation, axis=-1) > 0
        ) & brain_bin
    else:
        expanded_mask = (final_segmentation > 0) & brain_bin

    expansion_only_mask = expanded_mask & (~seed_mask_combined)

    # -------------------------------------------------------------------------
    # Pixel-wise TP / FN / FP Error Map
    # -------------------------------------------------------------------------
    pred_bin = (
        np.sum(final_segmentation, axis=-1) > 0
        if (
            final_segmentation.ndim > 2
            and final_segmentation.shape[-1] == 3
        )
        else (final_segmentation > 0)
    )

    gt_bin = (
        np.sum(gt_mask, axis=-1) > 0
        if gt_mask.ndim > 2
        else (gt_mask > 0)
    )

    tp = pred_bin & gt_bin & brain_bin
    fn = (~pred_bin) & gt_bin & brain_bin
    fp = pred_bin & (~gt_bin) & brain_bin

    error_map = np.zeros(
        (*brain_mask.shape, 3),
        dtype=np.float32,
    )

    error_map[tp] = [0.0, 1.0, 0.0]  # Green: TP
    error_map[fn] = [1.0, 0.0, 0.0]  # Red: FN
    error_map[fp] = [0.0, 0.0, 1.0]  # Blue: FP

    # =========================================================================
    # ROW 0
    # =========================================================================

    # -------------------------------------------------------------------------
    # (0, 0) Raw FLAIR -- channel 0
    # -------------------------------------------------------------------------
    # (0, 0) Raw FLAIR -- channel 0
    raw_im = image[..., 0].copy()

    # Mask background with NaN so it does not affect intensity scaling
    raw_im[~brain_bin] = np.nan

    # Grayscale colormap with masked/NaN pixels displayed as black
    cmap_gray = plt.cm.gray.copy()
    cmap_gray.set_bad(color="black")

    axes[0, 0].imshow(
        rotate_ccw(raw_im),
        cmap=cmap_gray,
    )

    axes[0, 0].set_title("Raw Modality (FLAIR)")
    axes[0, 0].axis("off")

    # -------------------------------------------------------------------------
    # (0, 1) Ground Truth
    # -------------------------------------------------------------------------
    axes[0, 1].imshow(
        rotate_ccw(gt_rgb)
    )
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis("off")

    # -------------------------------------------------------------------------
    # (0, 2) Posterior Map
    # -------------------------------------------------------------------------
    axes[0, 2].imshow(
        rotate_ccw(prob_composite)
    )
    axes[0, 2].set_title("Posterior Map")
    axes[0, 2].axis("off")

    # -------------------------------------------------------------------------
    # (0, 3) Entropy Map
    # -------------------------------------------------------------------------
    entropy_masked = np.copy(entropy_map)
    entropy_masked[~brain_bin] = np.nan

    cmap_entropy = plt.cm.magma.copy()
    cmap_entropy.set_bad(color="black")

    axes[0, 3].imshow(
        rotate_ccw(entropy_masked),
        cmap=cmap_entropy,
    )
    axes[0, 3].set_title("Entropy Map")
    axes[0, 3].axis("off")

    # =========================================================================
    # ROW 1
    # =========================================================================

    # -------------------------------------------------------------------------
    # (1, 0) Sobel Edge Map
    # -------------------------------------------------------------------------
    sobel_masked = np.copy(edge_map)
    sobel_masked[~brain_bin] = np.nan

    cmap_sobel = plt.cm.viridis.copy()
    cmap_sobel.set_bad(color="black")

    axes[1, 0].imshow(
        rotate_ccw(sobel_masked),
        cmap=cmap_sobel,
    )
    axes[1, 0].set_title("Sobel Edge Map")
    axes[1, 0].axis("off")

    # -------------------------------------------------------------------------
    # Rotate masks before extracting contours.
    #
    # This avoids manually transforming OpenCV contour coordinates.
    # -------------------------------------------------------------------------
    tumor_mask_rot = rotate_ccw(tumor_mask_combined)
    non_tumor_mask_rot = rotate_ccw(non_tumor_mask_combined)

    tumor_contours_rot, _ = cv2.findContours(
        tumor_mask_rot.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    non_tumor_contours_rot, _ = cv2.findContours(
        non_tumor_mask_rot.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    prob_rot = rotate_ccw(prob_composite)

    # -------------------------------------------------------------------------
    # (1, 1) Classified Contours
    # -------------------------------------------------------------------------
    axes[1, 1].imshow(prob_rot)

    for cnt in tumor_contours_rot:
        pts = cnt.squeeze(axis=1)

        if pts.ndim == 2 and len(pts) > 1:
            pts = np.vstack([pts, pts[0]])

            axes[1, 1].plot(
                pts[:, 0],
                pts[:, 1],
                color="red",
                linewidth=1.5,
            )

    for cnt in non_tumor_contours_rot:
        pts = cnt.squeeze(axis=1)

        if pts.ndim == 2 and len(pts) > 1:
            pts = np.vstack([pts, pts[0]])

            axes[1, 1].plot(
                pts[:, 0],
                pts[:, 1],
                color="lime",
                linewidth=1.5,
            )

    axes[1, 1].set_title("Classified Contours")
    axes[1, 1].axis("off")

    # -------------------------------------------------------------------------
    # (1, 2) Seed Expansion
    # -------------------------------------------------------------------------
    axes[1, 2].imshow(prob_rot)

    expansion_rot = rotate_ccw(expansion_only_mask)

    yellow_rgba = np.zeros(
        (*expansion_rot.shape, 4),
        dtype=np.float32,
    )

    yellow_rgba[expansion_rot] = [
        1.0,   # Red
        0.0,   # Green
        1.0,   # Blue
        0.65,  # 65% opacity
    ]

    axes[1, 2].imshow(yellow_rgba)

    for cnt in tumor_contours_rot:
        pts = cnt.squeeze(axis=1)

        if pts.ndim == 2 and len(pts) > 1:
            pts = np.vstack([pts, pts[0]])

            axes[1, 2].plot(
                pts[:, 0],
                pts[:, 1],
                color="red",
                linewidth=1.5,
            )

    axes[1, 2].set_title(
        "Seed Expansion (Magenta) & Tumor Seeds (Red)"
    )
    axes[1, 2].axis("off")

    # -------------------------------------------------------------------------
    # (1, 3) Final Segmentation Error Map
    # -------------------------------------------------------------------------
    axes[1, 3].imshow(
        rotate_ccw(error_map)
    )
    axes[1, 3].set_title("Final Segmentation")
    axes[1, 3].axis("off")

    # -------------------------------------------------------------------------
    # Save / Show
    # -------------------------------------------------------------------------
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=900,
        )

    plt.show()
    plt.close(fig)