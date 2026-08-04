import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def visualize_probability(slice_im, posteriors, brain_mask, gt_mask, save_path=None):
    """
    Visualizes T1ce slice (background masked to black), probability maps, 
    and ground truth tumor segmentation mask built from 3 separate RGB channels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ==============================================================================
    # SUBPLOT 1: T1ce Modality (Masked outside brain)
    # ==============================================================================
    t1ce_channel = 1 if slice_im.shape[-1] > 1 else 0
    t1ce_im = slice_im[..., t1ce_channel].copy()
    t1ce_im[brain_mask == 0] = 0.0

    axes[0].imshow(t1ce_im, cmap="gray")
    axes[0].set_title("T1ce")
    axes[0].axis("off")

    # ==============================================================================
    # SUBPLOT 2: Probabilities Composite
    # ==============================================================================
    tumor_rgb = posteriors[..., -3:]  # Shape: (H, W, 3)

    num_healthy = posteriors.shape[-1] - 3
    if num_healthy > 0:
        grayscale_weights = np.linspace(0.2, 0.8, num_healthy)
        healthy_gray = np.sum(
            posteriors[..., :num_healthy] * grayscale_weights, axis=-1, keepdims=True
        )
        prob_composite = np.clip(healthy_gray + tumor_rgb, 0.0, 1.0)
    else:
        prob_composite = np.clip(tumor_rgb, 0.0, 1.0)

    prob_composite[brain_mask == 0] = 0.0

    axes[1].imshow(prob_composite)
    axes[1].set_title("Probabilities")
    axes[1].axis("off")

    # ==============================================================================
    # SUBPLOT 3: Ground Truth Tumor Mask (3 Channels: R, G, B)
    # ==============================================================================
   
    axes[2].imshow(gt_mask)
    axes[2].set_title("Ground Truth Tumor Mask")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved successfully to: {save_path}")

    plt.close(fig)


def visualize_entropy(entropy_map, brain_mask, save_path=None):

    fig, ax = plt.subplots(figsize=(6, 6))

    # Mask background pixels outside brain to NaN so imshow background renders black
    entropy_masked = np.copy(entropy_map)
    entropy_masked[brain_mask == 0] = np.nan

    # Set black background for NaN values
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="black")

    im = ax.imshow(entropy_masked, cmap=cmap)
    ax.set_title("Posterior Entropy Map")
    ax.axis("off")

    # Add colorbar for scale
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Entropy (bits)", rotation=270, labelpad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Entropy map saved to: {save_path}")

    plt.close(fig)


def visualize_sobel_edges(sobel_map, brain_mask, save_path=None):
    """
    Visualizes and saves the 2D Sobel edge detection map.
    Background outside brain_mask is set to black.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Mask background pixels outside brain
    sobel_masked = np.copy(sobel_map)
    sobel_masked[brain_mask == 0] = np.nan

    # Set black background for NaN values
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="black")

    im = ax.imshow(sobel_masked, cmap=cmap)
    ax.set_title("Sobel Edge Map")
    ax.axis("off")

    # Add colorbar for scale
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Gradient Magnitude", rotation=270, labelpad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Sobel map saved to: {save_path}")

    plt.close(fig)



def visualize_contours(
    slice_im,
    posteriors,
    sobel_map,
    brain_mask,
    gt_mask,
    blob_array,
    is_tumor_list,
    save_path=None,
):
    """
    Visualizes a 1x4 subplot grid (T1ce, Sobel, Posteriors, Ground Truth)
    with classified contours overlaid (Red = Tumor, Green = Non-tumor).
    
    Parameters:
    - slice_im: (H, W, 4) MRI slice array
    - posteriors: (H, W, C) array of posterior probabilities
    - sobel_map: (H, W) edge detection map
    - brain_mask: (H, W) binary mask for the brain area
    - gt_mask: (H, W) ground truth mask
    - blob_array: (H, W, N_blobs) boolean array of input detected blobs
    - is_tumor_list: list of length N_blobs (True for tumor, False for non-tumor)
    - save_path: str, optional output file path
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # ==============================================================================
    # 1. Prepare Base Images
    # ==============================================================================
    height, width = brain_mask.shape

    # T1ce Base Image
    t1ce_channel = 1 if slice_im.shape[-1] > 1 else 0
    t1ce_base = slice_im[..., t1ce_channel].copy()
    t1ce_base[brain_mask == 0] = 0.0

    # Sobel Base Image
    sobel_base = sobel_map.copy()
    sobel_base[brain_mask == 0] = 0.0

    # Probabilities Base Composite (RGB for last 3 tumor classes, Grayscale for rest)
    tumor_rgb = posteriors[..., -3:]
    num_healthy = posteriors.shape[-1] - 3
    if num_healthy > 0:
        grayscale_weights = np.linspace(0.2, 0.8, num_healthy)
        healthy_gray = np.sum(
            posteriors[..., :num_healthy] * grayscale_weights, axis=-1, keepdims=True
        )
        prob_base = np.clip(healthy_gray + tumor_rgb, 0.0, 1.0)
    else:
        prob_base = np.clip(tumor_rgb, 0.0, 1.0)
    prob_base[brain_mask == 0] = 0.0

    # Ground Truth Base
    gt_base = gt_mask.copy()

    # ==============================================================================
    # 2. Extract and Draw Contours
    # ==============================================================================
    num_blobs = blob_array.shape[-1] if blob_array.ndim == 3 else 0

    tumor_contours = []
    non_tumor_contours = []

    for i in range(num_blobs):
        mask_i = blob_array[:, :, i].astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if is_tumor_list[i]:
            tumor_contours.extend(contours)
        else:
            non_tumor_contours.extend(contours)

    # Function to overlay extracted contours onto a Matplotlib axis
    def overlay_contours(ax):
        for cnt in tumor_contours:
            # OpenCV contours are (x, y); plot as (x, y) coordinates
            pts = cnt.squeeze(axis=1)
            if pts.ndim == 2 and len(pts) > 1:
                # Close the polygon loop for display
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], color="red", linewidth=1.5)

        for cnt in non_tumor_contours:
            pts = cnt.squeeze(axis=1)
            if pts.ndim == 2 and len(pts) > 1:
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], color="lime", linewidth=1.5)

    # ==============================================================================
    # 3. Render Subplots
    # ==============================================================================
    # Subplot 1: T1ce
    axes[0].imshow(t1ce_base, cmap="gray")
    axes[0].set_title("T1ce")
    axes[0].axis("off")
    overlay_contours(axes[0])

    # Subplot 2: Sobel
    axes[1].imshow(sobel_base, cmap="viridis")
    axes[1].set_title("Sobel")
    axes[1].axis("off")
    overlay_contours(axes[1])

    # Subplot 3: Posteriors
    axes[2].imshow(prob_base)
    axes[2].set_title("Posteriors")
    axes[2].axis("off")
    overlay_contours(axes[2])

    # Subplot 4: Ground Truth
    axes[3].imshow(gt_base)
    axes[3].set_title("Ground Truth Tumor Mask")
    axes[3].axis("off")
    overlay_contours(axes[3])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Contour visualization saved to: {save_path}")

    plt.close(fig)



def visualize_expansion(
    total_segmentation_mask,
    slice_im,
    posteriors,
    brain_mask,
    gt_mask,
    blob_array,
    is_tumor_list,
    save_path=None,
):
    """
    Visualizes T1ce, Probabilities, and GT Mask with overlays:
    - Yellow: Initial classified tumor seeds
    - Magenta: Expanded final segmentation mask
    - Cyan: Merged Ground Truth tumor outer boundary
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # ==============================================================================
    # 1. Prepare Base Images
    # ==============================================================================
    # T1ce Base Image
    t1ce_channel = 1 if slice_im.shape[-1] > 1 else 0
    t1ce_base = slice_im[..., t1ce_channel].copy()
    t1ce_base[brain_mask == 0] = 0.0

    # Probabilities Base Composite
    tumor_rgb = posteriors[..., -3:]
    num_healthy = posteriors.shape[-1] - 3
    if num_healthy > 0:
        grayscale_weights = np.linspace(0.2, 0.8, num_healthy)
        healthy_gray = np.sum(
            posteriors[..., :num_healthy] * grayscale_weights, axis=-1, keepdims=True
        )
        prob_base = np.clip(healthy_gray + tumor_rgb, 0.0, 1.0)
    else:
        prob_base = np.clip(tumor_rgb, 0.0, 1.0)
    prob_base[brain_mask == 0] = 0.0

    # Ground Truth Base Image (Render RGB if 2D label map, else direct display)
    if gt_mask.ndim == 3 and gt_mask.shape[-1] == 3:
        gt_base = gt_mask.copy()
        # Merge 3 binary channels along axis=-1 for boundary extraction
        gt_binary = (np.sum(gt_mask, axis=-1) > 0).astype(np.uint8)
    else:
        gt_mask_2d = gt_mask[..., 0] if gt_mask.ndim > 2 else gt_mask
        
        # Color palette for GT (0: Black, 1: Red, 2: Green, 3: Blue)
        color_map = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        gt_clean = np.clip(gt_mask_2d.astype(int), 0, len(color_map) - 1)
        gt_base = color_map[gt_clean]
        
        # Merged binary GT mask (any tumor class > 0)
        gt_binary = (gt_mask_2d > 0).astype(np.uint8)

    gt_base[brain_mask == 0] = 0.0

    # ==============================================================================
    # 2. Extract Contours
    # ==============================================================================
    # A. Initial Seed Tumor Blobs (Yellow)
    seed_contours = []
    num_blobs = blob_array.shape[-1] if blob_array.ndim == 3 else 0
    for i in range(num_blobs):
        if is_tumor_list[i]:
            mask_i = blob_array[:, :, i].astype(np.uint8)
            cnts, _ = cv2.findContours(
                mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            seed_contours.extend(cnts)

    # B. Post-Expansion Final Mask (Magenta)
    expansion_binary = (total_segmentation_mask > 0).astype(np.uint8)
    expansion_contours, _ = cv2.findContours(
        expansion_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # C. Merged Ground Truth Boundary (Cyan)
    gt_contours, _ = cv2.findContours(
        gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ==============================================================================
    # 3. Helper to Plot Overlay Lines
    # ==============================================================================
    def draw_all_overlays(ax):
        # Initial Seeds -> Yellow
        for cnt in seed_contours:
            pts = cnt.squeeze(axis=1)
            if pts.ndim == 2 and len(pts) > 1:
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], color="yellow", linewidth=1.5)

        # Expanded Mask -> Magenta
        for cnt in expansion_contours:
            pts = cnt.squeeze(axis=1)
            if pts.ndim == 2 and len(pts) > 1:
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], color="magenta", linewidth=1.5)

        # Merged GT Boundary -> Cyan
        for cnt in gt_contours:
            pts = cnt.squeeze(axis=1)
            if pts.ndim == 2 and len(pts) > 1:
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], color="cyan", linewidth=1.5)

    # ==============================================================================
    # 4. Render Subplots
    # ==============================================================================
    axes[0].imshow(t1ce_base, cmap="gray")
    axes[0].set_title("T1ce")
    axes[0].axis("off")
    draw_all_overlays(axes[0])

    axes[1].imshow(prob_base)
    axes[1].set_title("Probability Map")
    axes[1].axis("off")
    draw_all_overlays(axes[1])

    axes[2].imshow(gt_base)
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis("off")
    draw_all_overlays(axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Expansion visualization saved to: {save_path}")

    plt.close(fig)


def visualize_segmentation(
    total_segmentation_mask, gt_mask, brain_mask, save_path=None
):
    """
    Generates a pixel-wise error map overlay comparing total_segmentation_mask to gt_mask.
    
    Color mapping:
    - True Positives (TP): Green
    - False Negatives (FN): Red
    - False Positives (FP): Blue
    - True Negatives (TN): Gray
    - Outside Brain Mask: Black
    """
    # 1. Ensure 2D binary masks (H, W) for comparison
    pred_bin = (total_segmentation_mask > 0)
    
    if gt_mask.ndim > 2:
        gt_bin = (np.sum(gt_mask, axis=-1) > 0)
    else:
        gt_bin = (gt_mask > 0)

    brain_bin = brain_mask.astype(bool)

    # 2. Compute confusion matrix categories
    tp = pred_bin & gt_bin & brain_bin
    fn = (~pred_bin) & gt_bin & brain_bin
    fp = pred_bin & (~gt_bin) & brain_bin
    tn = (~pred_bin) & (~gt_bin) & brain_bin

    # 3. Build RGB image
    height, width = brain_mask.shape
    rgb_map = np.zeros((height, width, 3), dtype=np.float32)

    # True Positives -> Green (0, 1, 0)
    rgb_map[tp] = [0.0, 1.0, 0.0]

    # False Negatives -> Red (1, 0, 0)
    rgb_map[fn] = [1.0, 0.0, 0.0]

    # False Positives -> Blue (0, 0, 1)
    rgb_map[fp] = [0.0, 0.0, 1.0]

    # True Negatives -> Gray (0.5, 0.5, 0.5)
    rgb_map[tn] = [0.5, 0.5, 0.5]

    # Outside Brain Mask -> Black (0, 0, 0)
    rgb_map[~brain_bin] = [0.0, 0.0, 0.0]

    # 4. Render Figure
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb_map)
    ax.set_title("Segmentation Error Map (TP: Green, FN: Red, FP: Blue, TN: Gray)")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Segmentation visualization saved to: {save_path}")

    plt.close(fig)