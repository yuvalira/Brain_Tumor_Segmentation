import matplotlib.pyplot as plt
import numpy as np
from config import *
from utilities.utils import load_and_normalize_slice
from statistical_models.healthy_likelihood import healthy_gmm_joint_likelihood
from statistical_models.tumor_likelihoods import tumor_gmm_joint_likelihood

def plot_posterior_comparison(
    vol_num=350,
    slice_num=SLICE_NUM,
    modality_idx=1,
    modality_name="T1ce (Normalized)",
    save_path="posterior_rgb_comparison.png",
):
  """Plots raw MRI modality followed by 4 GMM posterior RGB configurations with GT overlay."""
  # 1. Load slice, brain mask, and ground truth
  image, brain_mask, gt_mask, _ = load_and_normalize_slice(vol_num, slice_num)

  # Create binary ground truth tumor contour mask
  binary_gt = (
      np.any(gt_mask > 0, axis=-1) if gt_mask.ndim == 3 else (gt_mask > 0)
  ) & brain_mask.astype(bool)

  # Rotate ground truth and brain mask 90 degrees CCW
  brain_mask_rot = np.rot90(brain_mask, k=1, axes=(0, 1))
  binary_gt_rot = np.rot90(binary_gt, k=1, axes=(0, 1))

  # Rotate raw modality
  raw_slice = np.where(brain_mask, image[:, :, modality_idx], np.nan)
  raw_slice_rot = np.rot90(raw_slice, k=1, axes=(0, 1))

  # 2. Define the four experiment configurations
  experiments = [
      {
          "title": "4D Raw Baseline",
          "healthy_file": "healthy_gmm_raw.npz",
          "tumor_file": "tumor_gmm_raw.npz",
      },
      {
          "title": "8D Symmetric (NDI)",
          "healthy_file": "healthy_gmm_symmetric.npz",
          "tumor_file": "tumor_gmm_symmetric.npz",
      },
      {
          "title": "5D Boundary Distance",
          "healthy_file": "healthy_gmm_boundary_distance.npz",
          "tumor_file": "tumor_gmm_boundary_distance.npz",
      },
      {
          "title": "9D Full Multimodal",
          "healthy_file": "healthy_gmm_all_modalities.npz",
          "tumor_file": "tumor_gmm_all_modalities.npz",
      },
  ]

  fig, axes = plt.subplots(1, 5, figsize=(24, 5))

  # Subplot 0: Raw modality
  cmap_raw = plt.get_cmap("gray").copy()
  cmap_raw.set_bad(color="black")
  axes[0].imshow(raw_slice_rot, cmap=cmap_raw)
  axes[0].set_title(modality_name, fontsize=12, fontweight="bold")
  axes[0].axis("off")

  # Subplots 1-4: Posterior RGB maps
  for idx, exp in enumerate(experiments):
    ax = axes[idx + 1]

    healthy_joint = healthy_gmm_joint_likelihood(
        vol_num=vol_num, filename=exp["healthy_file"], slice_num=slice_num
    )
    tumor_joint = tumor_gmm_joint_likelihood(
        vol_num=vol_num, filename=exp["tumor_file"], slice_num=slice_num
    )

    joint_stack = np.dstack([healthy_joint, tumor_joint])
    evidence = np.sum(joint_stack, axis=-1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
      posterior = np.where(
          evidence > 1e-12, joint_stack / (evidence + 1e-12), 0.0
      )

    tumor_rgb = posterior[:, :, 1:4]
    tumor_rgb = np.clip(tumor_rgb * brain_mask[:, :, np.newaxis], 0.0, 1.0)
    tumor_rgb_rot = np.rot90(tumor_rgb, k=1, axes=(0, 1))

    ax.imshow(tumor_rgb_rot)
    ax.set_title(exp["title"], fontsize=12, fontweight="bold")
    ax.axis("off")

  # Overlay ground truth contour across all subplots
  if np.any(binary_gt_rot):
    for ax in axes:
      ax.contour(
          binary_gt_rot,
          levels=[0.5],
          colors=["magenta"],
          linewidths=1.5,
          linestyles="solid",
      )

  plt.suptitle(
      f"Volume {vol_num}, Slice {slice_num} — Posterior Tumor Probability RGB"
      " (R: NCR/NET, G: ED, B: ET | Contour: Ground Truth)",
      fontsize=14,
      fontweight="bold",
      y=1.02,
  )
  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=900, bbox_inches="tight")
  plt.show()