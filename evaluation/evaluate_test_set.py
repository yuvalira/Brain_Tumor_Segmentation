from evaluation.evaluate_single_slice import eval_vol
from config import *
import numpy as np

def dataset_eval(
    volumes,
    slice_num,
    healthy_model_file,
    tumor_model_file,
    posterior_mean_threshold,
    entropy_expansion_threshold,
    posterior_expansion_threshold,
    verbose = False
):

  num_volumes = len(volumes)

  dice_scores = np.zeros(num_volumes)
  iou_scores = np.zeros(num_volumes)
  zero_score_cases = []

  for idx, vol_num in enumerate(volumes):
    results = eval_vol(
        vol_num,
        slice_num=slice_num,
        healthy_model_file=healthy_model_file,
        tumor_model_file=tumor_model_file,
        posterior_mean_threshold=posterior_mean_threshold,
        entropy_expansion_threshold=entropy_expansion_threshold,
        posterior_expansion_threshold=posterior_expansion_threshold,
        show_plots=False,
    )

    pred_bin = results["final_segmentation"].astype(bool)
    gt_raw = results["gt_mask"]

    # Extract binary ground truth mask
    if gt_raw.ndim == 3:
      gt_bin = np.any(gt_raw > 0, axis=-1)
    else:
      gt_bin = gt_raw > 0

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total_pred = pred_bin.sum()
    total_gt = gt_bin.sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    # Edge cases: True Negatives (No tumor present in ground truth)
    if total_gt == 0 and total_pred == 0:
      dice_scores[idx] = 1.0
      iou_scores[idx] = 1.0
    elif total_gt == 0 and total_pred > 0:
      dice_scores[idx] = 0.0
      iou_scores[idx] = 0.0
    else:
      dice_scores[idx] = (2.0 * intersection) / (total_pred + total_gt + 1e-12)
      iou_scores[idx] = intersection / (union + 1e-12)

    # Track indices with a Dice or IoU score of 0
    if dice_scores[idx] == 0.0 or iou_scores[idx] == 0.0:
      zero_score_cases.append(
          (idx, vol_num, int(total_pred), int(total_gt), int(intersection))
      )

  if verbose:

    print(f"Evaluated [{healthy_model_file}] -> Mean Dice: {np.mean(dice_scores):.4f} | Mean IoU: {np.mean(iou_scores):.4f}")

    if zero_score_cases:
        print(f"  Volumes with Dice == 0 or IoU == 0 ({len(zero_score_cases)} total):")
        for idx, v_num, pred_count, gt_count, inter_count in zero_score_cases:
            reason = "False Positive (GT=0)" if gt_count == 0 else ("False Negative (Pred=0)" if pred_count == 0 else "Zero Overlap")
            print(f"    - Index: {idx:2d} | Volume: {v_num:3d} | Reason: {reason:<24} | Pred Pixels: {pred_count:5d} | GT Pixels: {gt_count:5d}")
    else:
        print("  No volumes scored 0.")

  return dice_scores, iou_scores