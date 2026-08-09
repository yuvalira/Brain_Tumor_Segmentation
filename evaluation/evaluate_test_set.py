import numpy as np
import os
from pathlib import Path
from config import *
from evaluation.evaluate_single_slice import eval_vol


def eval_dataset(model_name=MODEL, symmetric=False):
    min_vol = MAX_VALIDATION_VOLUME + 1
    max_vol = TOTAL_VOLUMES

    num_of_volumes = max_vol - min_vol + 1
    mode_str = "symmetric" if symmetric else "regular"

    print(f"Evaluating {num_of_volumes} test volumes (Mode: {mode_str.upper()})...")

    dice_scores = np.zeros(num_of_volumes, dtype=np.float64)
    iou_scores = np.zeros(num_of_volumes, dtype=np.float64)

    total_intersection = 0
    total_union = 0
    total_pred = 0
    total_gt = 0

    missed_tumors_count = 0
    gt_tumors_count = 0

    for k in range(num_of_volumes):
        vol_num = k + min_vol

        dice, iou, pred_AND_gt, pred_OR_gt, pred, gt = eval_vol(
            vol_num, symmetric=symmetric
        )

        total_intersection += pred_AND_gt
        total_union += pred_OR_gt
        total_pred += pred
        total_gt += gt

        dice_scores[k] = dice
        iou_scores[k] = iou

        # Track false negatives (tumor present in GT, but zero intersection/prediction)
        if gt > 0:
            gt_tumors_count += 1
            if pred_AND_gt == 0:
                missed_tumors_count += 1

    # Metrics Calculation
    global_dice = (2.0 * total_intersection) / (total_pred + total_gt + 1e-12)
    global_iou = total_intersection / (total_union + 1e-12)

    mean_volumewise_dice = np.mean(dice_scores)
    mean_volumewise_iou = np.mean(iou_scores)

    # Save Results
    save_dir = Path(PROJECT_ROOT) / 'output' / 'evaluation_scores'
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{model_name}_{mode_str}_metrics_lambda_{LAMBDA}.npz"
    np.savez(
        save_path,
        model=np.array(model_name),
        symmetric=symmetric,
        global_dice=global_dice,
        global_iou=global_iou,
        mean_volumewise_dice=mean_volumewise_dice,
        mean_volumewise_iou=mean_volumewise_iou,
        dice_per_volume=dice_scores,
        iou_per_volume=iou_scores,
        missed_tumors_count=missed_tumors_count,
        gt_tumors_count=gt_tumors_count,
        total_intersection=total_intersection,
        total_union=total_union,
        total_pred=total_pred,
        total_gt=total_gt,
    )

    # Console Summary
    print("\n" + "=" * 50)
    print(f"  DATASET EVALUATION COMPLETE ({model_name.upper()} - {mode_str.upper()})")
    print("=" * 50)
    print(f"Global Dataset Dice   : {global_dice:.4f}")
    print(f"Global Dataset IoU    : {global_iou:.4f}")
    print("-" * 50)
    print(f"Mean Volumewise Dice  : {mean_volumewise_dice:.4f}")
    print(f"Mean Volumewise IoU   : {mean_volumewise_iou:.4f}")
    print("-" * 50)
    print(f"Completely Missed     : {missed_tumors_count} / {gt_tumors_count} volumes with tumors")
    print("=" * 50)


if __name__ == "__main__":
    eval_dataset(symmetric=False)
    # eval_dataset(symmetric=True)