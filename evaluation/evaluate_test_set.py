from pathlib import Path

import numpy as np

from config import *
from evaluation.evaluate_single_slice import eval_vol


def eval_dataset(model_name=MODEL, symmetric=False, lambda_val=LAMBDA, **eval_kwargs):
    """Evaluate one model on every held-out test volume and save per-volume scores."""
    volume_numbers = np.arange(MAX_VALIDATION_VOLUME + 1, TOTAL_VOLUMES + 1)
    dice_scores = np.zeros(len(volume_numbers), dtype=np.float64)
    iou_scores = np.zeros(len(volume_numbers), dtype=np.float64)
    total_intersection = total_union = total_pred = total_gt = 0
    missed_tumors_count = gt_tumors_count = 0

    print(
        f"Evaluating {len(volume_numbers)} test volumes: {model_name} "
        f"(symmetric={symmetric}, lambda={lambda_val:.2f})"
    )
    for index, vol_num in enumerate(volume_numbers):
        dice, iou, intersection, union, pred_size, gt_size = eval_vol(
            int(vol_num),
            symmetric=symmetric,
            lambda_val=lambda_val,
            model_name=model_name,
            **eval_kwargs,
        )
        dice_scores[index] = dice
        iou_scores[index] = iou
        total_intersection += intersection
        total_union += union
        total_pred += pred_size
        total_gt += gt_size
        if gt_size > 0:
            gt_tumors_count += 1
            if intersection == 0:
                missed_tumors_count += 1

    results = {
        "model": model_name,
        "symmetric": symmetric,
        "lambda_val": float(lambda_val),
        "volume_numbers": volume_numbers,
        "dice_per_volume": dice_scores,
        "iou_per_volume": iou_scores,
        "mean_dice": float(np.mean(dice_scores)),
        "std_dice": float(np.std(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "std_iou": float(np.std(iou_scores)),
        "global_dice": float(2.0 * total_intersection / (total_pred + total_gt)),
        "global_iou": float(total_intersection / total_union),
        "missed_tumors_count": missed_tumors_count,
        "gt_tumors_count": gt_tumors_count,
    }

    save_dir = Path(PROJECT_ROOT) / "output" / "evaluation_scores"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_dir / f"{model_name}_metrics.npz", **results)

    print(
        f"Dice: {results['mean_dice']:.4f} +/- {results['std_dice']:.4f}\n"
        f"IoU:  {results['mean_iou']:.4f} +/- {results['std_iou']:.4f}\n"
        f"Missed tumors: {missed_tumors_count}/{gt_tumors_count}\n"
    )
    return results


if __name__ == "__main__":
    eval_dataset(model_name="baseline_gmm", symmetric=False, lambda_val=0.0)
