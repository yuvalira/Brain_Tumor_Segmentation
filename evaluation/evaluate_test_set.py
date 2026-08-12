import csv
from pathlib import Path

import numpy as np

from config import *
from evaluation.evaluate_single_slice import eval_vol


def eval_dataset(
    model_name=MODEL,
    symmetric=False,
    lambda_val=LAMBDA,
    tumor_prior_scale=1.0,
    z_strength=0.0,
    z_posterior_gate=0.0,
    ndi_strength=0.0,
    ndi_percentile=90.0,
    ndi_posterior_gate=0.0,
    **eval_kwargs,
):
    """Evaluate one model on every held-out test volume and save diagnostics."""
    volume_numbers = np.arange(MAX_VALIDATION_VOLUME + 1, TOTAL_VOLUMES + 1)
    metric_names = ["dice", "iou", "precision", "recall", "pred_size", "gt_size"]
    per_volume = {name: np.zeros(len(volume_numbers), dtype=np.float64) for name in metric_names}
    intersections = np.zeros(len(volume_numbers), dtype=np.int64)
    unions = np.zeros(len(volume_numbers), dtype=np.int64)

    print(
        f"Evaluating {len(volume_numbers)} test volumes: {model_name} "
        f"(symmetric={symmetric}, lambda={lambda_val:.2f}, "
        f"tumor scale={tumor_prior_scale:.2f}, z strength={z_strength:.2f})"
    )
    for index, vol_num in enumerate(volume_numbers):
        details = eval_vol(
            int(vol_num),
            symmetric=symmetric,
            lambda_val=lambda_val,
            tumor_prior_scale=tumor_prior_scale,
            z_strength=z_strength,
            z_posterior_gate=z_posterior_gate,
            ndi_strength=ndi_strength,
            ndi_percentile=ndi_percentile,
            ndi_posterior_gate=ndi_posterior_gate,
            model_name=model_name,
            return_details=True,
            **eval_kwargs,
        )
        for name in metric_names:
            per_volume[name][index] = details[name]
        intersections[index] = details["intersection"]
        unions[index] = details["union"]

    tumor_present = per_volume["gt_size"] > 0
    missed_tumors = tumor_present & (intersections == 0)
    tumor_free = ~tumor_present
    false_positive_empty = tumor_free & (per_volume["pred_size"] > 0)
    total_pred = np.sum(per_volume["pred_size"])
    total_gt = np.sum(per_volume["gt_size"])
    total_intersection = np.sum(intersections)
    total_union = np.sum(unions)

    results = {
        "model": model_name,
        "symmetric": symmetric,
        "lambda_val": float(lambda_val),
        "tumor_prior_scale": float(tumor_prior_scale),
        "z_strength": float(z_strength),
        "z_posterior_gate": float(z_posterior_gate),
        "ndi_strength": float(ndi_strength),
        "ndi_percentile": float(ndi_percentile),
        "ndi_posterior_gate": float(ndi_posterior_gate),
        "volume_numbers": volume_numbers,
        "dice_per_volume": per_volume["dice"],
        "iou_per_volume": per_volume["iou"],
        "precision_per_volume": per_volume["precision"],
        "recall_per_volume": per_volume["recall"],
        "pred_size_per_volume": per_volume["pred_size"],
        "gt_size_per_volume": per_volume["gt_size"],
        "tumor_present": tumor_present,
        "mean_dice": float(np.mean(per_volume["dice"])),
        "std_dice": float(np.std(per_volume["dice"])),
        "mean_iou": float(np.mean(per_volume["iou"])),
        "std_iou": float(np.std(per_volume["iou"])),
        "tumor_present_mean_dice": float(np.mean(per_volume["dice"][tumor_present])),
        "tumor_present_std_dice": float(np.std(per_volume["dice"][tumor_present])),
        "tumor_present_mean_iou": float(np.mean(per_volume["iou"][tumor_present])),
        "tumor_present_mean_precision": float(np.mean(per_volume["precision"][tumor_present])),
        "tumor_present_mean_recall": float(np.mean(per_volume["recall"][tumor_present])),
        "global_dice": float(2.0 * total_intersection / (total_pred + total_gt)),
        "global_iou": float(total_intersection / total_union),
        "missed_tumors_count": int(np.sum(missed_tumors)),
        "gt_tumors_count": int(np.sum(tumor_present)),
        "missed_volume_numbers": volume_numbers[missed_tumors],
        "tumor_free_false_positives": int(np.sum(false_positive_empty)),
    }

    save_dir = Path(PROJECT_ROOT) / "output" / "evaluation_scores"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_dir / f"{model_name}_metrics.npz", **results)
    with open(save_dir / f"{model_name}_per_volume.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["volume", "dice", "iou", "precision", "recall", "gt_size", "pred_size", "tumor_present", "missed"]
        )
        for index, vol_num in enumerate(volume_numbers):
            writer.writerow(
                [
                    int(vol_num),
                    per_volume["dice"][index],
                    per_volume["iou"][index],
                    per_volume["precision"][index],
                    per_volume["recall"][index],
                    int(per_volume["gt_size"][index]),
                    int(per_volume["pred_size"][index]),
                    bool(tumor_present[index]),
                    bool(missed_tumors[index]),
                ]
            )

    print(
        f"All slices Dice:      {results['mean_dice']:.4f} +/- {results['std_dice']:.4f}\n"
        f"Tumor-present Dice:   {results['tumor_present_mean_dice']:.4f} +/- "
        f"{results['tumor_present_std_dice']:.4f}\n"
        f"Tumor-present recall: {results['tumor_present_mean_recall']:.4f}\n"
        f"Missed tumors:        {results['missed_tumors_count']}/{results['gt_tumors_count']} "
        f"{results['missed_volume_numbers'].tolist()}\n"
        f"Tumor-free false positives: {results['tumor_free_false_positives']}/{int(np.sum(tumor_free))}\n"
    )
    return results


if __name__ == "__main__":
    eval_dataset(model_name="baseline_gmm", symmetric=False, lambda_val=0.0)
