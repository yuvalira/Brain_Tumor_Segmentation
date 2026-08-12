import csv
from pathlib import Path

import numpy as np

from config import MAX_VALIDATION_VOLUME, PROJECT_ROOT, TOTAL_VOLUMES
from gmm_random_walker.model import evaluate_random_walker_volume


def evaluate_random_walker_test_set(**params):
    volumes = np.arange(MAX_VALIDATION_VOLUME + 1, TOTAL_VOLUMES + 1)
    records = []
    print(f"Evaluating GMM + Random Walker on {len(volumes)} test volumes...")
    for vol_num in volumes:
        metrics = evaluate_random_walker_volume(int(vol_num), **params)
        records.append({"volume": int(vol_num), **metrics})

    tumor_present = np.array([row["gt_size"] > 0 for row in records])
    intersections = np.array([row["intersection"] for row in records])
    pred_sizes = np.array([row["pred_size"] for row in records])
    dice = np.array([row["dice"] for row in records])
    iou = np.array([row["iou"] for row in records])
    precision = np.array([row["precision"] for row in records])
    recall = np.array([row["recall"] for row in records])
    missed = tumor_present & (intersections == 0)
    empty_fp = (~tumor_present) & (pred_sizes > 0)

    results = {
        "model": "gmm_random_walker",
        "volume_numbers": volumes,
        "dice_per_volume": dice,
        "iou_per_volume": iou,
        "precision_per_volume": precision,
        "recall_per_volume": recall,
        "gt_size_per_volume": np.array([row["gt_size"] for row in records]),
        "pred_size_per_volume": pred_sizes,
        "tumor_present": tumor_present,
        "mean_dice": float(np.mean(dice)),
        "std_dice": float(np.std(dice)),
        "mean_iou": float(np.mean(iou)),
        "std_iou": float(np.std(iou)),
        "tumor_present_mean_dice": float(np.mean(dice[tumor_present])),
        "tumor_present_mean_precision": float(np.mean(precision[tumor_present])),
        "tumor_present_mean_recall": float(np.mean(recall[tumor_present])),
        "missed_tumors_count": int(np.sum(missed)),
        "gt_tumors_count": int(np.sum(tumor_present)),
        "missed_volume_numbers": volumes[missed],
        "tumor_free_false_positives": int(np.sum(empty_fp)),
        **{name: float(value) for name, value in params.items()},
    }

    output_dir = Path(PROJECT_ROOT) / "output" / "gmm_random_walker"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "metrics.npz", **results)
    with open(output_dir / "per_volume.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(
        f"All slices Dice:      {results['mean_dice']:.4f} +/- {results['std_dice']:.4f}\n"
        f"Tumor-present Dice:   {results['tumor_present_mean_dice']:.4f}\n"
        f"Tumor-present recall: {results['tumor_present_mean_recall']:.4f}\n"
        f"Missed tumors:        {results['missed_tumors_count']}/{results['gt_tumors_count']} "
        f"{results['missed_volume_numbers'].tolist()}\n"
        f"Tumor-free FP:        {results['tumor_free_false_positives']}"
    )
    return results
