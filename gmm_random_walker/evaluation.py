import csv
from pathlib import Path

import numpy as np

from config import MAX_VALIDATION_VOLUME, PROJECT_ROOT, TOTAL_VOLUMES
from gmm_random_walker.model import evaluate_hybrid_volume


def evaluate_hybrid_test_set(spatial_params, **hybrid_params):
    volumes = np.arange(MAX_VALIDATION_VOLUME + 1, TOTAL_VOLUMES + 1)
    records = []
    print(f"Evaluating Spatial GMM + local Random Walker on {len(volumes)} volumes...")
    for vol_num in volumes:
        details = evaluate_hybrid_volume(
            int(vol_num),
            spatial_params=spatial_params,
            return_details=True,
            **hybrid_params,
        )
        records.append({
            "volume": int(vol_num),
            "base_dice": details["base_dice"],
            **{
                name: details[name]
                for name in [
                    "dice",
                    "iou",
                    "precision",
                    "recall",
                    "intersection",
                    "union",
                    "pred_size",
                    "gt_size",
                ]
            },
        })

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
        "model": "spatial_gmm_random_walker_hybrid",
        "volume_numbers": volumes,
        "dice_per_volume": dice,
        "iou_per_volume": iou,
        "precision_per_volume": precision,
        "recall_per_volume": recall,
        "base_dice_per_volume": np.array([row["base_dice"] for row in records]),
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
        **{f"spatial_{name}": float(value) for name, value in spatial_params.items()},
        **{name: float(value) for name, value in hybrid_params.items()},
    }

    output_dir = Path(PROJECT_ROOT) / "output" / "gmm_random_walker"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "hybrid_metrics.npz", **results)
    with open(output_dir / "hybrid_per_volume.csv", "w", newline="") as stream:
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
