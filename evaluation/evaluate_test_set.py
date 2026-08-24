import numpy as np

from config import SLICE_NUM
from evaluation.evaluate_single_slice import eval_vol


def dataset_eval(
    volumes,
    slice_num=SLICE_NUM,
    healthy_model_file="healthy_gmm_all_modalities.npz",
    tumor_model_file="tumor_gmm_all_modalities.npz",
    posterior_mean_threshold=0.5,
    entropy_expansion_threshold=0.1,
    posterior_expansion_threshold=0.1,
    image_processing_params=None,
    verbose=False,
    return_details=False,
):
    """Evaluate one model and optionally return complete per-volume diagnostics."""
    volumes = np.asarray(list(volumes), dtype=int)
    image_processing_params = image_processing_params or {}
    rows = []

    for vol_num in volumes:
        result = eval_vol(
            int(vol_num),
            slice_num=slice_num,
            healthy_model_file=healthy_model_file,
            tumor_model_file=tumor_model_file,
            posterior_mean_threshold=posterior_mean_threshold,
            entropy_expansion_threshold=entropy_expansion_threshold,
            posterior_expansion_threshold=posterior_expansion_threshold,
            show_plots=False,
            return_details=True,
            **image_processing_params,
        )
        rows.append({
            "volume": int(vol_num),
            "dice": result["dice"],
            "iou": result["iou"],
            "precision": result["precision"],
            "recall": result["recall"],
            "pred_size": result["pred_size"],
            "gt_size": result["gt_size"],
            "intersection": result["intersection"],
        })

    keys = [
        "volume", "dice", "iou", "precision", "recall",
        "pred_size", "gt_size", "intersection",
    ]
    values = {
        key: np.asarray([row[key] for row in rows]) if rows else np.array([])
        for key in keys
    }
    tumor_present = values["gt_size"] > 0
    tumor_free = ~tumor_present
    missed = tumor_present & (values["intersection"] == 0)
    false_positive = tumor_free & (values["pred_size"] > 0)

    details = {
        "volume_numbers": values["volume"].astype(int),
        "dice_per_volume": values["dice"],
        "iou_per_volume": values["iou"],
        "precision_per_volume": values["precision"],
        "recall_per_volume": values["recall"],
        "pred_size_per_volume": values["pred_size"].astype(int),
        "gt_size_per_volume": values["gt_size"].astype(int),
        "tumor_present": tumor_present,
        "mean_dice": float(np.mean(values["dice"])),
        "std_dice": float(np.std(values["dice"])),
        "mean_iou": float(np.mean(values["iou"])),
        "std_iou": float(np.std(values["iou"])),
        "tumor_present_mean_dice": float(np.mean(values["dice"][tumor_present])) if np.any(tumor_present) else np.nan,
        "tumor_present_mean_precision": float(np.mean(values["precision"][tumor_present])) if np.any(tumor_present) else np.nan,
        "tumor_present_mean_recall": float(np.mean(values["recall"][tumor_present])) if np.any(tumor_present) else np.nan,
        "gt_tumors_count": int(np.sum(tumor_present)),
        "missed_tumors_count": int(np.sum(missed)),
        "missed_volume_numbers": values["volume"][missed].astype(int),
        "tumor_free_count": int(np.sum(tumor_free)),
        "tumor_free_false_positives": int(np.sum(false_positive)),
        "false_positive_volume_numbers": values["volume"][false_positive].astype(int),
    }

    if verbose:
        print(
            f"{healthy_model_file}: Dice {details['mean_dice']:.4f} +/- "
            f"{details['std_dice']:.4f}, IoU {details['mean_iou']:.4f} +/- "
            f"{details['std_iou']:.4f}"
        )
        print(
            f"  Missed tumors: {details['missed_tumors_count']}/{details['gt_tumors_count']} "
            f"- volumes {details['missed_volume_numbers'].tolist()}"
        )
        print(
            f"  Empty-slice false positives: {details['tumor_free_false_positives']}/"
            f"{details['tumor_free_count']} - volumes "
            f"{details['false_positive_volume_numbers'].tolist()}"
        )

    if return_details:
        return details
    return details["dice_per_volume"], details["iou_per_volume"]
