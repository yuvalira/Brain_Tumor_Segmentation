from pathlib import Path

import numpy as np
import optuna

from config import (
    MAX_TRAINING_VOLUME,
    MAX_VALIDATION_VOLUME,
    PROJECT_ROOT,
    RANDOM_SEED,
    SLICE_NUM,
)
from gmm_random_walker.model import (
    calculate_metrics,
    segment_with_hybrid_random_walker,
    spatial_gmm_segmentation,
    tumor_posterior,
)
from statistical_models.healthy_likelihood import healthy_gmm_joint_likelihood
from statistical_models.tumor_likelihoods import tumor_joint_likelihood
from utilities.utils import load_and_normalize_slice

HYBRID_OPTIMIZATION_VERSION = 2
SPATIAL_PARAMETER_NAMES = [
    "lambda_val",
    "tumor_prior_scale",
    "min_pixels_per_blob",
    "binarization_factor",
    "blob_class_threshold",
    "large_contour_min_area",
    "top_posterior_mean_threshold",
    "high_posterior_fraction_threshold",
    "entropy_thresh",
    "posterior_min",
    "max_expansion_diameter",
]
HYBRID_PARAMETER_NAMES = [
    "rescue_seed_threshold",
    "healthy_seed_threshold",
    "min_rescue_seed_pixels",
    "roi_dilation",
    "base_seed_erosion",
    "beta",
    "posterior_weight",
    "min_added_component_mean_posterior",
]


def load_spatial_parameters():
    path = Path(PROJECT_ROOT) / "saved_parameters" / "spatial_gmm_best_params.npz"
    if not path.exists():
        raise FileNotFoundError("Run main.ipynb through Spatial-GMM tuning first.")
    with np.load(path) as saved:
        missing = [name for name in SPATIAL_PARAMETER_NAMES if name not in saved.files]
        if missing:
            raise ValueError(f"Spatial parameter file is missing: {missing}")
        return {name: saved[name].item() for name in SPATIAL_PARAMETER_NAMES}


def build_validation_cache(spatial_params):
    cache = []
    volumes = range(MAX_TRAINING_VOLUME + 1, MAX_VALIDATION_VOLUME + 1)
    print(f"Caching {len(volumes)} validation slices and Spatial-GMM masks...")
    for vol_num in volumes:
        image, brain_mask, gt_mask = load_and_normalize_slice(
            vol_num, SLICE_NUM, symmetric=False
        )
        healthy = healthy_gmm_joint_likelihood(
            vol_num, lambda_val=spatial_params["lambda_val"], symmetric=False
        )
        tumor = spatial_params["tumor_prior_scale"] * tumor_joint_likelihood(
            vol_num, symmetric=False
        )
        cache.append({
            "volume": vol_num,
            "image": image,
            "brain_mask": brain_mask,
            "ground_truth": np.any(gt_mask > 0, axis=-1),
            "posterior": tumor_posterior(healthy, tumor),
            "base_prediction": spatial_gmm_segmentation(
                healthy, tumor, brain_mask, spatial_params
            ),
        })
    return cache


def summarize_predictions(cache, prediction_key=None, hybrid_params=None):
    dice_all, dice_tumor, precision_tumor = [], [], []
    missed_tumors = false_positive_empty = 0
    for item in cache:
        if prediction_key:
            prediction = item[prediction_key]
        else:
            prediction = segment_with_hybrid_random_walker(
                item["image"],
                item["brain_mask"],
                item["posterior"],
                item["base_prediction"],
                **hybrid_params,
            )["prediction"]
        metrics = calculate_metrics(prediction, item["ground_truth"])
        dice_all.append(metrics["dice"])
        if metrics["gt_size"]:
            dice_tumor.append(metrics["dice"])
            precision_tumor.append(metrics["precision"])
            missed_tumors += metrics["intersection"] == 0
        else:
            false_positive_empty += metrics["pred_size"] > 0
    return {
        "all_slice_dice": float(np.mean(dice_all)),
        "tumor_present_dice": float(np.mean(dice_tumor)),
        "tumor_present_precision": float(np.mean(precision_tumor)),
        "missed_tumors": int(missed_tumors),
        "false_positive_empty": int(false_positive_empty),
    }


def suggest_hybrid_parameters(trial):
    return {
        "rescue_seed_threshold": trial.suggest_float(
            "rescue_seed_threshold", 0.50, 0.85, step=0.05
        ),
        "healthy_seed_threshold": trial.suggest_float(
            "healthy_seed_threshold", 0.01, 0.20, step=0.01
        ),
        "min_rescue_seed_pixels": trial.suggest_int(
            "min_rescue_seed_pixels", 1, 15
        ),
        "roi_dilation": trial.suggest_int("roi_dilation", 0, 18, step=2),
        "base_seed_erosion": trial.suggest_int("base_seed_erosion", 0, 3),
        "beta": trial.suggest_float("beta", 20.0, 200.0, log=True),
        "posterior_weight": trial.suggest_float(
            "posterior_weight", 0.25, 2.5, step=0.25
        ),
        "min_added_component_mean_posterior": trial.suggest_float(
            "min_added_component_mean_posterior", 0.02, 0.20, step=0.02
        ),
    }


def run_hybrid_optimization(spatial_params, n_trials=30):
    cache = build_validation_cache(spatial_params)
    reference = summarize_predictions(cache, prediction_key="base_prediction")

    def objective(trial):
        metrics = summarize_predictions(
            cache, hybrid_params=suggest_hybrid_parameters(trial)
        )
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        return (
            metrics["tumor_present_dice"]
            - 0.05 * max(0, metrics["missed_tumors"] - reference["missed_tumors"])
            - 0.03 * max(
                0,
                metrics["false_positive_empty"]
                - reference["false_positive_empty"],
            )
        )

    def report_trial(study, trial):
        if trial.value is None:
            return
        metrics = trial.user_attrs
        print(
            f"Trial {trial.number}: tumor Dice={metrics['tumor_present_dice']:.4f}, "
            f"all Dice={metrics['all_slice_dice']:.4f}, "
            f"misses={metrics['missed_tumors']}, "
            f"empty FP={metrics['false_positive_empty']}"
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.enqueue_trial({
        "rescue_seed_threshold": 0.65,
        "healthy_seed_threshold": 0.05,
        "min_rescue_seed_pixels": 5,
        "roi_dilation": 0,
        "base_seed_erosion": 2,
        "beta": 90.0,
        "posterior_weight": 1.0,
        "min_added_component_mean_posterior": 0.08,
    })
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[report_trial],
        show_progress_bar=True,
    )

    completed = [trial for trial in study.trials if trial.value is not None]
    acceptable = [
        trial for trial in completed
        if trial.user_attrs["all_slice_dice"] >= reference["all_slice_dice"]
        and trial.user_attrs["missed_tumors"] <= reference["missed_tumors"]
        and trial.user_attrs["false_positive_empty"]
        <= reference["false_positive_empty"]
        and trial.user_attrs["tumor_present_precision"]
        >= reference["tumor_present_precision"] - 0.01
    ]
    best_trial = max(
        acceptable or completed,
        key=lambda trial: (
            trial.user_attrs["tumor_present_dice"],
            trial.user_attrs["all_slice_dice"],
            -trial.user_attrs["missed_tumors"],
        ),
    )
    best_params = best_trial.params
    best = best_trial.user_attrs
    save_path = (
        Path(PROJECT_ROOT)
        / "saved_parameters"
        / "gmm_random_walker_best_params.npz"
    )
    np.savez(
        save_path,
        **best_params,
        optimization_version=HYBRID_OPTIMIZATION_VERSION,
        reference_validation_dice=reference["all_slice_dice"],
        reference_validation_tumor_dice=reference["tumor_present_dice"],
        reference_validation_missed_tumors=reference["missed_tumors"],
        reference_validation_false_positive_empty=reference[
            "false_positive_empty"
        ],
        validation_dice=best["all_slice_dice"],
        validation_tumor_present_dice=best["tumor_present_dice"],
        validation_precision=best["tumor_present_precision"],
        validation_missed_tumors=best["missed_tumors"],
        validation_false_positive_empty=best["false_positive_empty"],
    )
    print(
        f"Spatial reference: Dice={reference['all_slice_dice']:.4f}, "
        f"tumor Dice={reference['tumor_present_dice']:.4f}, "
        f"misses={reference['missed_tumors']}, "
        f"empty FP={reference['false_positive_empty']}"
    )
    print(
        f"Selected hybrid:  Dice={best['all_slice_dice']:.4f}, "
        f"tumor Dice={best['tumor_present_dice']:.4f}, "
        f"misses={best['missed_tumors']}, "
        f"empty FP={best['false_positive_empty']}"
    )
    if not acceptable:
        print("Warning: no hybrid trial improved the constrained Spatial reference.")
    print(f"Best hybrid parameters: {best_params}")
    print(f"Saved to: {save_path}")
    return best_params
