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
    segment_with_random_walker,
    tumor_posterior,
)
from statistical_models.healthy_likelihood import healthy_gmm_joint_likelihood
from statistical_models.tumor_likelihoods import tumor_joint_likelihood
from utilities.utils import load_and_normalize_slice

RANDOM_WALKER_OPTIMIZATION_VERSION = 1
MIN_VALIDATION_PRECISION = 0.75
MAX_VALIDATION_EMPTY_FALSE_POSITIVES = 1
PARAMETER_NAMES = [
    "lambda_val",
    "tumor_prior_scale",
    "tumor_seed_threshold",
    "healthy_seed_threshold",
    "min_tumor_seed_pixels",
    "beta",
    "posterior_weight",
]


def build_validation_cache():
    cache = []
    volumes = range(MAX_TRAINING_VOLUME + 1, MAX_VALIDATION_VOLUME + 1)
    print(f"Caching {len(volumes)} validation slices...")
    for vol_num in volumes:
        image, brain_mask, gt_mask = load_and_normalize_slice(
            vol_num, SLICE_NUM, symmetric=False
        )
        cache.append({
            "volume": vol_num,
            "image": image,
            "brain_mask": brain_mask,
            "ground_truth": np.any(gt_mask > 0, axis=-1),
            "healthy_global": healthy_gmm_joint_likelihood(
                vol_num, lambda_val=0.0, symmetric=False
            ),
            "healthy_local": healthy_gmm_joint_likelihood(
                vol_num, lambda_val=1.0, symmetric=False
            ),
            "tumor": tumor_joint_likelihood(vol_num, symmetric=False),
        })
    return cache


def evaluate_parameters(cache, params):
    dice_all, dice_tumor, precision_tumor = [], [], []
    missed_tumors = false_positive_empty = 0
    for item in cache:
        healthy = (
            (1.0 - params["lambda_val"]) * item["healthy_global"]
            + params["lambda_val"] * item["healthy_local"]
        )
        posterior = tumor_posterior(
            healthy, params["tumor_prior_scale"] * item["tumor"]
        )
        result = segment_with_random_walker(
            item["image"],
            item["brain_mask"],
            posterior,
            tumor_seed_threshold=params["tumor_seed_threshold"],
            healthy_seed_threshold=params["healthy_seed_threshold"],
            min_tumor_seed_pixels=params["min_tumor_seed_pixels"],
            beta=params["beta"],
            posterior_weight=params["posterior_weight"],
        )
        metrics = calculate_metrics(result["prediction"], item["ground_truth"])
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


def suggest_parameters(trial):
    return {
        "lambda_val": trial.suggest_float("lambda_val", 0.0, 0.6, step=0.05),
        "tumor_prior_scale": trial.suggest_float(
            "tumor_prior_scale", 0.3, 3.0, log=True
        ),
        "tumor_seed_threshold": trial.suggest_float(
            "tumor_seed_threshold", 0.45, 0.85, step=0.05
        ),
        "healthy_seed_threshold": trial.suggest_float(
            "healthy_seed_threshold", 0.01, 0.20, step=0.01
        ),
        "min_tumor_seed_pixels": trial.suggest_int(
            "min_tumor_seed_pixels", 1, 15
        ),
        "beta": trial.suggest_float("beta", 20.0, 250.0, log=True),
        "posterior_weight": trial.suggest_float(
            "posterior_weight", 0.25, 3.0, step=0.25
        ),
    }


def run_random_walker_optimization(n_trials=40):
    cache = build_validation_cache()

    def objective(trial):
        metrics = evaluate_parameters(cache, suggest_parameters(trial))
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        return (
            metrics["tumor_present_dice"]
            - 0.03 * metrics["missed_tumors"]
            - 0.02 * metrics["false_positive_empty"]
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.enqueue_trial({
        "lambda_val": 0.15,
        "tumor_prior_scale": 1.0,
        "tumor_seed_threshold": 0.65,
        "healthy_seed_threshold": 0.03,
        "min_tumor_seed_pixels": 5,
        "beta": 90.0,
        "posterior_weight": 1.0,
    })
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    completed_trials = [trial for trial in study.trials if trial.value is not None]
    acceptable_trials = [
        trial for trial in completed_trials
        if trial.user_attrs["tumor_present_precision"] >= MIN_VALIDATION_PRECISION
        and trial.user_attrs["false_positive_empty"]
        <= MAX_VALIDATION_EMPTY_FALSE_POSITIVES
    ]
    candidates = acceptable_trials or completed_trials
    best_trial = max(
        candidates,
        key=lambda trial: (
            -trial.user_attrs["missed_tumors"],
            trial.user_attrs["tumor_present_dice"],
            -trial.user_attrs["false_positive_empty"],
            trial.user_attrs["tumor_present_precision"],
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
        optimization_version=RANDOM_WALKER_OPTIMIZATION_VERSION,
        validation_dice=best["all_slice_dice"],
        validation_tumor_present_dice=best["tumor_present_dice"],
        validation_precision=best["tumor_present_precision"],
        validation_missed_tumors=best["missed_tumors"],
        validation_false_positive_empty=best["false_positive_empty"],
    )
    print(
        f"Selected validation Dice={best['all_slice_dice']:.4f}, "
        f"tumor Dice={best['tumor_present_dice']:.4f}, "
        f"misses={best['missed_tumors']}, "
        f"empty FP={best['false_positive_empty']}"
    )
    if not acceptable_trials:
        print("Warning: no trial satisfied the precision/empty-slice constraints.")
    print(f"Best parameters: {best_params}")
    print(f"Saved to: {save_path}")
    return best_params
