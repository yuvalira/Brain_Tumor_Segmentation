import os

import numpy as np
import optuna

from config import *
from evaluation.evaluate_single_slice import (
    apply_ndi_fusion,
    calculate_metrics,
    segment_likelihoods,
)
from statistical_models.healthy_likelihood import healthy_gmm_joint_likelihood
from statistical_models.tumor_likelihoods import tumor_joint_likelihood
from utilities.utils import load_and_normalize_slice

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_validation_cache(symmetric=False):
    """Precompute likelihood maps once so Optuna trials only rerun post-processing."""
    cache = []
    validation_volumes = range(MAX_TRAINING_VOLUME + 1, MAX_VALIDATION_VOLUME + 1)
    print(f"Caching {len(validation_volumes)} validation slices (symmetric={symmetric})...")
    for vol_num in validation_volumes:
        slice_output = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=symmetric)
        features, brain_mask, gt_mask = slice_output[:3]
        ground_truth = (
            np.any(gt_mask > 0, axis=-1) if gt_mask.ndim == 3 else gt_mask > 0
        )
        cache.append(
            {
                "volume": vol_num,
                "brain_mask": brain_mask,
                "ground_truth": ground_truth,
                "healthy_global": healthy_gmm_joint_likelihood(
                    vol_num, lambda_val=0.0, symmetric=False
                ),
                "healthy_local": healthy_gmm_joint_likelihood(
                    vol_num, lambda_val=1.0, symmetric=False
                ),
                "tumor": tumor_joint_likelihood(vol_num, symmetric=False),
                "ndi_features": features[:, :, 4:] if symmetric else None,
                "symmetric_brain_mask": slice_output[3] if symmetric else None,
            }
        )
    return cache


def objective(trial, cache):
    params = {
        "lambda_val": trial.suggest_float("lambda_val", 0.0, 0.6, step=0.02),
        "tumor_prior_scale": trial.suggest_float(
            "tumor_prior_scale", 0.5, 4.0, log=True
        ),
        "min_pixels_per_blob": trial.suggest_int("min_pixels_per_blob", 5, 40),
        "binarization_factor": trial.suggest_float(
            "binarization_factor", 0.4, 1.2, step=0.02
        ),
        "blob_class_threshold": trial.suggest_float(
            "blob_class_threshold", 0.05, 0.6, step=0.02
        ),
        "entropy_thresh": trial.suggest_float("entropy_thresh", 0.02, 0.5, step=0.02),
        "posterior_min": trial.suggest_float("posterior_min", 0.05, 0.6, step=0.02),
        "max_expansion_diameter": trial.suggest_int("max_expansion_diameter", 5, 35),
    }
    dice_scores = []
    missed_tumors = 0
    for item in cache:
        healthy = (
            (1.0 - params["lambda_val"]) * item["healthy_global"]
            + params["lambda_val"] * item["healthy_local"]
        )
        segmentation = segment_likelihoods(
            healthy,
            params["tumor_prior_scale"] * item["tumor"],
            item["brain_mask"],
            min_pixels_per_blob=params["min_pixels_per_blob"],
            binarization_factor=params["binarization_factor"],
            blob_class_threshold=params["blob_class_threshold"],
            entropy_thresh=params["entropy_thresh"],
            posterior_min=params["posterior_min"],
            max_expansion_diameter=params["max_expansion_diameter"],
        )
        metrics = calculate_metrics(segmentation["prediction"], item["ground_truth"])
        dice_scores.append(metrics["dice"])
        missed_tumors += metrics["gt_size"] > 0 and metrics["intersection"] == 0

    trial.set_user_attr("missed_tumors", int(missed_tumors))
    return float(np.mean(dice_scores))


def run_optimization(n_trials=40, symmetric=False):
    """Tune the Spatial GMM using validation volumes only."""
    if symmetric:
        raise ValueError("Use run_ndi_optimization() for the NDI fusion model.")
    mode = "spatial_gmm"
    cache = build_validation_cache(symmetric=False)
    study = optuna.create_study(
        study_name=f"{mode}_validation",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(lambda trial: objective(trial, cache), n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    save_path = os.path.join(PROJECT_ROOT, "saved_parameters", f"{mode}_best_params.npz")
    np.savez(
        save_path,
        **best_params,
        validation_dice=study.best_value,
        validation_missed_tumors=study.best_trial.user_attrs["missed_tumors"],
    )
    print(f"Best validation Dice: {study.best_value:.4f}")
    print(f"Missed validation tumors: {study.best_trial.user_attrs['missed_tumors']}")
    print("Best parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")
    print(f"Saved to: {save_path}")
    return best_params


def evaluate_ndi_params(
    cache, spatial_params, ndi_strength, ndi_percentile, ndi_posterior_gate
):
    """Evaluate fixed Spatial GMM parameters with a candidate NDI fusion."""
    dice_scores = []
    missed_tumors = false_positive_empty = 0
    for item in cache:
        healthy = (
            (1.0 - spatial_params["lambda_val"]) * item["healthy_global"]
            + spatial_params["lambda_val"] * item["healthy_local"]
        )
        tumor, _ = apply_ndi_fusion(
            spatial_params["tumor_prior_scale"] * item["tumor"],
            item["ndi_features"],
            item["symmetric_brain_mask"],
            ndi_strength=ndi_strength,
            ndi_percentile=ndi_percentile,
            healthy_likelihood=healthy,
            ndi_posterior_gate=ndi_posterior_gate,
        )
        segmentation = segment_likelihoods(
            healthy,
            tumor,
            item["brain_mask"],
            min_pixels_per_blob=spatial_params["min_pixels_per_blob"],
            binarization_factor=spatial_params["binarization_factor"],
            blob_class_threshold=spatial_params["blob_class_threshold"],
            entropy_thresh=spatial_params["entropy_thresh"],
            posterior_min=spatial_params["posterior_min"],
            max_expansion_diameter=spatial_params["max_expansion_diameter"],
        )
        metrics = calculate_metrics(segmentation["prediction"], item["ground_truth"])
        dice_scores.append(metrics["dice"])
        missed_tumors += metrics["gt_size"] > 0 and metrics["intersection"] == 0
        false_positive_empty += metrics["gt_size"] == 0 and metrics["pred_size"] > 0
    return float(np.mean(dice_scores)), int(missed_tumors), int(false_positive_empty)


def run_ndi_optimization(spatial_params, n_trials=40):
    """Tune only a bounded NDI boost while keeping the Spatial GMM fixed."""
    cache = build_validation_cache(symmetric=True)
    baseline = evaluate_ndi_params(cache, spatial_params, 0.0, 90.0, 0.1)

    def ndi_objective(trial):
        strength = trial.suggest_float("ndi_strength", 0.0, 2.0, step=0.1)
        percentile = trial.suggest_float("ndi_percentile", 80.0, 98.0, step=1.0)
        posterior_gate = trial.suggest_float(
            "ndi_posterior_gate", 0.05, 0.4, step=0.05
        )
        mean_dice, missed, false_positives = evaluate_ndi_params(
            cache, spatial_params, strength, percentile, posterior_gate
        )
        trial.set_user_attr("missed_tumors", missed)
        trial.set_user_attr("false_positive_empty", false_positives)
        return mean_dice

    study = optuna.create_study(
        study_name="spatial_gmm_ndi_fusion_validation",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.enqueue_trial(
        {"ndi_strength": 0.0, "ndi_percentile": 90.0, "ndi_posterior_gate": 0.1}
    )
    study.optimize(ndi_objective, n_trials=n_trials, show_progress_bar=True)
    acceptable_trials = [
        trial for trial in study.trials
        if trial.value is not None
        and trial.user_attrs["missed_tumors"] <= baseline[1]
        and trial.user_attrs["false_positive_empty"] <= baseline[2]
    ]
    best_trial = max(acceptable_trials, key=lambda trial: trial.value)
    best_params = {**spatial_params, **best_trial.params}
    save_path = os.path.join(
        PROJECT_ROOT, "saved_parameters", "spatial_gmm_ndi_fusion_best_params.npz"
    )
    np.savez(
        save_path,
        **best_params,
        validation_dice=best_trial.value,
        validation_missed_tumors=best_trial.user_attrs["missed_tumors"],
        validation_false_positive_empty=best_trial.user_attrs["false_positive_empty"],
    )
    print(
        f"Spatial validation Dice: {baseline[0]:.4f}; "
        f"NDI fusion Dice: {best_trial.value:.4f}"
    )
    print(
        f"NDI validation misses: {best_trial.user_attrs['missed_tumors']}; "
        f"empty-slice false positives: {best_trial.user_attrs['false_positive_empty']}"
    )
    print(f"Best NDI parameters: {best_trial.params}")
    print(f"Saved to: {save_path}")
    return best_params


if __name__ == "__main__":
    run_optimization(n_trials=40, symmetric=False)
