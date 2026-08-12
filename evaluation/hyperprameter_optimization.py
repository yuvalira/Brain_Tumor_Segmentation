import os

import numpy as np
import optuna

from config import *
from evaluation.evaluate_single_slice import (
    apply_ndi_fusion,
    apply_z_context_fusion,
    build_z_context_score,
    calculate_metrics,
    load_z_neighbor_likelihoods,
    segment_likelihoods,
)
from statistical_models.healthy_likelihood import healthy_gmm_joint_likelihood
from statistical_models.tumor_likelihoods import tumor_joint_likelihood
from utilities.utils import load_and_normalize_slice

optuna.logging.set_verbosity(optuna.logging.WARNING)

SPATIAL_OPTIMIZATION_VERSION = 2
Z_OPTIMIZATION_VERSION = 1
SPATIAL_PRECISION_TOLERANCE = 0.01
SPATIAL_PARAMETER_NAMES = [
    "lambda_val", "tumor_prior_scale", "min_pixels_per_blob",
    "binarization_factor", "blob_class_threshold", "large_contour_min_area",
    "top_posterior_mean_threshold", "high_posterior_fraction_threshold",
    "entropy_thresh", "posterior_min", "max_expansion_diameter",
]


def build_validation_cache(symmetric=False, include_z=False):
    """Precompute validation likelihood maps once."""
    cache = []
    validation_volumes = range(MAX_TRAINING_VOLUME + 1, MAX_VALIDATION_VOLUME + 1)
    print(f"Caching {len(validation_volumes)} validation slices (symmetric={symmetric})...")
    for vol_num in validation_volumes:
        slice_output = load_and_normalize_slice(vol_num, SLICE_NUM, symmetric=symmetric)
        features, brain_mask, gt_mask = slice_output[:3]
        ground_truth = np.any(gt_mask > 0, axis=-1) if gt_mask.ndim == 3 else gt_mask > 0
        cache.append({
            "volume": vol_num,
            "brain_mask": brain_mask,
            "ground_truth": ground_truth,
            "healthy_global": healthy_gmm_joint_likelihood(vol_num, lambda_val=0.0, symmetric=False),
            "healthy_local": healthy_gmm_joint_likelihood(vol_num, lambda_val=1.0, symmetric=False),
            "tumor": tumor_joint_likelihood(vol_num, symmetric=False),
            "z_neighbors": load_z_neighbor_likelihoods(vol_num) if include_z else None,
            "ndi_features": features[:, :, 4:] if symmetric else None,
            "symmetric_brain_mask": slice_output[3] if symmetric else None,
        })
    return cache


def default_spatial_params():
    return {
        "lambda_val": LAMBDA,
        "tumor_prior_scale": 1.0,
        "min_pixels_per_blob": MIN_NUM_PIXELS_PER_BLOB_DEFAULT,
        "binarization_factor": SOBEL_BINARIZATION_OTSU_FACTOR,
        "blob_class_threshold": WEIGHTED_POSTERIOR_MEAN_THRESHOLD,
        "large_contour_min_area": LARGE_CONTOUR_MIN_AREA_DEFAULT,
        "top_posterior_mean_threshold": TOP_POSTERIOR_MEAN_THRESHOLD_DEFAULT,
        "high_posterior_fraction_threshold": HIGH_POSTERIOR_FRACTION_THRESHOLD_DEFAULT,
        "entropy_thresh": ENTROPY_THRESHOLD_DEFAULT,
        "posterior_min": POSTERIOR_THRESHOLD_DEFAULT,
        "max_expansion_diameter": MAX_EXPANSION_DIAMETER_DEFAULT,
    }


def load_reference_params(save_path):
    """Load the previous Spatial solution as the minimum acceptable reference."""
    if os.path.exists(save_path):
        with np.load(save_path) as saved:
            if all(name in saved.files for name in SPATIAL_PARAMETER_NAMES):
                return {name: saved[name].item() for name in SPATIAL_PARAMETER_NAMES}
    return default_spatial_params()


def evaluate_spatial_params(cache, params):
    dice_all, dice_tumor, precision_tumor = [], [], []
    missed_tumors = false_positive_empty = 0
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
            large_contour_min_area=params["large_contour_min_area"],
            top_posterior_mean_threshold=params["top_posterior_mean_threshold"],
            high_posterior_fraction_threshold=params["high_posterior_fraction_threshold"],
            entropy_thresh=params["entropy_thresh"],
            posterior_min=params["posterior_min"],
            max_expansion_diameter=params["max_expansion_diameter"],
        )
        metrics = calculate_metrics(segmentation["prediction"], item["ground_truth"])
        dice_all.append(metrics["dice"])
        if metrics["gt_size"] > 0:
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


def suggest_spatial_params(trial):
    return {
        "lambda_val": trial.suggest_float("lambda_val", 0.0, 1.0, step=0.02),
        "tumor_prior_scale": trial.suggest_float("tumor_prior_scale", 0.5, 8.0, log=True),
        "min_pixels_per_blob": trial.suggest_int("min_pixels_per_blob", 5, 40),
        "binarization_factor": trial.suggest_float("binarization_factor", 0.4, 1.2, step=0.02),
        "blob_class_threshold": trial.suggest_float("blob_class_threshold", 0.05, 0.6, step=0.02),
        "large_contour_min_area": trial.suggest_int("large_contour_min_area", 100, 2000, step=100),
        "top_posterior_mean_threshold": trial.suggest_float("top_posterior_mean_threshold", 0.3, 0.8, step=0.05),
        "high_posterior_fraction_threshold": trial.suggest_float("high_posterior_fraction_threshold", 0.1, 0.6, step=0.05),
        "entropy_thresh": trial.suggest_float("entropy_thresh", 0.02, 0.5, step=0.02),
        "posterior_min": trial.suggest_float("posterior_min", 0.05, 0.6, step=0.02),
        "max_expansion_diameter": trial.suggest_int("max_expansion_diameter", 5, 35),
    }


def objective(trial, cache):
    metrics = evaluate_spatial_params(cache, suggest_spatial_params(trial))
    for name, value in metrics.items():
        trial.set_user_attr(name, value)
    return metrics["tumor_present_dice"]


def run_optimization(n_trials=100, symmetric=False):
    """Maximize tumor Dice without sacrificing current false-positive control."""
    if symmetric:
        raise ValueError("Use run_ndi_optimization() for the NDI fusion model.")

    mode = "spatial_gmm"
    save_path = os.path.join(PROJECT_ROOT, "saved_parameters", f"{mode}_best_params.npz")
    cache = build_validation_cache(symmetric=False)
    reference_params = load_reference_params(save_path)
    reference = evaluate_spatial_params(cache, reference_params)
    precision_floor = max(
        0.0, reference["tumor_present_precision"] - SPATIAL_PRECISION_TOLERANCE
    )

    study = optuna.create_study(
        study_name=f"{mode}_constrained_validation",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.enqueue_trial(reference_params)
    study.optimize(lambda trial: objective(trial, cache), n_trials=n_trials, show_progress_bar=True)

    acceptable_trials = [
        trial for trial in study.trials
        if trial.value is not None
        and trial.user_attrs["false_positive_empty"] <= reference["false_positive_empty"]
        and trial.user_attrs["missed_tumors"] <= reference["missed_tumors"]
        and trial.user_attrs["tumor_present_precision"] >= precision_floor
    ]
    best_trial = max(
        acceptable_trials,
        key=lambda trial: (
            trial.user_attrs["tumor_present_dice"],
            trial.user_attrs["all_slice_dice"],
        ),
    )
    best_params = best_trial.params
    best = best_trial.user_attrs

    np.savez(
        save_path,
        **best_params,
        optimization_version=SPATIAL_OPTIMIZATION_VERSION,
        validation_dice=best["all_slice_dice"],
        validation_tumor_present_dice=best["tumor_present_dice"],
        validation_tumor_present_precision=best["tumor_present_precision"],
        validation_missed_tumors=best["missed_tumors"],
        validation_false_positive_empty=best["false_positive_empty"],
    )
    print(
        f"Reference validation: Dice={reference['all_slice_dice']:.4f}, "
        f"tumor Dice={reference['tumor_present_dice']:.4f}, "
        f"precision={reference['tumor_present_precision']:.4f}, "
        f"misses={reference['missed_tumors']}, empty FP={reference['false_positive_empty']}"
    )
    print(
        f"Selected validation:  Dice={best['all_slice_dice']:.4f}, "
        f"tumor Dice={best['tumor_present_dice']:.4f}, "
        f"precision={best['tumor_present_precision']:.4f}, "
        f"misses={best['missed_tumors']}, empty FP={best['false_positive_empty']}"
    )
    print(f"Precision constraint: >= {precision_floor:.4f}")
    print("Best constrained parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")
    print(f"Saved to: {save_path}")
    return best_params


def evaluate_z_params(cache, spatial_params, z_strength, z_posterior_gate):
    """Evaluate fixed Spatial GMM parameters with axial-neighbor support."""
    dice_all, dice_tumor, precision_tumor = [], [], []
    missed_tumors = false_positive_empty = 0
    for item in cache:
        healthy = (
            (1.0 - spatial_params["lambda_val"]) * item["healthy_global"]
            + spatial_params["lambda_val"] * item["healthy_local"]
        )
        tumor = spatial_params["tumor_prior_scale"] * item["tumor"]
        z_score = build_z_context_score(
            healthy,
            tumor,
            item["z_neighbors"],
            tumor_prior_scale=spatial_params["tumor_prior_scale"],
        )
        tumor = apply_z_context_fusion(
            tumor,
            healthy,
            z_score,
            z_strength=z_strength,
            z_posterior_gate=z_posterior_gate,
        )
        segmentation = segment_likelihoods(
            healthy,
            tumor,
            item["brain_mask"],
            min_pixels_per_blob=spatial_params["min_pixels_per_blob"],
            binarization_factor=spatial_params["binarization_factor"],
            blob_class_threshold=spatial_params["blob_class_threshold"],
            large_contour_min_area=spatial_params["large_contour_min_area"],
            top_posterior_mean_threshold=spatial_params["top_posterior_mean_threshold"],
            high_posterior_fraction_threshold=spatial_params["high_posterior_fraction_threshold"],
            entropy_thresh=spatial_params["entropy_thresh"],
            posterior_min=spatial_params["posterior_min"],
            max_expansion_diameter=spatial_params["max_expansion_diameter"],
        )
        metrics = calculate_metrics(segmentation["prediction"], item["ground_truth"])
        dice_all.append(metrics["dice"])
        if metrics["gt_size"] > 0:
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


def run_z_optimization(spatial_params, n_trials=40):
    """Tune only the Z-context boost while keeping the Spatial GMM fixed."""
    cache = build_validation_cache(symmetric=False, include_z=True)
    baseline = evaluate_z_params(cache, spatial_params, 0.0, 0.0)
    precision_floor = max(
        0.0, baseline["tumor_present_precision"] - SPATIAL_PRECISION_TOLERANCE
    )

    def z_objective(trial):
        strength = trial.suggest_float("z_strength", 0.0, 4.0, step=0.2)
        posterior_gate = trial.suggest_float(
            "z_posterior_gate", 0.02, 0.30, step=0.02
        )
        metrics = evaluate_z_params(cache, spatial_params, strength, posterior_gate)
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        return metrics["tumor_present_dice"]

    study = optuna.create_study(
        study_name="spatial_gmm_z_context_validation",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.enqueue_trial({"z_strength": 0.0, "z_posterior_gate": 0.02})
    study.optimize(z_objective, n_trials=n_trials, show_progress_bar=True)
    acceptable_trials = [
        trial for trial in study.trials
        if trial.value is not None
        and trial.user_attrs["false_positive_empty"] <= baseline["false_positive_empty"]
        and trial.user_attrs["missed_tumors"] <= baseline["missed_tumors"]
        and trial.user_attrs["tumor_present_precision"] >= precision_floor
    ]
    best_trial = max(
        acceptable_trials,
        key=lambda trial: (
            trial.user_attrs["tumor_present_dice"],
            trial.user_attrs["all_slice_dice"],
        ),
    )
    best_params = {**spatial_params, **best_trial.params}
    best = best_trial.user_attrs
    save_path = os.path.join(
        PROJECT_ROOT, "saved_parameters", "spatial_gmm_z_best_params.npz"
    )
    np.savez(
        save_path,
        **best_params,
        z_optimization_version=Z_OPTIMIZATION_VERSION,
        spatial_optimization_version=SPATIAL_OPTIMIZATION_VERSION,
        validation_dice=best["all_slice_dice"],
        validation_tumor_present_dice=best["tumor_present_dice"],
        validation_tumor_present_precision=best["tumor_present_precision"],
        validation_missed_tumors=best["missed_tumors"],
        validation_false_positive_empty=best["false_positive_empty"],
    )
    print(
        f"Spatial validation: Dice={baseline['all_slice_dice']:.4f}, "
        f"tumor Dice={baseline['tumor_present_dice']:.4f}, "
        f"misses={baseline['missed_tumors']}, empty FP={baseline['false_positive_empty']}"
    )
    print(
        f"Spatial + Z:        Dice={best['all_slice_dice']:.4f}, "
        f"tumor Dice={best['tumor_present_dice']:.4f}, "
        f"misses={best['missed_tumors']}, empty FP={best['false_positive_empty']}"
    )
    print(f"Best Z parameters: {best_trial.params}")
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
        tumor = spatial_params["tumor_prior_scale"] * item["tumor"]
        z_score = build_z_context_score(
            healthy,
            tumor,
            item["z_neighbors"],
            tumor_prior_scale=spatial_params["tumor_prior_scale"],
        )
        tumor = apply_z_context_fusion(
            tumor,
            healthy,
            z_score,
            z_strength=spatial_params["z_strength"],
            z_posterior_gate=spatial_params["z_posterior_gate"],
        )
        tumor, _ = apply_ndi_fusion(
            tumor,
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
            large_contour_min_area=spatial_params["large_contour_min_area"],
            top_posterior_mean_threshold=spatial_params["top_posterior_mean_threshold"],
            high_posterior_fraction_threshold=spatial_params["high_posterior_fraction_threshold"],
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
    cache = build_validation_cache(symmetric=True, include_z=True)
    baseline = evaluate_ndi_params(cache, spatial_params, 0.0, 90.0, 0.1)

    def ndi_objective(trial):
        strength = trial.suggest_float("ndi_strength", 0.0, 2.0, step=0.1)
        percentile = trial.suggest_float("ndi_percentile", 80.0, 98.0, step=1.0)
        posterior_gate = trial.suggest_float("ndi_posterior_gate", 0.05, 0.4, step=0.05)
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
        spatial_optimization_version=SPATIAL_OPTIMIZATION_VERSION,
        z_optimization_version=Z_OPTIMIZATION_VERSION,
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
    run_optimization(n_trials=100, symmetric=False)
