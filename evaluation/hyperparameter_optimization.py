import json
from pathlib import Path

import numpy as np
import optuna

from config import (
    ALLOW_INTERNAL_CONTOURS,
    ENTROPY_THRESHOLD_ALL,
    ENTROPY_THRESHOLD_BOUNDARY_DISTANCE,
    ENTROPY_THRESHOLD_RAW,
    ENTROPY_THRESHOLD_SYMMETRIC,
    MAX_EXPANSION_DIAMETER_DEFAULT,
    MIN_NUM_PIXELS_PER_BLOB_DEFAULT,
    POSTERIOR_THRESHOLD_ALL,
    POSTERIOR_THRESHOLD_BOUNDARY_DISTANCE,
    POSTERIOR_THRESHOLD_RAW,
    POSTERIOR_THRESHOLD_SYMMETRIC,
    PROJECT_ROOT,
    RANDOM_SEED,
    SLICE_NUM,
    SOBEL_BINARIZATION_OTSU_FACTOR,
    WEIGHTED_POSTERIOR_MEAN_THRESHOLD_ALL,
    WEIGHTED_POSTERIOR_MEAN_THRESHOLD_BOUNDARY_DISTANCE,
    WEIGHTED_POSTERIOR_MEAN_THRESHOLD_RAW,
    WEIGHTED_POSTERIOR_MEAN_THRESHOLD_SYMMETRIC,
)
from evaluation.evaluate_test_set import dataset_eval


OPTIMIZATION_PATH = Path(PROJECT_ROOT) / "saved_parameters" / "validation_optimization.json"
REFERENCE_MODEL_PROBABILITY_PARAMS = {
    "Raw (4D)": {
        "posterior_mean_threshold": WEIGHTED_POSTERIOR_MEAN_THRESHOLD_RAW,
        "entropy_expansion_threshold": ENTROPY_THRESHOLD_RAW,
        "posterior_expansion_threshold": POSTERIOR_THRESHOLD_RAW,
    },
    "Boundary distance (5D)": {
        "posterior_mean_threshold": WEIGHTED_POSTERIOR_MEAN_THRESHOLD_BOUNDARY_DISTANCE,
        "entropy_expansion_threshold": ENTROPY_THRESHOLD_BOUNDARY_DISTANCE,
        "posterior_expansion_threshold": POSTERIOR_THRESHOLD_BOUNDARY_DISTANCE,
    },
    "Symmetry (8D)": {
        "posterior_mean_threshold": WEIGHTED_POSTERIOR_MEAN_THRESHOLD_SYMMETRIC,
        "entropy_expansion_threshold": ENTROPY_THRESHOLD_SYMMETRIC,
        "posterior_expansion_threshold": POSTERIOR_THRESHOLD_SYMMETRIC,
    },
    "Combined (9D)": {
        "posterior_mean_threshold": WEIGHTED_POSTERIOR_MEAN_THRESHOLD_ALL,
        "entropy_expansion_threshold": ENTROPY_THRESHOLD_ALL,
        "posterior_expansion_threshold": POSTERIOR_THRESHOLD_ALL,
    },
}


def _save_selection(selection, path=OPTIMIZATION_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2), encoding="utf-8")


def load_validation_parameters(path=OPTIMIZATION_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    return json.loads(path.read_text(encoding="utf-8"))


def optimize_baseline_parameters(
    baseline_name,
    baseline_files,
    validation_volumes,
    n_trials=60,
    seed=RANDOM_SEED,
    save_path=OPTIMIZATION_PATH,
):
    """Jointly optimize baseline spatial processing and probability thresholds."""
    validation_volumes = list(validation_volumes)

    def objective(trial):
        image_params = {
            "min_pixels_per_blob": trial.suggest_int(
                "min_pixels_per_blob", 10, 80
            ),
            "sobel_binarization_factor": trial.suggest_float(
                "sobel_binarization_factor", 0.35, 0.85
            ),
            "allow_internal_contours": trial.suggest_categorical(
                "allow_internal_contours", [False, True]
            ),
            "max_expansion_diameter": trial.suggest_int(
                "max_expansion_diameter", 10, 100
            ),
        }
        probability_params = {
            "posterior_mean_threshold": trial.suggest_float(
                "posterior_mean_threshold", 0.30, 0.90
            ),
            "entropy_expansion_threshold": trial.suggest_float(
                "entropy_expansion_threshold", 0.02, 0.40
            ),
            "posterior_expansion_threshold": trial.suggest_float(
                "posterior_expansion_threshold", 0.02, 0.60
            ),
        }
        dice, _ = dataset_eval(
            validation_volumes,
            slice_num=SLICE_NUM,
            image_processing_params=image_params,
            **baseline_files,
            **probability_params,
        )
        return float(np.mean(dice))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    reference_params = {
        "min_pixels_per_blob": MIN_NUM_PIXELS_PER_BLOB_DEFAULT,
        "sobel_binarization_factor": SOBEL_BINARIZATION_OTSU_FACTOR,
        "allow_internal_contours": ALLOW_INTERNAL_CONTOURS,
        "max_expansion_diameter": MAX_EXPANSION_DIAMETER_DEFAULT,
        "posterior_mean_threshold": WEIGHTED_POSTERIOR_MEAN_THRESHOLD_RAW,
        "entropy_expansion_threshold": ENTROPY_THRESHOLD_RAW,
        "posterior_expansion_threshold": POSTERIOR_THRESHOLD_RAW,
    }
    study.enqueue_trial(reference_params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    image_keys = [
        "min_pixels_per_blob",
        "sobel_binarization_factor",
        "allow_internal_contours",
        "max_expansion_diameter",
    ]
    probability_keys = [
        "posterior_mean_threshold",
        "entropy_expansion_threshold",
        "posterior_expansion_threshold",
    ]
    selection = {
        "workflow_version": "baseline_then_improvements_v2_no_large_contour",
        "selection_split": (
            f"volumes {validation_volumes[0]}-{validation_volumes[-1]}"
        ),
        "baseline_model": baseline_name,
        "selection_method": (
            "Baseline study jointly selects image-processing parameters and "
            "baseline probability thresholds. Improved models reuse the frozen "
            "image-processing parameters and optimize only their probability thresholds."
        ),
        "shared_image_processing_params": {
            key: study.best_params[key] for key in image_keys
        },
        "model_probability_params": {
            baseline_name: {
                key: study.best_params[key] for key in probability_keys
            }
        },
        "model_validation_mean_dice": {baseline_name: study.best_value},
        "trial_counts": {baseline_name: n_trials},
    }
    _save_selection(selection, save_path)
    return selection, study


def optimize_model_probability_parameters(
    model_name,
    model_files,
    validation_volumes,
    selected_parameters,
    n_trials=60,
    seed=RANDOM_SEED,
    save_path=OPTIMIZATION_PATH,
):
    """Optimize one improved model and checkpoint its selected thresholds."""
    validation_volumes = list(validation_volumes)
    shared_params = selected_parameters["shared_image_processing_params"]

    def objective(trial):
        probability_params = {
            "posterior_mean_threshold": trial.suggest_float(
                "posterior_mean_threshold", 0.30, 0.90
            ),
            "entropy_expansion_threshold": trial.suggest_float(
                "entropy_expansion_threshold", 0.02, 0.40
            ),
            "posterior_expansion_threshold": trial.suggest_float(
                "posterior_expansion_threshold", 0.02, 0.60
            ),
        }
        dice, _ = dataset_eval(
            validation_volumes,
            slice_num=SLICE_NUM,
            image_processing_params=shared_params,
            **model_files,
            **probability_params,
        )
        return float(np.mean(dice))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    if model_name in REFERENCE_MODEL_PROBABILITY_PARAMS:
        study.enqueue_trial(REFERENCE_MODEL_PROBABILITY_PARAMS[model_name])
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    selected_parameters["model_probability_params"][model_name] = study.best_params
    selected_parameters["model_validation_mean_dice"][model_name] = study.best_value
    selected_parameters.setdefault("trial_counts", {})[model_name] = n_trials
    _save_selection(selected_parameters, save_path)
    return selected_parameters, study
