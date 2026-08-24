import json
from pathlib import Path

import numpy as np
import optuna

from config import PROJECT_ROOT, RANDOM_SEED, SLICE_NUM
from evaluation.evaluate_test_set import dataset_eval


OPTIMIZATION_PATH = Path(PROJECT_ROOT) / "saved_parameters" / "validation_optimization.json"


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
                "min_pixels_per_blob", 10, 80, step=5
            ),
            "sobel_binarization_factor": trial.suggest_float(
                "sobel_binarization_factor", 0.35, 0.85, step=0.05
            ),
            "allow_internal_contours": trial.suggest_categorical(
                "allow_internal_contours", [False, True]
            ),
            "large_contour_min_area": trial.suggest_int(
                "large_contour_min_area", 200, 12000, step=200
            ),
            "max_expansion_diameter": trial.suggest_int(
                "max_expansion_diameter", 10, 100, step=10
            ),
        }
        probability_params = {
            "posterior_mean_threshold": trial.suggest_float(
                "posterior_mean_threshold", 0.30, 0.90, step=0.02
            ),
            "entropy_expansion_threshold": trial.suggest_float(
                "entropy_expansion_threshold", 0.02, 0.40, step=0.02
            ),
            "posterior_expansion_threshold": trial.suggest_float(
                "posterior_expansion_threshold", 0.02, 0.60, step=0.02
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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    image_keys = [
        "min_pixels_per_blob",
        "sobel_binarization_factor",
        "allow_internal_contours",
        "large_contour_min_area",
        "max_expansion_diameter",
    ]
    probability_keys = [
        "posterior_mean_threshold",
        "entropy_expansion_threshold",
        "posterior_expansion_threshold",
    ]
    selection = {
        "workflow_version": "baseline_then_improvements_v1",
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
                "posterior_mean_threshold", 0.30, 0.90, step=0.02
            ),
            "entropy_expansion_threshold": trial.suggest_float(
                "entropy_expansion_threshold", 0.02, 0.40, step=0.02
            ),
            "posterior_expansion_threshold": trial.suggest_float(
                "posterior_expansion_threshold", 0.02, 0.60, step=0.02
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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    selected_parameters["model_probability_params"][model_name] = study.best_params
    selected_parameters["model_validation_mean_dice"][model_name] = study.best_value
    selected_parameters.setdefault("trial_counts", {})[model_name] = n_trials
    _save_selection(selected_parameters, save_path)
    return selected_parameters, study
