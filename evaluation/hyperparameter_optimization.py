import json
from pathlib import Path

import numpy as np
import optuna

from config import PROJECT_ROOT, RANDOM_SEED, SLICE_NUM
from evaluation.evaluate_test_set import dataset_eval


OPTIMIZATION_PATH = Path(PROJECT_ROOT) / "saved_parameters" / "validation_optimization.json"


def _optimize_shared_image_parameters(
    model_specs,
    initial_probability_params,
    validation_volumes,
    n_trials,
    seed,
):
    def objective(trial):
        shared = {
            "min_pixels_per_blob": trial.suggest_int("min_pixels_per_blob", 10, 80, step=5),
            "sobel_binarization_factor": trial.suggest_float("sobel_otsu_factor", 0.35, 0.85, step=0.05),
            "allow_internal_contours": trial.suggest_categorical("allow_internal_contours", [False, True]),
            "large_contour_min_area": trial.suggest_int("large_contour_min_area", 200, 12000, step=200),
            "max_expansion_diameter": trial.suggest_int("max_expansion_diameter", 10, 100, step=10),
        }
        scores = []
        for model_name, files in model_specs.items():
            dice, _ = dataset_eval(
                validation_volumes,
                slice_num=SLICE_NUM,
                image_processing_params=shared,
                **files,
                **initial_probability_params[model_name],
            )
            scores.append(float(np.mean(dice)))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def _optimize_one_model(
    files,
    shared_params,
    validation_volumes,
    n_trials,
    seed,
):
    def objective(trial):
        dice, _ = dataset_eval(
            validation_volumes,
            slice_num=SLICE_NUM,
            image_processing_params=shared_params,
            posterior_mean_threshold=trial.suggest_float(
                "posterior_mean_threshold", 0.30, 0.90, step=0.02
            ),
            entropy_expansion_threshold=trial.suggest_float(
                "entropy_expansion_threshold", 0.02, 0.40, step=0.02
            ),
            posterior_expansion_threshold=trial.suggest_float(
                "posterior_expansion_threshold", 0.02, 0.60, step=0.02
            ),
            **files,
        )
        return float(np.mean(dice))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def optimize_validation_parameters(
    model_specs,
    initial_probability_params,
    validation_volumes,
    n_shared_trials=60,
    n_model_trials=100,
    seed=RANDOM_SEED,
    save_path=OPTIMIZATION_PATH,
):
    """Tune shared spatial parameters, then tune each model independently.

    Stage 1 selects one image-processing configuration by maximizing the mean
    validation Dice across all models. Stage 2 freezes that configuration and
    runs an independent probability-threshold study for every model. Test
    volumes are never used.
    """
    validation_volumes = list(validation_volumes)
    shared_study = _optimize_shared_image_parameters(
        model_specs,
        initial_probability_params,
        validation_volumes,
        n_shared_trials,
        seed,
    )
    shared_params = shared_study.best_params

    model_studies = {}
    model_params = {}
    validation_scores = {}
    for index, (model_name, files) in enumerate(model_specs.items()):
        study = _optimize_one_model(
            files,
            shared_params,
            validation_volumes,
            n_model_trials,
            seed + index + 1,
        )
        model_studies[model_name] = study
        model_params[model_name] = study.best_params
        validation_scores[model_name] = study.best_value

    selected = {
        "selection_split": f"volumes {validation_volumes[0]}-{validation_volumes[-1]}",
        "selection_method": (
            "Stage 1: shared image-processing parameters maximize mean validation "
            "Dice across all models. Stage 2: separate Optuna study per model with "
            "shared parameters frozen."
        ),
        "shared_trials": n_shared_trials,
        "model_trials_each": n_model_trials,
        "shared_best_validation_objective": shared_study.best_value,
        "shared_image_processing_params": shared_params,
        "model_probability_params": model_params,
        "model_validation_mean_dice": validation_scores,
    }
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected, {"Shared image processing": shared_study, **model_studies}


def load_validation_parameters(path=OPTIMIZATION_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run optimize_validation_parameters first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
