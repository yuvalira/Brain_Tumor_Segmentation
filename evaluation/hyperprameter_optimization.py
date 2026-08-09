import os
import optuna
import numpy as np
from config import *
from evaluation.evaluate_single_slice import eval_vol

# Suppress Optuna per-trial verbose logging for cleaner console output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, symmetric: bool = False, metric_mode: str = "volumewise"):
    """
    Optuna objective function that samples pipeline hyperparameters and returns
    either the mean per-volume Dice score or the global aggregate Dice score across K validation folds.

    :param trial: Optuna Trial object.
    :param symmetric: If True, uses 8D features (4 base + 4 spatial symmetry features).
    :param metric_mode: 'volumewise' (average of individual volume Dices) or
                        'global' (pooled aggregate Dice using total pixel counts).
    """
    # 1. Sample Hyperparameters from Search Space
    lambda_val = trial.suggest_float("lambda_val", 0.0, 1.0, step=0.01)
    min_pixels_per_blob = trial.suggest_int("min_pixels_per_blob", 5, 31)
    binarization_factor = trial.suggest_float("binarization_factor", 0.5, 2.0, step=0.01)
    blob_class_threshold = trial.suggest_float("blob_class_threshold", 0, 1, step=0.01)
    entropy_thresh = trial.suggest_float("entropy_thresh", 0, 1.0, step=0.01)
    posterior_min = trial.suggest_float("posterior_min", 0, 1, step=0.01)
    max_expansion_diameter = trial.suggest_int("max_expansion_diameter", 3, 31)
    allow_internal = True

    # 2. Define Validation Range and Split into 5 Folds
    min_vol = MAX_TRAINING_VOLUME + 1
    max_vol = MAX_VALIDATION_VOLUME

    opt_volumes = np.array(list(range(min_vol, max_vol + 1)))

    np.random.seed(42)
    shuffled_volumes = opt_volumes.copy()
    np.random.shuffle(shuffled_volumes)

    folds = np.array_split(shuffled_volumes, 5)

    # 3. Evaluate Pipeline across Folds based on metric_mode
    if metric_mode == "volumewise":
        fold_mean_dices = []

        for fold in folds:
            volume_dices = []
            for vol_num in fold:
                dice, _, _, _, _, _ = eval_vol(
                    vol_num=vol_num,
                    target_row=None,
                    diagnostic_figures=False,
                    verbose=False,
                    symmetric=symmetric,
                    lambda_val=lambda_val,
                    min_pixels_per_blob=min_pixels_per_blob,
                    allow_internal=allow_internal,
                    binarization_factor=binarization_factor,
                    blob_class_threshold=blob_class_threshold,
                    entropy_thresh=entropy_thresh,
                    posterior_min=posterior_min,
                    max_expansion_diameter=max_expansion_diameter,
                )
                volume_dices.append(dice)

            fold_mean_dices.append(np.mean(volume_dices))

        return float(np.mean(fold_mean_dices))

    elif metric_mode == "global":
        total_intersection = 0
        total_pred_sum = 0
        total_gt_sum = 0

        for fold in folds:
            for vol_num in fold:
                _, _, pred_AND_gt, _, pred, gt = eval_vol(
                    vol_num=vol_num,
                    target_row=None,
                    diagnostic_figures=False,
                    verbose=False,
                    symmetric=symmetric,
                    lambda_val=lambda_val,
                    min_pixels_per_blob=min_pixels_per_blob,
                    allow_internal=allow_internal,
                    binarization_factor=binarization_factor,
                    blob_class_threshold=blob_class_threshold,
                    entropy_thresh=entropy_thresh,
                    posterior_min=posterior_min,
                    max_expansion_diameter=max_expansion_diameter,
                )
                total_intersection += pred_AND_gt
                total_pred_sum += pred
                total_gt_sum += gt

        # Calculate dataset-wide pooled global Dice score
        global_dice = (2.0 * total_intersection) / (total_pred_sum + total_gt_sum + 1e-12)
        return float(global_dice)

    else:
        raise ValueError(f"Invalid metric_mode '{metric_mode}'. Choose 'volumewise' or 'global'.")


def run_optimization(n_trials=50, symmetric: bool = False, metric_mode: str = "volumewise"):
    """
    Executes the Optuna study maximization loop and prints best parameters.

    :param n_trials: Total number of optimization trials.
    :param symmetric: If True, uses 8D feature space.
    :param metric_mode: 'volumewise' or 'global'.
    """
    mode_str = "symmetric" if symmetric else "regular"
    study_name = f"mri_pipeline_{mode_str}_{metric_mode}"

    # Create study with CMA-ES Sampler targeting maximum Dice score
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(
            seed=RANDOM_SEED,
            consider_pruned_trials=True
        ),
    )

    print(f"Starting Optuna Optimization across {n_trials} trials...")
    print(f"Mode: {mode_str.upper()} | Target Metric: {metric_mode.upper()}\n")

    study.optimize(
        lambda trial: objective(trial, symmetric=symmetric, metric_mode=metric_mode),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n==========================================")
    print("      OPTIMIZATION COMPLETE               ")
    print("==========================================")
    print(f"Best ({metric_mode.upper()}) Dice Score: {study.best_value:.4f}\n")
    print("Optimal Hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key:<25} = {value:.2f}")
        else:
            print(f"  {key:<25} = {value}")
    print("==========================================")

    # Save best parameters to disk with descriptive filename
    param_filename = f"optuna_best_params_{mode_str}_{metric_mode}.npz"
    save_path = os.path.join(PROJECT_ROOT, "saved_parameters", param_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(save_path, **study.best_params, best_dice=study.best_value, metric_mode=metric_mode)
    print(f"Best parameters saved to '{save_path}'")

    return study.best_params


if __name__ == "__main__":
    # Examples:
    # 1. Optimize for Mean Per-Volume Dice Score
    run_optimization(n_trials=50, symmetric=False, metric_mode="volumewise")

    # 2. Optimize for Pooled Global Dataset Dice Score
    # run_optimization(n_trials=50, symmetric=True, metric_mode="global")