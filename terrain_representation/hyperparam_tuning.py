import argparse
import copy
import datetime as dt
import functools
import multiprocessing as mp
import time
import typing as t

import numpy as np
import optuna
from optuna.trial import TrialState
from terrain_representation.utils.pipeline_utils import create_tools
from terrain_representation.utils.utils import update_args


def optuna_objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    run_pipe: t.Callable,
) -> float:
    """Optuna objective function that is evaluated at each trial. Can work with train/val datasets and multiple metrics. Parameters to tune are specified in args.opt_params_to_tune.
    Args:
        trial: Optuna trial.
        args: Arguments for the study.
        run_pipe: Function to run the pipeline that returns metric history.
    Returns:
        Target metric averaged over the training history.
    """
    params = {}

    if "occupancy_threshold" in args.opt_params_to_tune:
        occupancy_threshold_suggest = trial.suggest_float(
            "occupancy_threshold", 0.3, 0.8
        )
        params["occupancy_threshold"] = occupancy_threshold_suggest

    if "pos_weight" in args.opt_params_to_tune:
        pos_weight_suggest = trial.suggest_float("pos_weight", 3, 10)
        params["pos_weight"] = pos_weight_suggest

    if "centroid_weight" in args.opt_params_to_tune:
        centroid_weight_suggest = trial.suggest_float("centroid_weight", 0.2, 4)
        params["centroid_weight"] = centroid_weight_suggest

    if "start_mae_from_n_epoch" in args.opt_params_to_tune:
        start_mae_from_n_epoch_suggest = trial.suggest_int(
            "start_mae_from_n_epoch", 0, 200
        )
        params["start_mae_from_n_epoch"] = start_mae_from_n_epoch_suggest
    if "lr" in args.opt_params_to_tune:
        lr_suggest = trial.suggest_float("lr", 1e-4, 1e-2)
        params["lr"] = lr_suggest
    if "lr_decay" in args.opt_params_to_tune:
        lr_decay_suggest = trial.suggest_float("lr_decay", 0.95, 0.999)
        params["lr_decay"] = lr_decay_suggest

    args = update_args(args, params)

    tools = create_tools(args)
    exp = tools["exp"]
    exp.add_tags([f"trial_{trial.number}"])

    fit_metrics = run_pipe(
        args=args,
        tools=tools,
    )

    exp.end()

    target_metrics = args.opt_target_metrics
    ds_key_for_metrics = "val" if args.use_val_ds else "train"
    stage_history = fit_metrics[ds_key_for_metrics]
    if target_metrics == "mixed":
        return 2 * np.mean(stage_history["f1"]) + 1 / (
            1 + np.mean(stage_history["mae"])
        )
    elif "val_real" in fit_metrics:
        stage_history_val_real = fit_metrics["val_real"]

        return (
            np.mean(stage_history[target_metrics])
            + np.mean(stage_history_val_real[target_metrics])
        ) / 2
    else:
        return np.mean(stage_history[target_metrics])


def create_study(args: argparse.Namespace, study_name: str) -> optuna.study.Study:
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    direction = (
        "maximize"
        if args.opt_target_metrics in ["f1", "recall", "precision", "mixed"]
        else "minimize"
    )
    study = optuna.create_study(
        direction=direction,
        pruner=pruner,
        study_name=study_name,
        storage=args.opt_storage,
        load_if_exists=True,
    )
    return study


def _run_study(args, run_pipe, study_name):

    args = copy.deepcopy(args)
    args.num_epochs = args.opt_n_epochs
    args.vis_disabled = True
    args.use_async_vis = False
    args.do_calc_dist_metrics = (
        False if args.opt_target_metrics not in ["mae", "mixed"] else True
    )
    study = create_study(args, study_name)
    objective = functools.partial(optuna_objective, args=args, run_pipe=run_pipe)
    study.optimize(objective, n_trials=args.opt_n_trials, timeout=None)
    return study


def run_study(args: argparse.Namespace, run_pipe: t.Callable) -> t.Dict[str, t.Any]:
    """Run optuna study to find best hyperparameters based on the target metric.
    Args:
        args: Arguments for the study.
        run_pipe: Function to run the pipeline that returns metric history.
    Returns:
        Best hyperparameters found by the study.
    """

    now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    study_name = f"{now}_{args.opt_study_prefix}_{args.opt_target_metrics}"
    pool_args = [(args, run_pipe, study_name) for _ in range(args.opt_n_jobs)]
    study = create_study(args, study_name)
    with mp.Pool(args.opt_n_jobs) as pool:
        results = pool.starmap(_run_study, pool_args)

    study = optuna.load_study(study_name=study_name, storage=args.opt_storage)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("Trail with : \n")
    print("=========================================")
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial.params
