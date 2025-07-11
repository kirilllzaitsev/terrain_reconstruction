import argparse
import glob
import os
import re
import typing as t
from pathlib import Path

import comet_ml
import torch
import yaml
from comet_ml.api import API

root_dir = Path(__file__).parent.parent.parent


def get_latest_ckpt_epoch(
    exp_name: str,
    model_name_regex: str = r"model_(\d+)\.pt*",
    project_name: str = "terrain-reconstruction",
) -> int:
    """Infers the latest checkpoint epoch from the experiment's assets."""

    api = API(api_key=os.environ["COMET_API_KEY"])
    exp_api = api.get(f"kirilllzaitsev/{project_name}/{exp_name}")
    ckpt_epochs = [
        int(re.match(model_name_regex, x["fileName"]).group(1))
        for x in exp_api.get_asset_list(asset_type="all")
        if re.match(model_name_regex, x["fileName"])
    ]
    return max(ckpt_epochs)


def load_artifacts_from_comet(
    exp_name: str,
    args_file_path: str,
    model_checkpoint_path: str,
    local_artifacts_dir: str,
    model_artifact_name: str,
    args_name_regex: str = "train_args",
    session_artifact_name: t.Optional[str] = None,
    session_checkpoint_path: t.Optional[str] = None,
    project_name: str = "terrain-reconstruction",
    api: t.Optional[API] = None,
    epoch: t.Optional[int] = None,
) -> dict:
    """Downloads artifacts from comet.ml if they don't exist locally and returns the paths to them.
    Args:
        exp_name: The name of the Comet experiment.
        args_file_path: The local path to the args file.
        model_checkpoint_path: The local path to the model checkpoint.
        local_artifacts_dir: The directory to save the artifacts to.
        model_artifact_name: The path to the model artifact in the experiment Assets.
        args_name_regex: The regex to match the args file name.
        session_artifact_name: The name of the session artifact.
        session_checkpoint_path: The path to the session checkpoint.
        project_name: The name of the project.
        api: The comet.ml API object (takes time to initialize, so it's better to pass it as an argument if done multiple times).
        epoch: The epoch of the model checkpoint to download. It will be inferred if not provided.
    Returns:
        A dictionary containing the paths to the artifacts.
    """

    include_session = (
        session_artifact_name is not None and session_checkpoint_path is not None
    )
    args_not_exist = not os.path.exists(args_file_path)
    weights_not_exist = not os.path.exists(model_checkpoint_path)

    if any([args_not_exist, weights_not_exist]):
        if api is None:
            api = API(api_key=os.environ["COMET_API_KEY"])
        exp_api = api.get(f"kirilllzaitsev/{project_name}/{exp_name}")
        os.makedirs(local_artifacts_dir, exist_ok=True)
        if args_not_exist:
            try:
                asset_id = [
                    x
                    for x in exp_api.get_asset_list(asset_type="all")
                    if f"{args_name_regex}" in x["fileName"]
                ][0]["assetId"]
                api.download_experiment_asset(
                    exp_api.id,
                    asset_id,
                    args_file_path,
                )
            except IndexError:
                print(f"No args found with name {args_name_regex}")
                args_file_path = None
        if weights_not_exist:
            if epoch is None:
                epoch = get_latest_ckpt_epoch(exp_name)
            exp_api.download_model(
                f"{model_artifact_name}_{epoch}", local_artifacts_dir
            )
        if include_session:
            session_not_exist = not os.path.exists(session_checkpoint_path)
            if session_not_exist:
                try:
                    # has to be sorted by step (another attr of an object?!)
                    asset_id = [
                        x
                        for x in exp_api.get_asset_list(asset_type="all")
                        if f"{session_artifact_name}" in x["fileName"]
                    ][0]["assetId"]
                    api.download_experiment_asset(
                        exp_api.id,
                        asset_id,
                        session_checkpoint_path,
                    )
                except IndexError:
                    print(f"No session found with name {session_artifact_name}")
                    session_checkpoint_path = None
    results = {
        "checkpoint_path": model_checkpoint_path,
    }
    if args_file_path is not None:
        results["args_path"] = args_file_path
    if include_session:
        results["session_checkpoint_path"] = session_checkpoint_path
    return results


def log_args(
    exp: comet_ml.Experiment, args: argparse.Namespace, save_path: str
) -> None:
    """Logs the args to the experiment and saves them to a file."""
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    args = {k: v for k, v in sorted(args.items())}
    with open(save_path, "w") as f:
        yaml.dump(
            args,
            f,
            default_flow_style=False,
        )
    exp.log_asset(save_path)


def log_params_to_exp(
    experiment: comet_ml.Experiment, params: dict, prefix: str = ""
) -> None:
    """Logs the given parameter map to the experiment, putting the given prefix before each key."""
    experiment.log_parameters({f"{prefix}/{str(k)}": v for k, v in params.items()})
    if "SLURM_JOBID" in os.environ:
        experiment.log_parameter("slurm_job_id", os.environ["SLURM_JOBID"])


def log_ckpt_to_exp(
    experiment: comet_ml.Experiment, ckpt_path: str, model_name: str
) -> None:
    experiment.log_model(model_name, ckpt_path, overwrite=False)


def create_tracking_exp(
    args: argparse.Namespace,
    exp_disabled: bool = True,
    force_disabled: bool = False,
    project_name: str = "terrain-reconstruction",
) -> comet_ml.Experiment:
    """Creates a Comet.ml experiment if args.resume_exp is False, otherwise resumes the experiment with the given name. Logs the package code."""

    if "COMET_GIT_DIRECTORY" not in os.environ:
        os.environ["COMET_GIT_DIRECTORY"] = str(root_dir)
    disabled = getattr(args, "exp_disabled", exp_disabled) or force_disabled
    exp_init_args = dict(
        api_key=os.environ["COMET_API_KEY"],
        auto_output_logging="simple",
        auto_metric_logging=True,
        auto_param_logging=True,
        log_env_details=True,
        log_env_host=False,
        log_env_gpu=True,
        log_env_cpu=True,
        log_code=False,
        parse_args=True,
        display_summary_level=0,
        disabled=disabled,
    )
    if getattr(args, "resume_exp", False):
        from comet_ml.api import API

        api = API(api_key=os.environ["COMET_API_KEY"])
        exp_api = api.get(f"kirilllzaitsev/{project_name}/{args.exp_name}")
        experiment = comet_ml.ExistingExperiment(
            **exp_init_args, experiment_key=exp_api.id
        )
    else:
        experiment = comet_ml.Experiment(**exp_init_args, project_name=project_name)

    # scripts dir
    for code_file in glob.glob("./*.py"):
        experiment.log_code(code_file)
    log_pkg_code(experiment)

    if not disabled:
        print(
            f'Please leave a note about the experiment at {experiment._get_experiment_url(tab="notes")}'
        )

    return experiment


def log_pkg_code(exp: comet_ml.Experiment, overwrite: bool = False) -> None:
    """Recursively logs the package code to the experiment."""
    import terrain_representation

    pkg_dir = Path(terrain_representation.__file__).parent
    current_dir = os.getcwd()
    os.chdir(pkg_dir)
    for code_file in glob.glob(str(pkg_dir / "**/*.py"), recursive=True):
        code_file = Path(code_file)
        exp.log_code(file_name=code_file.relative_to(pkg_dir), overwrite=overwrite)
    os.chdir(current_dir)


def load_artifacts_from_comet_v2(
    exp_name, save_path_model, save_path_args, comet_api=None
):
    if comet_api is None:
        comet_api = API(api_key=os.environ["COMET_API_KEY"])

    save_dir = os.path.dirname(save_path_model)
    model_ckpt_regex = r".*model_(\d+).pt"
    model_paths = [
        x for x in glob.glob(f"{save_dir}/*.pt") if re.match(model_ckpt_regex, x)
    ]
    model_paths = sorted(
        model_paths, key=lambda x: int(re.match(model_ckpt_regex, x).groups()[0])
    )
    if len(model_paths) > 0:
        model_path = model_paths[-1]
        # rename to save_path_model
        os.rename(model_path, save_path_model)

    do_load_model = not os.path.exists(save_path_model)
    do_load_args = not os.path.exists(save_path_args)
    save_path_dir = os.path.dirname(save_path_model)
    os.makedirs(save_path_dir, exist_ok=True)
    if do_load_model or do_load_args:
        experiment = comet_api.get(
            "kirilllzaitsev",
            project_name="terrain-reconstruction",
            experiment=exp_name,
        )
    if do_load_model:
        model_artifacts = [
            {"id": x["assetId"], "name": f'{x["dir"]}/{x["fileName"]}'}
            for x in experiment.get_asset_list()
            if re.match(model_ckpt_regex, x["fileName"])
        ]
        model_artifacts = sorted(
            model_artifacts,
            key=lambda x: int(re.match(model_ckpt_regex, x["name"]).group(1)),
        )
        artifact_id_model = model_artifacts[-1]["id"]
        print(f"Using {model_artifacts[-1]=} for the model")
        model_binary = experiment.get_asset(artifact_id_model)
        # Save the asset to a file
        with open(save_path_model, "wb") as f:
            f.write(model_binary)
    if do_load_args:
        args_p = r".*args.yaml"
        args_artifacts = [
            {"id": x["assetId"], "name": f'{x["dir"]}/{x["fileName"]}'}
            for x in experiment.get_asset_list()
            if re.match(args_p, x["fileName"])
        ]
        artifact_id_args = args_artifacts[-1]["id"]
        print(f"Using {artifact_id_args=} for the args")
        args_binary = experiment.get_asset(artifact_id_args)
        with open(save_path_args, "wb") as f:
            f.write(args_binary)
    args_yaml = yaml.safe_load(open(save_path_args))
    if "args" in args_yaml:
        args_yaml = args_yaml["args"]
    artifacts = {
        "args": argparse.Namespace(**args_yaml),
        "model_path": save_path_model,
    }
    return artifacts
