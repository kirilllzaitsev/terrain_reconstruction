import argparse
import collections
import copy
import json
import os
import re
import sys
import typing as t
from datetime import datetime
from pathlib import Path
from socket import gethostname

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from terrain_representation import TREP_DATA_DIR, TREP_RESULTS_DIR
from terrain_representation.losses.metrics import (
    compute_abs_errors_btw_point_clouds,
    compute_metrics_for_known_elevated_areas,
    is_elevated_known,
)
from terrain_representation.storage.dataset import PointCloudDataset
from terrain_representation.storage.dataset_torch import (
    PointCloudDataset as PointCloudDatasetTorch,
)
from terrain_representation.storage.dataset_torch import (
    PointCloudDatasetPickle,
    PointCloudDatasetRos,
)
from terrain_representation.utils.callbacks import EarlyStopping
from terrain_representation.utils.comet_utils import (
    create_tracking_exp,
    log_args,
    log_ckpt_to_exp,
    log_params_to_exp,
)
from terrain_representation.utils.detect_elevation import is_elevated
from terrain_representation.utils.loss_utils import (
    compute_occupancy_metrics,
    compute_occupancy_metrics_sparse,
    get_vg_mask_from_logits,
    get_vg_occupancy_mask,
    loss_func_dense,
    loss_func_sparse,
)
from terrain_representation.utils.noisifier import Noisifier
from terrain_representation.utils.pointcloud_utils import (
    cylindrical_to_xyz,
    shift_centered_pts,
)
from terrain_representation.utils.utils import (
    DeviceType,
    TensorOrArrOrList,
    load_from_checkpoint,
    set_seed,
)
from terrain_representation.utils.visualization import (
    PointCloudVisualizer,
    PointCloudVisualizerAsync,
)
from terrain_representation.utils.voxel_grid_utils import convert_vg_to_pcl
from torch import optim
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


def create_tools(args: argparse.Namespace, save_args: bool = True) -> dict:
    """Creates tools for the pipeline:
    - Comet experiment
    - writer
    Logs the arguments and tags to the experiment.
    """
    exp = create_tracking_exp(args, project_name="terrain_reconstruction")
    args.run_name = exp.name
    log_params_to_exp(
        exp,
        vars(args),
        "args",
    )
    log_tags(args, exp)

    log_dir = get_log_dir(args)
    os.makedirs(log_dir, exist_ok=True)

    if save_args:
        log_args(exp, args, f"{log_dir}/train_args.yaml")

    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
    return {
        "exp": exp,
        "writer": writer,
    }


def log_tags(args: argparse.Namespace, exp: comet_ml.Experiment) -> None:
    """Logs tags to the experiment."""

    extra_tags = [str(args.map_dimension)]
    if os.path.exists("/home/kirillz"):
        extra_tags.append("e_remote")
    elif os.path.exists("/cluster"):
        extra_tags.append("e_euler")
    else:
        extra_tags.append("e_local")
    if "anymal" in args.traj_folder:
        extra_tags.append("ds_anymal")
    for k, v in vars(args).items():
        if k in [
            "use_dataloader",
            "use_centroids_for_dist_loss",
            "use_async_vis",
            "use_early_stopping",
            "use_val_ds",
            "do_calc_dist_metrics",
        ]:
            continue
        p = r"^(use_|do_)"
        if re.match(p, k) and v:
            extra_tags.append(re.sub(p, "", k))
        p = r"^(disable_|no_)"
        if re.match(p, k) and v:
            extra_tags.append(f"no_{re.sub(p, '', k)}")
    if args.model_name != "unet":
        extra_tags.append(f"m_{args.model_name}")
    if args.do_overfit:
        extra_tags.append("p_overfit")
    if args.do_optimize:
        extra_tags.append("opt")
        extra_tags.append(f"opt_target_{args.opt_target_metrics}")
    if args.hp_group_num:
        extra_tags.append(f"hp_{args.hp_group_num}")
    if args.do_benchmark:
        extra_tags.append("p_benchmark")
    if "temporal" in args.traj_folder:
        extra_tags.append("ds_temporal")
    if "v11" in args.traj_folder:
        extra_tags.append("ds_synt_and_real")

    tags_to_log = extra_tags
    if len(args.exp_tags) > 0 and args.exp_tags[0] != "":
        tags_to_log += args.exp_tags
    exp.add_tags(tags_to_log)


def init_statistics(
    use_occupancy_loss_with_input=False,
) -> t.Dict[str, collections.deque]:
    """Creates a dictionary with statistics deques."""
    return collections.defaultdict(lambda: collections.deque(maxlen=200))


def init_statistics_elevated(*args, **kwargs) -> t.Dict[str, collections.deque]:
    """Creates a dictionary with statistics deques for elevated samples."""
    stats = init_statistics(*args, **kwargs)
    stats["share_elevated_samples"] = collections.deque(maxlen=200)
    return stats


def init_statistics_elevated_known(*args, **kwargs) -> t.Dict[str, collections.deque]:
    """Creates a dictionary with statistics deques for known elevated samples."""
    stats = init_statistics_elevated(*args, **kwargs)
    stats["num_pts_pred"] = collections.deque(maxlen=200)
    stats["num_pts_gt"] = collections.deque(maxlen=200)
    return stats


def get_log_dir(args: argparse.Namespace) -> str:
    """Infers the log directory based on the arguments."""
    log_base_path = os.path.join(TREP_RESULTS_DIR, args.log_subdir)
    if args.run_name is None:
        log_dir = os.path.join(log_base_path, datetime.now().strftime("%b%d_%H-%M-%S"))
    else:
        log_dir = os.path.join(log_base_path, f"{args.run_name}")
    return log_dir


def create_noisifier(
    args: argparse.Namespace,
    device: t.Any,
    map_dim: torch.Tensor,
    map_resolution: torch.Tensor,
) -> t.Optional[Noisifier]:
    """Creates a noisifier if the do_noisify flag is set."""
    if args.do_noisify:
        noisifier = Noisifier(
            map_dim=map_dim,
            map_resolution=map_resolution,
            device=device,
            num_blob_points=args.noise_num_blob_points,
            noise_scale=args.noise_noise_scale,
            num_blobs=args.noise_num_blobs,
            blob_extent=args.noise_blob_extent,
            noise_prob=args.noise_prob,
            subsampling_ratio_min=args.noise_subsampling_ratio_min,
            subsampling_ratio_max=args.noise_subsampling_ratio_max,
            add_boxes_prob=args.noise_add_boxes_prob,
            add_blobs_prob=args.noise_add_blobs_prob,
        )
    else:
        noisifier = None
    return noisifier


def create_datasets(
    args: argparse.Namespace,
    device: t.Any,
    map_dim: torch.Tensor,
    map_resolution: torch.Tensor,
    noisifier: t.Optional[Noisifier] = None,
    use_real_ds_in_train: bool = False,
    add_test_ds: bool = False,
    ds_portion_from_end_hong_test: float = 1.0,
    ds_portion_from_end_bremg_test: float = 0.2,
    ds_portion_from_end_hong2_test: float = 1.0,
) -> t.Dict[str, PointCloudDataset]:
    """Creates datasets for the pipeline. If the use_dataloader flag is set, the torch dataset is used.
    Returns:
        A dictionary with datasets, where the keys are 'train' and/or 'val'.
    """

    if args.use_dataloader:
        ds_cls = PointCloudDatasetTorch
    else:
        ds_cls = PointCloudDataset
    overfit_config = (
        {
            "num_terrains": args.overfit_num_terrains,
            "num_traj_per_terrain": args.overfit_num_traj_per_terrain,
            "num_scans_in_traj": args.overfit_num_scans_in_traj,
            "traj_idxs": args.overfit_traj_idxs,
            "scan_idxs_in_traj": args.overfit_scan_idxs
            or list(range((args.overfit_num_scans_in_traj or 1))),
        }
        if args.do_overfit
        else None
    )
    robot_params = json.load(
        open(Path(args.traj_folder).parent.parent / "robot_params.json")
    )
    common_ds_kwargs = dict(
        map_dim=map_dim,
        map_resolution=map_resolution,
        device="cpu",
        use_cylindrical_coords=args.use_cylindrical_coords,
        use_sparse=args.use_sparse,
        do_convert_pose_to_quat_and_t=not args.use_sparse,
        mesh_cell_resolution=args.mesh_cell_resolution,
        add_noise_to_real_data=args.do_add_noise_to_real_data,
        overfit_config=overfit_config,
        robot_params=robot_params,
        sequence_length=args.sequence_length,
        step_skip=args.step_skip,
        seq_start=0 if args.do_overfit else None,
        num_pts=getattr(args, "num_pts", None),
    )
    use_labels = getattr(args, "use_seed_loss", False) and not getattr(
        args, "use_seed_loss_unsup", False
    )
    train_dataset = ds_cls(
        base_folder=args.traj_folder,
        **common_ds_kwargs,
        split="train",
        apply_rotation_prob=args.noise_apply_rotation_prob,
        apply_mask_prob=args.noise_apply_mask_prob,
        apply_shift_prob=args.noise_apply_shift_prob,
        noisifier=noisifier,
        use_labels=use_labels,
    )
    common_ds_ros_params = dict(
        device=device,
        map_dim=map_dim,
        map_resolution=map_resolution,
        use_cylindrical_coords=False,
        use_sparse=False,
        do_convert_pose_to_quat_and_t=False,
    )
    if use_real_ds_in_train:
        ds_portion_from_start_hong = 0.8
        ds_portion_from_start_bremg = 0.4
        ds_portion_from_start_hong2 = 0.8
        ds_portion_from_end_hong_test = 1 - ds_portion_from_start_hong
        ds_portion_from_end_bremg_test = 0.05
        ds_portion_from_end_hong2_test = 1 - ds_portion_from_start_hong2

    if use_real_ds_in_train:
        # concat three datasets
        real_ds = []
        if args.real_ds_hong_folder is not None:
            ds_hong = load_hong_ds(
                args.real_ds_hong_folder,
                device=device,
                overfit_config=overfit_config,
                noisifier=noisifier,
                sequence_length=args.sequence_length,
                step_skip=args.step_skip,
                ds_portion_from_start=ds_portion_from_start_hong,
            )
            real_ds.append(ds_hong)
        if args.real_ds_bremgarten_folder is not None:
            ds_bremg = PointCloudDatasetRos(
                base_folder=args.real_ds_bremgarten_folder,
                **common_ds_ros_params,
                ds_portion_from_start=ds_portion_from_start_bremg,
                sequence_length=args.sequence_length,
                step_skip=args.step_skip,
            )
            real_ds.append(ds_bremg)
        if args.real_ds_hong2_folder is not None:
            ds_hong2 = PointCloudDatasetRos(
                base_folder=args.real_ds_hong2_folder,
                **common_ds_ros_params,
                ds_portion_from_start=ds_portion_from_start_hong2,
                sequence_length=args.sequence_length,
                step_skip=args.step_skip,
            )
            real_ds.append(ds_hong2)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, *real_ds])
    datasets = {"train": train_dataset}
    if add_test_ds:
        val_dataset = ds_cls(
            base_folder=args.traj_folder,
            **common_ds_kwargs,
            split="val",
        )
        datasets["val"] = val_dataset
        if args.real_ds_hong_folder is not None:
            if not os.path.exists(args.real_ds_hong_folder):
                print(f"WARNING: {args.real_ds_hong_folder} does not exist")
            else:
                ds_hong = load_hong_ds(
                    args.real_ds_hong_folder,
                    device=device,
                    overfit_config=overfit_config,
                    ds_portion_from_end=ds_portion_from_end_hong_test,
                )
                datasets["test_hong"] = ds_hong
        if args.real_ds_bremgarten_folder is not None:
            if not os.path.exists(args.real_ds_bremgarten_folder):
                print(f"WARNING: {args.real_ds_bremgarten_folder} does not exist")
            else:
                ds_bremg = PointCloudDatasetRos(
                    **common_ds_ros_params,
                    base_folder=args.real_ds_bremgarten_folder,
                    ds_portion_from_end=ds_portion_from_end_bremg_test,
                )
                datasets["test_bremgarten"] = ds_bremg
        real_ds_hong2_folder = getattr(args, "real_ds_hong2_folder", None)
        if real_ds_hong2_folder is not None:
            if not os.path.exists(real_ds_hong2_folder):
                print(f"WARNING: {real_ds_hong2_folder} does not exist")
            else:
                ds_hong2 = PointCloudDatasetRos(
                    **common_ds_ros_params,
                    base_folder=args.real_ds_hong2_folder,
                    ds_portion_from_end=ds_portion_from_end_hong2_test,
                )
                datasets["test_hong2"] = ds_hong2
    for k, ds in datasets.items():
        print(f"{k} dataset: {len(ds)=}")
        assert len(ds) > 0, f"Dataset {k} has zero samples"
    return datasets


def load_hong_ds(real_ds_hong_folder, **kwargs):
    ds_test = PointCloudDatasetPickle(
        base_folder=real_ds_hong_folder,
        file_suffix="real.pkl",
        **kwargs,
    )
    ds_test.load_new_trajectories()
    return ds_test


def create_visualizer(
    args: argparse.Namespace,
) -> t.Union[PointCloudVisualizer, PointCloudVisualizerAsync]:
    """Creates a visualizer for the pipeline. If the use_async_vis flag is set, the async visualizer is used."""
    common_vis_kwargs = dict(
        extra_axes_params=dict(
            interactive=args.plots_interactive,
            resetcam=True,
            elevation=-10,
            offscreen=args.plots_disabled,
            zoom=args.plots_zoom,
        ),
        figure_name="sample",
        input_in_cylindrical=args.use_cylindrical_coords,
        use_pyplot=args.use_ddp,
        pts_radius=10,
        # pts_radius=1 if args.use_pc_norm else 10,
    )
    if args.use_async_vis:
        visualizer = PointCloudVisualizerAsync(
            **common_vis_kwargs,
        )
    else:
        visualizer = PointCloudVisualizer(
            **common_vis_kwargs,
        )

    return visualizer


def log_statistics_and_assemble_log_msg(
    writer: SummaryWriter,
    statistics: dict,
    epoch: int,
    pad: int,
    heading: str = "",
    stat_key_prefix: t.Optional[str] = None,
) -> str:
    """Logs statistics to the tensorboard and the console."""
    for dataset_key, elem in statistics.items():
        for stat_key, stat in elem.items():
            if len(stat) > 0:
                stat = list(stat)
                if isinstance(stat[0], (torch.Tensor, np.ndarray)):
                    mean_stat = get_mean_of_arr(stat).item()
                else:
                    mean_stat = np.mean(stat)
            else:
                mean_stat = 0
            if stat_key_prefix is not None:
                writer_key = f"{dataset_key}/{stat_key_prefix}/{stat_key}"
            else:
                writer_key = f"{dataset_key}/{stat_key}"
            writer.add_scalar(writer_key, mean_stat, epoch)
            heading += f"""{f'{dataset_key}/{stat_key}:':>{pad}} {mean_stat:.4f}\n"""
    return heading


def get_mean_of_arr(all_values_in_epoch):
    if (
        len(all_values_in_epoch) > 0
        and isinstance(all_values_in_epoch[0], torch.Tensor)
        and all_values_in_epoch[0].ndim > 0
    ):
        all_values_in_epoch = [torch.mean(v) for v in all_values_in_epoch]
    metric_avg = torch.mean(torch.tensor(list(all_values_in_epoch)))
    return metric_avg


def print_main_log_msg(x):
    print(x)
    print("-" * 20)


def calc_epoch_duration(start_time: datetime) -> float:
    end_time = datetime.now()
    epoch_duration = (end_time - start_time).total_seconds()
    return epoch_duration


def get_avg_metric(maes: list) -> float:
    return get_quantile_metric(maes, 0.5)


def get_quantile_metric(maes: list, q: float) -> float:
    return torch.quantile(torch.stack(maes), q).item()


def set_network_mode(network: nn.Module, key: str) -> None:
    if key == "train":
        network.train()
    else:
        network.eval()


def load_from_ckpt(
    checkpoint_path: str,
    device: DeviceType,
    trep_net: nn.Module,
    optimizer: t.Any = None,
    scheduler: t.Any = None,
) -> dict:
    state_dicts = torch.load(
        checkpoint_path,
        map_location=device,
    )
    trep_net.load_state_dict(state_dicts["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state_dicts["optimizer"])
        for g in optimizer.param_groups:
            g["lr"] = 0.005
    if scheduler is not None:
        scheduler.load_state_dict(state_dicts["scheduler"])

    return {
        "model": trep_net,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def log_artifacts(
    exp: comet_ml.Experiment, artifacts: dict, log_dir: str, epoch: int
) -> str:
    """Logs the training artifacts to the experiment and saves the model and session to the log directory."""

    save_path_model = os.path.join(log_dir, f"model_{epoch}.pt")
    save_path_session = os.path.join(log_dir, f"session_{epoch}.pt")
    torch.save(
        {
            "model": artifacts["network"].state_dict(),
        },
        save_path_model,
    )
    torch.save(
        {
            "optimizer": artifacts["optimizer"].state_dict(),
            "scheduler": artifacts["scheduler"].state_dict(),
            "epoch": epoch,
        },
        save_path_session,
    )
    log_ckpt_to_exp(exp, save_path_model, "ckpt")
    log_ckpt_to_exp(exp, save_path_session, "ckpt")
    return save_path_model


def log_net_params(exp: comet_ml.Experiment, network: nn.Module) -> None:
    num_model_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_model_params}")
    exp.log_parameters({"model/num_params": num_model_params})


def print_time_stats(epoch_durations: list, epoch_duration: float) -> None:
    print(f"Epoch time (s): {epoch_duration}")
    print(f"Mean epoch time (s): {np.mean(epoch_durations)}")


def is_dist_loss_used(args: argparse.Namespace, epoch: int) -> bool:
    return (
        args.use_mae_loss or args.use_chamfer_loss
    ) and epoch >= args.start_mae_from_n_epoch


@torch.no_grad()
def compute_dist_metrics_and_elevated_idxs(
    args,
    pts_gts,
    pts_preds,
    pose,
    area_maps,
) -> t.Tuple[dict, dict]:
    """Compute metrics for a batch of samples. Three types of metrics are computed:
    1. Metrics for all samples in a batch
    2. Metrics for samples detected as elevated by heuristic
    3. Metrics for samples for known elevated areas. Distance-based metrics are computed for the points that belong exclusively to the elevated areas while other metrics (e.g., F1) are computed for the whole point cloud.
    """

    maes = []
    maes_height = []
    chamfer_dists = []
    mae_height_granular = []

    if args.use_cylindrical_coords:
        pts_gts = cylindrical_to_xyz(pts_gts)
        pts_preds = cylindrical_to_xyz(pts_preds)

    num_samples_pred = len(pts_preds)
    num_samples_gt = len(pts_gts)
    if num_samples_pred != num_samples_gt:
        print(f"Warning: {num_samples_pred=} is different from the {num_samples_gt=}")
    for take_idx in range(num_samples_pred):
        pts_gt = pts_gts[take_idx]
        pts_pred = pts_preds[take_idx]
        pts_gt = pts_gt.to(pts_pred.device)

        if len(pts_pred) > 20_000:
            print(
                f"Warning: {len(pts_pred)=} which is too much to compute distance metrics. The points will be downsampled to 20k."
            )
            pts_pred = pts_pred[torch.randperm(len(pts_pred))[:20_000]]

        abs_errors = compute_abs_errors_btw_point_clouds(pts_pred, pts_gt)
        maes.append(abs_errors["mae"])
        maes_height.append(abs_errors["mae_height"])
        mae_height_granular.append(abs_errors["mae_height_granular"])
        chamfer_dists.append(abs_errors["chamfer_dist"])

    elevated_idxs = get_elevated_idxs(args, pts_gts)

    if area_maps is not None and len(area_maps[0]) > 0:
        known_elevated_idxs = get_known_elevated_idxs(
            args, pose, area_maps, pts_gts, pts_preds
        )
    else:
        known_elevated_idxs = []
    known_elevated_idxs = []

    return {
        "elevated_idxs": elevated_idxs,
        "known_elevated_idxs": known_elevated_idxs,
    }, {
        "maes": maes,
        "maes_height": maes_height,
        "chamfer_dists": chamfer_dists,
        "mae_height_granular": mae_height_granular,
    }


def convert_gt_and_pred_voxel_grids_to_pts(
    map_dim: torch.Tensor,
    map_resolution: torch.Tensor,
    net_outputs: dict,
    target: torch.Tensor,
    occupancy_threshold: float,
    use_joint_mask=True,
) -> t.Tuple[list, list]:
    """Helps to convert ground truth and predicted voxel grids to point clouds."""

    pts_gts = []
    pts_preds = []
    for take_idx in range(len(target)):
        vg_gt = target[take_idx].clone()
        if isinstance(net_outputs, dict):
            vg_pred = net_outputs["centroids"][take_idx].clone().detach()
            occupancy_logits = (
                net_outputs["occupancy_logits"][take_idx].clone().detach()
            )
            pred_mask = get_vg_mask_from_logits(occupancy_logits, occupancy_threshold)
        else:
            vg_pred = net_outputs[take_idx].clone().detach()
            pred_mask = get_vg_occupancy_mask(vg_pred)

        gt_mask = get_vg_occupancy_mask(vg_gt)
        if use_joint_mask:
            centroid_mask = gt_mask & pred_mask
            gt_mask = centroid_mask
            pred_mask = centroid_mask
        pts_gt = convert_vg_to_pcl(vg_gt, map_dim, map_resolution, gt_mask)
        pts_pred = convert_vg_to_pcl(vg_pred, map_dim, map_resolution, pred_mask)

        pts_gts.append(pts_gt)
        pts_preds.append(pts_pred)
    return pts_gts, pts_preds


def get_known_elevated_idxs(
    args: argparse.Namespace,
    pose,
    area_maps: list,
    pts_gts: TensorOrArrOrList,
    pts_preds: TensorOrArrOrList,
) -> list:
    """Infers the indices of the samples that contain areas known to be elevated (e.g., piles).
    Since the points are centered, they are shifted back to the original coordinates. Elevation is detected based on the ground truth point cloud and the area map.
    """

    known_elevated_idxs = []
    metrics_known_elevated_samples = []
    for take_idx in range(len(pts_gts)):
        pts_gt = pts_gts[take_idx]
        pts_pred = pts_preds[take_idx]
        # metrics for known elevated areas (e.g., pile)
        sensor_translation = pose[take_idx, :3, 3].to(pts_gt.device)
        shifted_pts_pred = shift_centered_pts(
            pts_pred,
            sensor_translation,
        )

        shifted_pts_gt = shift_centered_pts(
            pts_gt,
            sensor_translation,
        )

        if is_elevated_known(
            shifted_pts_gt,
            area_maps[take_idx],
            elev_area_names=["pile"],
        ):
            known_elevated_idxs.append(take_idx)
            metrics_known_elevated_samples.append(
                compute_metrics_for_known_elevated_areas(
                    shifted_pts_pred,
                    shifted_pts_gt,
                    area_maps[take_idx],
                    elev_area_names=["pile"],
                )
            )

    return known_elevated_idxs


def get_elevated_idxs(args: argparse.Namespace, pts_gts: TensorOrArrOrList) -> list:
    elevated_idxs = []
    for take_idx in range(len(pts_gts)):
        pts_gt = pts_gts[take_idx]
        # apply heuristic to detect elevated areas
        if is_elevated(
            pts_gt,
            flat_region_z_lower_limit=args.flat_region_z_lower_limit,
            flat_region_z_upper_limit=args.flat_region_z_upper_limit,
        ):
            elevated_idxs.append(take_idx)
    return elevated_idxs


def update_statistics(
    statistics: dict,
    maes: list,
    maes_height: list,
    mae_height_granular: list,
    chamfer_dists: list,
    take_idxs: t.Optional[list] = None,
    losses: t.Optional[dict] = None,
) -> dict:
    """Helps to update the statistics with the metrics computed for a batch of samples. The metrics to be added are averaged. If take_idxs is provided, only the metrics for the samples with the corresponding indices are added."""

    if take_idxs is not None:
        maes = [maes[i] for i in take_idxs]
        maes_height = [maes_height[i] for i in take_idxs]
        chamfer_dists = [chamfer_dists[i] for i in take_idxs]
        mae_height_granular = [mae_height_granular[i] for i in take_idxs]

    # flatten mae_height_granular
    mae_height_granular = [elem for sublist in mae_height_granular for elem in sublist]
    if len(mae_height_granular) > 0:
        statistics["mae_height_p50"].append(
            get_quantile_metric(mae_height_granular, q=0.5)
        )
        statistics["mae_height_p75"].append(
            get_quantile_metric(mae_height_granular, q=0.75)
        )
        statistics["mae_height_p90"].append(
            get_quantile_metric(mae_height_granular, q=0.9)
        )
        statistics["mae_height_p100"].append(
            get_quantile_metric(mae_height_granular, q=1.0)
        )

    statistics["chamfer_dist"].append(get_avg_metric(chamfer_dists))
    statistics["mae"].append(get_avg_metric(maes))
    statistics["mae_height"].append(get_avg_metric(maes_height))
    if losses is not None:
        statistics = update_statistics_with_losses(statistics, losses)
    return statistics


def update_statistics_with_losses(
    statistics: dict,
    losses: dict,
) -> dict:
    for n, elem in losses.items():
        if isinstance(elem, torch.Tensor):
            elem = elem.detach()
        statistics[n].append(elem)
    return statistics


@torch.no_grad()
def update_statistics_with_idxs(
    statistics_by_ds: dict,
    target_scan: torch.Tensor,
    net_outputs: dict,
    elevated_stats: dict,
    common_loss_kwargs: dict,
    known_elevated_idxs: list,
    is_sparse: bool = False,
    map_dim: t.Optional[torch.Tensor] = None,
    map_resolution: t.Optional[torch.Tensor] = None,
    mesh: t.Optional[torch.Tensor] = None,
) -> None:
    """Similar to update_statistics, but the statistics are updated only for the samples with the specified indices."""

    common_loss_kwargs = common_loss_kwargs.copy()
    is_synthetic = common_loss_kwargs.pop("is_synthetic")
    if is_synthetic is not None:
        is_synthetic = [
            x for i, x in enumerate(is_synthetic) if i in known_elevated_idxs
        ]
    if len(known_elevated_idxs) != 0:
        # compute loss only for known elevated samples
        if is_sparse:
            loss_data = {
                "layer_outputs": net_outputs["layer_outputs"],
                "layer_targets": net_outputs["layer_targets"],
            }
            losses = loss_func_sparse(
                data=loss_data,
                **common_loss_kwargs,
            )
            occupancy_metrics = compute_occupancy_metrics_sparse(
                net_outputs["output"],
                target_scan,
                net_outputs["kernel_map_out"],
                batch_idxs=known_elevated_idxs,
            )
        else:
            target_mesh = mesh[known_elevated_idxs] if mesh is not None else None

            loss_data = {
                "occupancy_logits": net_outputs["occupancy_logits"][
                    known_elevated_idxs
                ],
                "occupancy": net_outputs["occupancy"][known_elevated_idxs],
                "centroids": net_outputs["centroids"][known_elevated_idxs],
                "target": target_scan[known_elevated_idxs],
                "map_dim": map_dim,
                "map_resolution": map_resolution,
                "target_mesh": target_mesh,
            }
            losses = loss_func_dense(
                data=loss_data,
                **common_loss_kwargs,
            )
            occupancy_metrics = compute_occupancy_metrics(
                net_outputs["occupancy"][known_elevated_idxs],
                (
                    target_scan[known_elevated_idxs]
                    if target_mesh is None
                    else target_mesh
                ),
            )

        for n, elem in occupancy_metrics.items():
            if isinstance(elem, torch.Tensor):
                elem = elem.detach()
            statistics_by_ds[n].append(elem)
        update_statistics(
            statistics_by_ds,
            elevated_stats["maes"],
            elevated_stats["maes_height"],
            elevated_stats["mae_height_granular"],
            elevated_stats["chamfer_dists"],
            take_idxs=known_elevated_idxs,
            losses=losses,
        )


def get_model(args, device, map_dim, map_resolution, mini_batch_size, **kwargs):
    if args.model_name == "unet":
        if args.use_sparse:
            from terrain_representation.modules.sparse_trep import (
                TerrainCompletionNetSparse,
            )

            network = TerrainCompletionNetSparse(
                map_dim,
                map_resolution,
            ).to(device)
        else:
            from terrain_representation.modules.dense_trep import TerrainNetLayerNorm

            network = TerrainNetLayerNorm(
                map_dim,
                map_resolution,
                device_handle=device,
                use_skip_conn=args.use_skip_conn,
                use_prev_pred_as_input=getattr(args, "use_extended_input", False),
                norm_layer_name=args.norm_layer_name,
                use_dropout=args.use_dropout,
                dropout_rate=args.dropout_rate,
                hidden_dims_scaler=args.hidden_dims_scaler,
            ).to(device)
            network.initialize(mini_batch_size)

    elif args.model_name == "pointattn":
        from pointattn.model import Model as PointAttnModel

        network = PointAttnModel(
            step1=2,
            step2=4,
            encoder_channel=16,
            refine_channel=64,
        )
    elif args.model_name == "pflow":
        from pflow.models.networks import PointFlow

        batch_norm = True if args.mini_batch_size > 1 else False
        use_large = kwargs.get("use_large_model", False)
        assert args.mini_batch_size > 1, "BatchNorms require mini_batch_size > 1"
        dims = "128-256-512-512-1024"
        if use_large:
            dims += "-2048"
        pflow_args = argparse.Namespace(
            atol=1e-05,
            batch_norm=batch_norm,
            bn_lag=0,
            dims=dims,
            distributed=args.use_ddp,
            entropy_weight=0.0,
            input_dim=3,
            latent_dims=dims,
            latent_num_blocks=1,
            layer_type="concatsquash",
            nonlinearity="tanh",
            num_blocks=1,
            prior_weight=0.0,
            recon_weight=1.0,
            rtol=1e-05,
            solver="dopri5",
            sync_bn=False,
            te_max_sample_points=2048,
            time_length=0.5,
            tr_max_sample_points=2048,
            train_T=True,
            use_adjoint=True,
            use_deterministic_encoder=True,
            use_latent_flow=False,
            zdim=512,
        )

        if args.pflow_use_gen:
            pflow_args.use_latent_flow = True
            pflow_args.use_deterministic_encoder = False
            pflow_args.prior_weight = 1.0
            pflow_args.recon_weight = 1.0
            pflow_args.entropy_weight = 1.0

        network = PointFlow(pflow_args)

    elif args.model_name == "seedformer":
        from seedformer.model import seedformer_dim128

        use_large = kwargs.get("use_large_model", False)
        if use_large:
            network = seedformer_dim128(
                feat_dim=1024, embed_dim=256, n_knn=20, up_factors=[1, 2, 2]
            )
        else:
            network = seedformer_dim128(
                feat_dim=512, embed_dim=128, n_knn=20, up_factors=[1, 2, 2]
            )
    elif args.model_name == "snowflakenet":
        from pointr.models.SnowFlakeNet import SnowFlakeNet

        config = argparse.Namespace()
        config.dim_feat = 512
        config.num_pc = 256
        config.num_p0 = 512
        config.radius = 1
        config.up_factors = [2, 2]
        network = SnowFlakeNet(config)
    elif args.model_name == "foldingnet":
        from pointr.models.FoldingNet import FoldingNet

        assert args.gt_pts_scaler == 2, "FoldingNet requires 2x2048 GT points"
        config = argparse.Namespace()
        config.num_pred = 2048 * args.gt_pts_scaler
        config.encoder_channel = 1024 * 2
        network = FoldingNet(config)
    elif args.model_name == "adapointr":
        from pointr.models.AdaPoinTr import AdaPoinTr

        config = argparse.Namespace()
        config.num_query = 512
        config.num_points = 2048
        config.center_num = [512, 256]
        config.global_feature_dim = 1024
        config.encoder_type = "graph"
        config.decoder_type = "fc"
        config.encoder_config = argparse.Namespace(
            **{
                "embed_dim": 384,
                "depth": 6,
                "num_heads": 6,
                "k": 8,
                "n_group": 2,
                "mlp_ratio": 2.0,
                "block_style_list": [
                    "attn-graph",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                ],
                "combine_style": "concat",
            }
        )
        config.decoder_config = argparse.Namespace(
            **{
                "embed_dim": 384,
                "depth": 8,
                "num_heads": 6,
                "k": 8,
                "n_group": 2,
                "mlp_ratio": 2.0,
                "self_attn_block_style_list": [
                    "attn-graph",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                ],
                "self_attn_combine_style": "concat",
                "cross_attn_block_style_list": [
                    "attn-graph",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                ],
                "cross_attn_combine_style": "concat",
            }
        )
        network = AdaPoinTr(config)
    elif args.model_name == "grnet":
        assert args.use_pc_norm, "GRNet requires normalized point clouds"
        from pointr.models.GRNet import GRNet

        config = argparse.Namespace(
            **{
                "num_pred": 4096 // 2,
                "gridding_loss_scales": 64,
                "gridding_loss_alphas": 0.1,
            }
        )
        network = GRNet(config)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    network.to(device)

    return network


def prepare_pipe(rank, world_size, args, tools):
    if args.use_ddp:
        gpus_per_node = int(
            os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count())
        )
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")
        ddp_setup(rank, world_size, master_port=args.master_port)
        print(
            f"Hello from rank {rank} of {world_size - 1} on {gethostname()} where there are"
            f" {gpus_per_node} allocated GPUs per node.",
            flush=True,
        )
        if rank == 0:
            print(f"Group initialized? {dist.is_initialized()}", flush=True)
        torch.cuda.set_device(local_rank)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
    set_seed(args.seed)

    is_main_process = rank == 0
    if not is_main_process:
        args = copy.deepcopy(args)
        args.exp_disabled = True
        args.vis_disabled = True

    external_tools = True
    if tools is None:
        external_tools = False
        tools = create_tools(args)
    exp = tools["exp"]
    writer = tools["writer"]
    log_dir = writer.log_dir

    if not args.exp_disabled:
        args.run_name = exp.name

    device = f"cuda:{rank}" if "cuda" in args.device else "cpu"

    map_dim = torch.tensor(
        [args.map_dimension, args.map_dimension, args.map_dimension]
    ).to(device)
    map_resolution = torch.tensor(
        [args.map_resolution_xy, args.map_resolution_xy, args.map_resolution_z]
    ).to(device)

    noisifier = create_noisifier(args, "cpu", map_dim, map_resolution)
    datasets = create_datasets(
        args,
        "cpu",
        map_dim,
        map_resolution,
        noisifier=noisifier,
        use_real_ds_in_train=args.use_real_ds_in_train,
        add_test_ds=args.use_val_ds and not args.do_overfit,
    )

    mini_batch_size = min([args.mini_batch_size] + [len(d) for d in datasets.values()])

    network = get_model(
        args,
        device,
        map_dim,
        map_resolution,
        mini_batch_size,
        use_large_model=args.use_large_model,
    )

    if args.use_ddp:
        network_ddp = DDP(network, device_ids=[local_rank])
    else:
        network_ddp = network

    log_net_params(exp, network)

    # Scale the learning rate based on the batch size
    scaled_lr = postprocess_lr(world_size, args, mini_batch_size)

    optimizer = optim.Adam(
        network_ddp.parameters(),
        lr=scaled_lr,
    )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    if args.checkpoint_path:
        load_from_checkpoint(
            ckpt_path=os.path.join(TREP_DATA_DIR, "models", args.checkpoint_path),
            network=network,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    vis_enabled = args.vis_enabled and is_main_process
    if is_main_process:
        visualizer = create_visualizer(args)
    else:
        visualizer = None

    statistics = dict()
    statistics_elevated = dict()
    statistics_elevated_known = dict()

    epoch_durations = []
    history = {k: collections.defaultdict(list) for k in datasets.keys()}
    start_calc_metrics_from_n_epoch = (
        args.start_dist_metrics_from_n_epoch
        if args.map_dimension < 96
        else max(args.start_dist_metrics_from_n_epoch, 10)
    )
    es_patience = (
        args.early_stop_n_epochs / args.log_every_n_epochs
        if args.use_early_stopping
        else np.inf
    )
    early_stop = EarlyStopping(patience=es_patience, delta=args.early_stop_delta)

    save_artifacts = {
        "network": network,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

    if is_main_process:
        print(f"### CLI command:\n{' '.join(sys.argv)}")
        print("### Args:")
        print(yaml.dump(args, default_flow_style=False))
        print(f"### Log dir: {log_dir}")
        print(f"Original lr: {args.lr=}\nRescaled lr: {scaled_lr=}")

    return {
        "args": args,
        "is_main_process": is_main_process,
        "external_tools": external_tools,
        "exp": exp,
        "writer": writer,
        "log_dir": log_dir,
        "device": device,
        "map_dim": map_dim,
        "map_resolution": map_resolution,
        "datasets": datasets,
        "mini_batch_size": mini_batch_size,
        "network": network,
        "network_ddp": network_ddp,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "vis_enabled": vis_enabled,
        "visualizer": visualizer,
        "statistics": statistics,
        "statistics_elevated": statistics_elevated,
        "statistics_elevated_known": statistics_elevated_known,
        "epoch_durations": epoch_durations,
        "history": history,
        "start_calc_metrics_from_n_epoch": start_calc_metrics_from_n_epoch,
        "early_stop": early_stop,
        "save_artifacts": save_artifacts,
    }


def postprocess_lr(world_size, args, mini_batch_size):
    if args.do_rescale_lr:
        baseline_batch_size = 16
        scaled_lr = args.lr * (mini_batch_size / baseline_batch_size)
    else:
        scaled_lr = args.lr
    scaled_lr *= np.sqrt(world_size)
    return scaled_lr


def poke_visualizer(exp, visualizer, save_dir=None):
    while not visualizer.res_queue.empty():
        vis_res = visualizer.res_queue.get()
        exp.log_image(
            vis_res["fig"],
            vis_res["log_img_name"],
            step=vis_res["step"],
        )
        if save_dir is not None:
            filename = f"{vis_res['log_img_name']}.png"
            p = r".*/(.*)/(.*)"
            if re.match(p, filename):
                ds_name, sample_name = re.match(p, filename).groups()
                filename = f"{ds_name}_{sample_name}"

            save_path = os.path.join(save_dir, filename)
            plt.imsave(save_path, vis_res["fig"])
        if "step_0" in vis_res["log_img_name"] and "train" in vis_res["log_img_name"]:
            exp.log_image(
                vis_res["fig"],
                "train/sample",
                step=vis_res["step"],
            )


def ddp_setup(rank, world_size, master_port=12355):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(master_port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
