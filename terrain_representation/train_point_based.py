"""Pipeline for training the dense version of the model for terrain reconstruction.
Conventions:
- elevated areas - areas determined by a heuristic to be elevated
- known elevated areas - areas that are known to be elevated (e.g., pile) and are annotated in the area map. The latter is obtained from the terrain synthesis pipeline.
"""

import argparse
import functools
import typing as t
from datetime import datetime

import comet_ml
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# TREP
from terrain_representation.losses.metrics import compute_chamfer_dist
from terrain_representation.storage.dataset_adapters import (
    PointCloudDatasetAdapterPointAttn,
)
from terrain_representation.storage.dataset_torch import custom_collate_fn
from terrain_representation.utils import parse_trep_args
from terrain_representation.utils.comet_utils import root_dir
from terrain_representation.utils.data_utils import get_dataloaders
from terrain_representation.utils.loss_seeds import get_seed_discrepancy_loss
from terrain_representation.utils.loss_utils import compute_occupancy_metrics
from terrain_representation.utils.pipeline_utils import (
    calc_epoch_duration,
    compute_dist_metrics_and_elevated_idxs,
    get_mean_of_arr,
    init_statistics,
    init_statistics_elevated,
    init_statistics_elevated_known,
    log_artifacts,
    log_statistics_and_assemble_log_msg,
    poke_visualizer,
    prepare_pipe,
    print_main_log_msg,
    print_time_stats,
    set_network_mode,
    update_statistics,
    update_statistics_with_losses,
)
from terrain_representation.utils.pointcloud_utils import denormalize_pcl
from terrain_representation.utils.profiling_utils import profile_func
from terrain_representation.utils.utils import transfer_batch_to_device, update_args
from terrain_representation.utils.vis_utils import vis_gen_based, vis_point_based
from terrain_representation.utils.voxel_grid_utils import convert_pts_to_vg
from torch.distributed import destroy_process_group
from tqdm.auto import tqdm

try:
    from pointr.extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
except ImportError:
    print("ChamferDistanceL1 and ChamferDistanceL2 are not available")


def point_net_forward(args, network, key, data_dict, do_compute_loss=True, **kwargs):
    target = data_dict["gt"].clone()

    measured = data_dict["measured"].clone()

    is_training = key == "train"
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        if args.model_name == "pointattn":
            net_outputs = network(
                measured.transpose(2, 1).contiguous(),
                target,
                is_training=is_training,
                do_compute_loss=do_compute_loss,
            )
            if do_compute_loss:
                losses = {
                    "loss": net_outputs["loss"],
                    "loss_coarse": net_outputs["loss_coarse"],
                    "loss_fine": net_outputs["loss_fine"],
                }
            if is_training:
                pcds_pred = net_outputs["fine"]
            else:
                pcds_pred = net_outputs["out2"]
        elif args.model_name == "adapointr":
            net_outputs = network(measured)
            pcds_pred = net_outputs["pred_fine"]
            if do_compute_loss:
                losses = network.get_loss(net_outputs, target)
        elif args.model_name == "pflow":
            assert (
                "optimizer" in kwargs and "step" in kwargs and "writer" in kwargs
            ), f"Missing {kwargs=} for PointFlow"
            optimizer = kwargs["optimizer"] if is_training else None
            losses = network(measured, optimizer)

            writer = kwargs["writer"]
            if writer is not None:
                step = kwargs["step"]
                for k, v in losses.items():
                    if not args.pflow_use_gen and (
                        "entropy" in k or "prior_nats" in k or "prior" in k
                    ):
                        continue
                    writer.add_scalar(f"{key}/{k}", v, step)

            pcds_pred = torch.zeros(len(measured), 0, 3).to(measured.device)
        elif args.model_name == "foldingnet":
            pcds_pred = network(measured)
            if do_compute_loss:
                losses = network.get_loss(pcds_pred, target)
        elif args.model_name == "seedformer":
            from seedformer.utils.loss_utils import get_loss

            out = network(measured)
            pcds_pred_multires = out["pred_pcds"]

            pcds_pred = pcds_pred_multires[-1]
            if do_compute_loss:
                loss_total, losses_list, gts = get_loss(
                    pcds_pred_multires, measured, target, sqrt=True
                )
                losses = {
                    "loss": loss_total,
                    "loss_coarse": losses_list[0],
                    "loss_fine": losses_list[3],
                    "cdc": losses_list[0],
                    "cd1": losses_list[1],
                    "cd2": losses_list[2],
                    "cd3": losses_list[3],
                    "partial_matching": losses_list[4],
                }
        elif args.model_name == "snowflakenet":
            pcds_pred_multires = network(measured)
            if do_compute_loss:
                losses = network.get_loss(pcds_pred_multires, target)
            pcds_pred = pcds_pred_multires[-1]
        elif args.model_name == "grnet":
            coarse_pcl, fine_pcl = network(measured)
            if do_compute_loss:
                loss_coarse = compute_chamfer_dist(coarse_pcl, target)
                loss_fine = compute_chamfer_dist(fine_pcl, target)
                loss = loss_coarse + loss_fine
                losses = {
                    "loss": loss,
                    "loss_coarse": loss_coarse,
                    "loss_fine": loss_fine,
                }
            pcds_pred = fine_pcl
        else:
            raise ValueError(f"Unknown model name: {args.model_name}")

    if not do_compute_loss:
        losses = {}

    res = {
        "losses": losses,
        "preds": pcds_pred,
    }
    if args.model_name == "seedformer":
        res["out"] = out

    return res


def main(
    rank: int, world_size: int, args: argparse.Namespace, tools: t.Optional[dict] = None
) -> dict:
    """Main pipeline for training the dense version of the model for terrain reconstruction.
    Args:
        args: Parsed arguments.
        tools: Optional dictionary with pipeline tools (e.g., Comet experiment, tensorboard writer).
    Returns:
        history: Dictionary with training and validation statistics.
    """
    prepare_pipe_res = prepare_pipe(rank, world_size, args, tools)
    args = prepare_pipe_res["args"]
    is_main_process = prepare_pipe_res["is_main_process"]
    external_tools = prepare_pipe_res["external_tools"]
    exp = prepare_pipe_res["exp"]
    writer = prepare_pipe_res["writer"]
    log_dir = prepare_pipe_res["log_dir"]
    device = prepare_pipe_res["device"]
    map_dim = prepare_pipe_res["map_dim"]
    map_resolution = prepare_pipe_res["map_resolution"]
    datasets = prepare_pipe_res["datasets"]
    mini_batch_size = prepare_pipe_res["mini_batch_size"]
    network = prepare_pipe_res["network"]
    network_ddp = prepare_pipe_res["network_ddp"]
    optimizer = prepare_pipe_res["optimizer"]
    scheduler = prepare_pipe_res["scheduler"]
    vis_enabled = prepare_pipe_res["vis_enabled"]
    visualizer = prepare_pipe_res["visualizer"]
    statistics = prepare_pipe_res["statistics"]
    statistics_elevated = prepare_pipe_res["statistics_elevated"]
    statistics_elevated_known = prepare_pipe_res["statistics_elevated_known"]
    epoch_durations = prepare_pipe_res["epoch_durations"]
    history = prepare_pipe_res["history"]
    start_calc_metrics_from_n_epoch = prepare_pipe_res[
        "start_calc_metrics_from_n_epoch"
    ]
    early_stop = prepare_pipe_res["early_stop"]
    save_artifacts = prepare_pipe_res["save_artifacts"]

    for k, ds in datasets.items():
        datasets[k] = PointCloudDatasetAdapterPointAttn(
            ds,
            gt_pts_scaler=args.gt_pts_scaler,
            use_pc_norm=args.use_pc_norm,
            use_randperm=args.use_randperm,
        )
    dataloaders = get_dataloaders(
        args, datasets, mini_batch_size, collate_fn=custom_collate_fn
    )

    for epoch in tqdm(range(args.num_epochs), desc="Epochs", total=args.num_epochs):
        start_time = datetime.now()
        is_last_epoch = epoch == args.num_epochs - 1
        for key, dl in dataloaders.items():

            is_test_epoch = "test" in key
            if is_test_epoch and (epoch + 1) % args.test_freq_epoch != 0:
                continue

            do_calc_and_log_stats = (
                (epoch + 1) % args.log_every_n_epochs == 0
                or is_last_epoch
                or is_test_epoch
            )
            do_calc_and_log_stats = do_calc_and_log_stats and is_main_process

            if dl.sampler is not None and hasattr(dl.sampler, "set_epoch"):
                dl.sampler.set_epoch(epoch)
            num_batches = len(dl)
            progress = tqdm(total=num_batches, desc=key, leave=False)
            set_network_mode(network_ddp, key)

            statistics[key] = init_statistics(
                use_occupancy_loss_with_input=args.use_occupancy_loss_with_input,
            )

            # init statistics for elevated areas
            num_elevated_samples = 0
            num_known_elevated_samples = 0
            statistics_elevated[key] = init_statistics_elevated()
            statistics_elevated_known[key] = init_statistics_elevated_known()

            num_samples = 0

            vis_freq_batch_per_epoch = args.vis_freq_batch_per_epoch
            if vis_freq_batch_per_epoch > num_batches:
                if is_main_process:
                    print(
                        f"{vis_freq_batch_per_epoch=} is greater than {num_batches=}. Clipping it to 2"
                    )
                vis_freq_batch_per_epoch = 2

            for batch_idx, new_data_dict in enumerate(dl):

                assert (
                    len(new_data_dict["measured"]) == mini_batch_size
                ), f"Network works with fixed batch size: {mini_batch_size=}"

                new_data_dict = transfer_batch_to_device(new_data_dict, device)
                for k, v in new_data_dict.items():
                    if isinstance(v, list) and all(
                        isinstance(elem, torch.Tensor) for elem in v
                    ):
                        new_data_dict[k] = torch.stack(v)
                for t, data_dict in enumerate([new_data_dict]):
                    target = data_dict["gt"].clone()
                    measured = data_dict["measured"].clone()

                    if args.model_name == "pflow":
                        forward_kwargs = {
                            "optimizer": optimizer,
                            "step": epoch * len(dl) + batch_idx,
                            "writer": writer,
                        }
                    else:
                        forward_kwargs = {}
                    point_net_forward_res = point_net_forward(
                        args, network, key, data_dict, **forward_kwargs
                    )
                    losses = point_net_forward_res["losses"]

                    if key == "train" and args.model_name != "pflow":

                        if args.model_name == "seedformer" and args.use_seed_loss:
                            seed_loss = get_seed_discrepancy_loss(
                                measured,
                                seed_xyz=point_net_forward_res["out"]["seed_xyz"],
                                seed_feat=point_net_forward_res["out"]["seed_feat"],
                                labels=(
                                    None
                                    if args.use_seed_loss_unsup
                                    else data_dict["labels"]
                                ),
                                unsup_seed_subsample_ratio=args.unsup_seed_subsample_ratio,
                                similar_dist_thresh=args.similar_dist_thresh,
                                margin=args.triplet_loss_margin,
                                use_full_triplet_loss=args.sl_use_full_triplet_loss,
                                do_flatten_batch_dim=args.sl_do_flatten_batch_dim,
                            )
                            losses["seed_loss"] = (
                                args.seed_loss_coef * seed_loss["loss"]
                            )
                            losses["loss"] += seed_loss["loss"]

                        optimizer.zero_grad()
                        losses["loss"].backward()
                        max_norm = 1.0
                        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm)
                        optimizer.step()

                    pcds_pred = point_net_forward_res["preds"]

                    if vis_freq_batch_per_epoch:
                        vis_batch_idxs = np.linspace(
                            0, num_batches - 1, vis_freq_batch_per_epoch, dtype=int
                        )
                    else:
                        vis_batch_idxs = [0]
                    do_vis_this_batch = (
                        batch_idx in vis_batch_idxs
                        and ((epoch + 1) % args.vis_freq_epoch == 0 or is_last_epoch)
                        and vis_enabled
                        and is_main_process
                    )
                    if args.model_name == "pflow" and (
                        do_vis_this_batch or do_calc_and_log_stats
                    ):
                        pcds_pred = network.reconstruct(measured)

                    if (
                        args.model_name == "pflow"
                        and args.pflow_use_gen
                        and (do_vis_this_batch or do_calc_and_log_stats)
                    ):
                        _, pcds_gen = network.sample(
                            measured.shape[0], measured.shape[1], gpu=rank
                        )
                        from pflow.metrics.evaluation_metrics import (
                            jsd_between_point_cloud_sets,
                        )

                        jsd = jsd_between_point_cloud_sets(
                            pcds_gen, target, resolution=32
                        )
                        vis_gen_based(
                            visualizer=visualizer,
                            measured=pcds_gen,
                            log_img_name=f"{key}/gen_batch_{batch_idx}_step_{t}",
                            step=epoch * len(dl) + batch_idx,
                        )
                        losses["jsd"] = jsd

                    update_statistics_with_losses(
                        statistics[key],
                        losses,
                    )

                    if args.use_pc_norm:
                        data_dict["mean"] = data_dict["mean"].unsqueeze(1)
                        data_dict["max_dist"] = (
                            data_dict["max_dist"].unsqueeze(1).unsqueeze(1)
                        )
                    if args.use_pc_norm:
                        pcds_pred_denorm = denormalize_pcl(
                            pcds_pred, data_dict["mean"], data_dict["max_dist"]
                        )
                        target_denorm = denormalize_pcl(
                            target, data_dict["mean"], data_dict["max_dist"]
                        )
                    else:
                        pcds_pred_denorm = pcds_pred
                        target_denorm = target
                    if do_vis_this_batch:
                        vis_point_based(
                            visualizer,
                            data_dict,
                            measured,
                            pcds_pred_denorm,
                            target_denorm,
                            log_img_name=f"{key}/batch_{batch_idx}_step_{t}",
                            step=epoch * len(dl) + batch_idx,
                            do_vis_entire_batch=args.do_vis_entire_batch,
                            use_pc_norm=args.use_pc_norm,
                        )

                    if not do_calc_and_log_stats:
                        continue

                    occupancy = (
                        convert_pts_to_vg(
                            pcds_pred_denorm,
                            map_dim,
                            map_resolution,
                            use_xy_centroids=False,
                        )[..., 0]
                        >= 0
                    )

                    target_vg = convert_pts_to_vg(
                        target_denorm,
                        map_dim,
                        map_resolution,
                        use_xy_centroids=False,
                    )

                    stats = compute_occupancy_metrics(
                        occupancy_estimate=occupancy,
                        target=target_vg,
                    )
                    for stat_key, elem in stats.items():
                        if isinstance(elem, torch.Tensor):
                            elem = elem.detach()
                        statistics[key][stat_key].append(elem)

                    do_calc_dist_metrics = (
                        epoch >= start_calc_metrics_from_n_epoch
                    ) and args.do_calc_dist_metrics

                    if not do_calc_dist_metrics:
                        continue

                    # Metrics

                    pts_gts = target_denorm
                    pts_preds = pcds_pred_denorm

                    for i in range(len(pts_gts)):
                        if len(pts_preds[i]) == 0:
                            print("Empty pts_pred. Cannot compute dist metrics")
                            do_calc_dist_metrics = False
                            break
                        if len(pts_gts[i]) == 0:
                            print("Empty pts_gt. Cannot compute dist metrics")
                            do_calc_dist_metrics = False
                            break

                    if not do_calc_dist_metrics:
                        continue

                    area_maps = data_dict.get("area_map")
                    elevation_info, dist_metrics = (
                        compute_dist_metrics_and_elevated_idxs(
                            args=args,
                            pts_gts=pts_gts,
                            pts_preds=pts_preds,
                            pose=data_dict.get("pose"),
                            area_maps=area_maps,
                        )
                    )

                    update_statistics(
                        statistics[key],
                        dist_metrics["maes"],
                        dist_metrics["maes_height"],
                        dist_metrics["mae_height_granular"],
                        dist_metrics["chamfer_dists"],
                    )

                progress.update(1)
                num_samples += mini_batch_size * args.sequence_length

            if do_calc_and_log_stats:
                # calc share of elevated samples out of all samples seen in the epoch
                statistics_elevated[key]["share_elevated_samples"].append(
                    num_elevated_samples / num_samples if num_samples > 0 else 0
                )
                statistics_elevated_known[key]["share_elevated_samples"].append(
                    num_known_elevated_samples / num_samples if num_samples > 0 else 0
                )

        epoch_duration = calc_epoch_duration(start_time)
        epoch_durations.append(epoch_duration)
        print_time_stats(epoch_durations, epoch_duration)

        scheduler.step()

        if is_main_process:
            poke_visualizer(exp, visualizer)

        if is_main_process and (not args.do_overfit or args.do_save_model):
            if (epoch + 1) % args.save_every_n_epochs == 0 or is_last_epoch:
                save_path_model = log_artifacts(exp, save_artifacts, log_dir, epoch)
                print(f"Saved model to {save_path_model}")

        if args.use_ddp:
            dist.barrier()

        if not do_calc_and_log_stats:
            continue

        for stage, stats in statistics.items():
            for metric_name, all_values_in_epoch in stats.items():
                metric_avg = get_mean_of_arr(all_values_in_epoch)
                if torch.isnan(metric_avg):
                    if metric_name not in [
                        "loss",
                        "loss_centroid",
                        "loss_occupancy",
                        "mae",
                        "chamfer_dist",
                    ]:
                        metric_avg = 0.0
                history[stage][metric_name].append(metric_avg)

        if args.use_early_stopping and args.use_val_ds:
            if args.model_name == "pflow":
                es_kwargs = dict(loss=history["val"]["recon"][-1])
            else:
                es_kwargs = dict(loss=history["val"]["loss"][-1])
            early_stop(**es_kwargs)
            if early_stop.do_stop:
                print(f"Early stopping on epoch {epoch}")
                if is_main_process and (not args.do_overfit or args.do_save_model):
                    save_path_model = log_artifacts(exp, save_artifacts, log_dir, epoch)
                    print(f"Saved model to {save_path_model}")
                break
        # logging and printing epoch statistics
        width = 80
        pad = 28
        str = f" \033[1m Learning epoch {epoch}/{args.num_epochs} \033[0m "
        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'LR:':>{pad}} {scheduler.get_last_lr()[0]:.5f}\n"""
        )
        print_main_log_msg(
            log_statistics_and_assemble_log_msg(
                writer, statistics, epoch, pad, heading=log_string
            )
        )

        for key in statistics_elevated.keys():
            found_elev_areas = (
                sum(statistics_elevated[key]["share_elevated_samples"]) > 0
            )
            if found_elev_areas:
                print_main_log_msg(
                    log_statistics_and_assemble_log_msg(
                        writer,
                        statistics_elevated,
                        epoch=epoch,
                        pad=pad,
                        heading="ELEVATED AREAS\n",
                        stat_key_prefix="elevated",
                    )
                )

            found_known_elev_areas = (
                sum(statistics_elevated_known[key]["share_elevated_samples"]) > 0
            )
            if found_known_elev_areas:
                print_main_log_msg(
                    log_statistics_and_assemble_log_msg(
                        writer,
                        statistics_elevated_known,
                        epoch=epoch,
                        pad=pad,
                        heading="KNOWN ELEVATED AREAS\n",
                        stat_key_prefix="elevated_known",
                    )
                )

        writer.add_scalar("Optimizer/learning_rate", scheduler.get_last_lr()[0], epoch)

    if args.use_ddp:
        destroy_process_group()

    if not external_tools and is_main_process:
        exp.end()
    return history


if __name__ == "__main__":
    args = parse_trep_args()

    if args.do_optimize:
        from terrain_representation.hyperparam_tuning import run_study

        optimal_params = run_study(
            args, run_pipe=functools.partial(main, rank=0, world_size=1)
        )
        update_args(args, optimal_params)
        args.exp_tags += ["best_trial"]
        print("Running pipe with optimal params: ", optimal_params)

    if args.do_benchmark:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        profile_func(
            functools.partial(main, rank=0, world_size=1),
            f"{root_dir}/assets/profiling/train_point_based_{now}.prof",
            args=args,
        )
    else:
        if args.use_ddp:
            world_size = torch.cuda.device_count()
            args.master_port = np.random.randint(20000, 30000)
            mp.spawn(
                main,
                args=(
                    world_size,
                    args,
                ),
                nprocs=world_size,
            )
        else:
            main(0, 1, args)
