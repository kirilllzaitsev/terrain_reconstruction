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
from terrain_representation.storage.dataset_torch import custom_collate_fn
from terrain_representation.utils import parse_trep_args
from terrain_representation.utils.comet_utils import root_dir
from terrain_representation.utils.data_utils import get_dataloaders
from terrain_representation.utils.loss_utils import (
    compute_occupancy_metrics_sparse,
    loss_func_sparse,
)
from terrain_representation.utils.pipeline_utils import (
    calc_epoch_duration,
    compute_dist_metrics_and_elevated_idxs,
    init_statistics,
    init_statistics_elevated,
    init_statistics_elevated_known,
    is_dist_loss_used,
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
from terrain_representation.utils.pointcloud_utils import sparse_tensor_to_point_cloud
from terrain_representation.utils.profiling_utils import profile_func
from terrain_representation.utils.utils import (
    align_current_data_dict_to_prev_sparse,
    transfer_batch_to_device,
    update_args,
)
from torch.distributed import destroy_process_group
from tqdm.auto import tqdm


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
    assert (
        args.model_name == "unet"
    ), "Only 'unet' model is supported for voxel-based pipe"

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

    pos_weight = torch.tensor([args.pos_weight], device=device)
    threshold = args.occupancy_threshold
    centroid_weight = args.centroid_weight * (32 / args.map_dimension) ** 3
    writer.add_scalar("Optimizer/pos_weight", pos_weight.item(), 0)
    writer.add_scalar("Optimizer/threshold", threshold, 0)
    writer.add_scalar("Optimizer/centroid", centroid_weight, 0)

    dataloaders = get_dataloaders(
        args, datasets, mini_batch_size, collate_fn=custom_collate_fn
    )

    for epoch in tqdm(range(args.num_epochs), desc="Epochs", total=args.num_epochs):
        start_time = datetime.now()
        is_last_epoch = epoch == args.num_epochs - 1
        for key, dl in dataloaders.items():

            do_calc_and_log_stats = (
                epoch + 1
            ) % args.log_every_n_epochs == 0 or is_last_epoch
            do_calc_and_log_stats = do_calc_and_log_stats and is_main_process

            if "test" in key and (epoch + 1) % args.test_freq_epoch != 0:
                continue

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
                network.reset()

                assert (
                    len(new_data_dict["measured"]) == mini_batch_size
                ), f"Network works with fixed batch size: {mini_batch_size=}"

                new_data_dict = transfer_batch_to_device(new_data_dict, device)
                data_dicts = align_current_data_dict_to_prev_sparse(new_data_dict)
                for t, data_dict in enumerate(data_dicts):

                    measured = data_dict["measured"]

                    if key == "train":
                        context = torch.enable_grad()
                    else:
                        context = torch.no_grad()

                    with context:
                        net_outputs = network(measured, data_dict["gt"])

                    use_dist_loss = is_dist_loss_used(args, epoch)

                    common_loss_kwargs = dict(
                        pos_weight=pos_weight,
                        centroid_weight=centroid_weight,
                        use_mae_loss=args.use_mae_loss and use_dist_loss,
                        use_chamfer_loss=args.use_chamfer_loss and use_dist_loss,
                        added_noise=data_dict["added_noise"],
                    )
                    mesh = data_dict.get("mesh")

                    gt = data_dict["gt"]
                    layer_outputs = net_outputs["layer_outputs"]
                    layer_targets = net_outputs["layer_targets"]
                    losses = loss_func_sparse(
                        data={
                            "layer_outputs": layer_outputs,
                            "layer_targets": layer_targets,
                        },
                        batch_size=mini_batch_size,
                        map_resolution=map_resolution,
                        target_pts=mesh,
                        **common_loss_kwargs,
                    )

                    if key == "train":
                        optimizer.zero_grad()
                        losses["loss"].backward()
                        # max_norm = 1.0
                        # torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm)
                        optimizer.step()

                    update_statistics_with_losses(
                        statistics[key],
                        losses,
                    )

                    if vis_freq_batch_per_epoch:
                        vis_batch_idxs = np.linspace(
                            1, num_batches, vis_freq_batch_per_epoch, dtype=int
                        )
                    else:
                        vis_batch_idxs = [1]

                    if (
                        (batch_idx + 1) in vis_batch_idxs
                        and ((epoch + 1) % args.vis_freq_epoch == 0 or is_last_epoch)
                        and vis_enabled
                        and is_main_process
                    ):

                        if args.do_vis_entire_batch:
                            take_idxs = range(mini_batch_size)
                        else:
                            take_idxs = [0]
                        for take_idx in take_idxs:
                            vis_data_dict = {
                                "measurement": measured[take_idx],
                                "input": sparse_tensor_to_point_cloud(
                                    network.input,
                                    take_idx,
                                    map_resolution=map_resolution,
                                ),
                                "output": sparse_tensor_to_point_cloud(
                                    net_outputs["output"],
                                    take_idx,
                                    map_resolution=map_resolution,
                                ),
                                "gt": gt[take_idx],
                                "log_img_name": f"{key}/batch_{batch_idx}_step_{t}",
                                "step": epoch * len(dl) + batch_idx,
                            }
                            if data_dict.get("mesh") is not None:
                                vis_data_dict["mesh"] = data_dict["mesh"][take_idx]

                            visualizer.display_point_clouds(vis_data_dict)

                    if not do_calc_and_log_stats:
                        continue

                    stats = compute_occupancy_metrics_sparse(
                        net_outputs["output"],
                        net_outputs["target_sparse"],
                        net_outputs["kernel_map_out"],
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

                    pts_gts = mesh
                    pts_preds = [
                        sparse_tensor_to_point_cloud(
                            net_outputs["output"],
                            batch_idx=i,
                            map_resolution=map_resolution,
                        )
                        for i in range(mini_batch_size)
                    ]

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

                    elevation_info, dist_metrics = (
                        compute_dist_metrics_and_elevated_idxs(
                            args=args,
                            pts_gts=pts_gts,
                            pts_preds=pts_preds,
                            pose=data_dict["pose"],
                            area_maps=None,
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
                metric_avg = torch.mean(torch.tensor(list(all_values_in_epoch)))
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
            early_stop(metric=history["val"]["f1"][-1])
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

        for k in statistics_elevated.keys():
            found_elev_areas = sum(statistics_elevated[k]["share_elevated_samples"]) > 0
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
                sum(statistics_elevated_known[k]["share_elevated_samples"]) > 0
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

    if is_main_process:
        poke_visualizer(exp, visualizer)

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
            f"{root_dir}/assets/profiling/train_dense_v2_{now}.prof",
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
            # world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
            # rank = int(os.environ.get("SLURM_PROCID", 0))
            # args.master_port = os.environ.get(
            #     "MASTER_PORT", np.random.randint(20000, 30000)
            # )
            # main(rank, world_size, args)
        else:
            main(0, 1, args)
