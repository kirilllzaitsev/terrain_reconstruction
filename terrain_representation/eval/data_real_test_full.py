import argparse
import copy
import datetime as dt
import json
import os
import shutil
import time
import traceback
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from comet_ml import API
from terrain_representation.eval.eval_utils import (
    ablation_exp_names,
    ar_exp_names,
    benchmark_exp_names,
    compute_dist_metrics,
    create_eval_exp,
    ds_alias_to_name,
    ds_name_to_full_name,
    exp_name_to_meaning,
    fix_args,
    grid_resolution_exp_names,
    req_stats_keys,
    seed_loss_exp_names,
    sparse_exp_names,
    vis_sparse_voxel_based,
    vis_voxel_based,
)
from terrain_representation.storage.dataset_adapters import (
    PointCloudDatasetAdapterPointAttn,
)
from terrain_representation.storage.dataset_torch import custom_collate_fn
from terrain_representation.train_dense_v2 import poke_visualizer
from terrain_representation.train_point_based import point_net_forward, vis_point_based
from terrain_representation.utils.comet_utils import load_artifacts_from_comet_v2
from terrain_representation.utils.loss_utils import (
    compute_occupancy_metrics,
    compute_occupancy_metrics_sparse,
    loss_func_dense,
    loss_func_sparse,
)
from terrain_representation.utils.pipeline_utils import (
    convert_gt_and_pred_voxel_grids_to_pts,
    create_datasets,
    create_visualizer,
    get_model,
)
from terrain_representation.utils.pointcloud_utils import (
    denormalize_pcl,
    point_clouds_to_sparse_tensor,
    sparse_tensor_to_point_cloud,
)
from terrain_representation.utils.utils import (
    align_current_data_dict_to_prev_dense,
    align_current_data_dict_to_prev_sparse,
    load_from_checkpoint,
    set_seed,
    transfer_batch_to_device,
)
from terrain_representation.utils.voxel_grid_utils import (
    convert_batch_of_pts_to_vg,
    convert_pts_to_vg,
    convert_vg_to_pcl,
    merge_vgs_in_common_frame,
)
from tqdm import tqdm

SAVE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "results",
        dt.datetime.now().strftime("%Y_%m_%d_%H_%M"),
    )
)


def copy_df(df):
    import pyperclip

    pyperclip.copy(df.to_markdown())


def format_exp_stats(stats):
    stats = copy.deepcopy(stats)
    for k, v in stats.items():
        if isinstance(v, torch.Tensor):
            stats[k] = v.item()
        else:
            stats[k] = torch.mean(torch.tensor(v)).item()
    return stats


def main(cli_args):

    set_seed(100)

    comet_api = API(api_key=os.environ["COMET_API_KEY"])

    if cli_args.exp_names_pack == "ablation":
        exp_names = ablation_exp_names
    elif cli_args.exp_names_pack == "benchmark":
        exp_names = benchmark_exp_names
    elif cli_args.exp_names_pack == "ar":
        exp_names = ar_exp_names
    elif cli_args.exp_names_pack == "res":
        exp_names = grid_resolution_exp_names
    elif cli_args.exp_names_pack == "sparse":
        exp_names = sparse_exp_names
    elif cli_args.exp_names_pack == "seed_loss":
        exp_names = seed_loss_exp_names
    elif cli_args.exp_names_pack == "all":
        exp_names = ablation_exp_names + benchmark_exp_names + ar_exp_names

    if cli_args.exp_names:
        exp_names = cli_args.exp_names

    artifact_dir = cli_args.artifact_dir
    device = "cuda"
    use_dist_loss = True

    exp_stats = defaultdict(dict)
    ds_cache = {
        "points": defaultdict(dict),
        "voxels": defaultdict(dict),
    }

    if cli_args.do_vis:
        vis_args = argparse.Namespace(
            **{
                "plots_interactive": False,
                "plots_disabled": True,
                "plots_zoom": 1.0,
                "use_cylindrical_coords": False,
                "use_ddp": False,
            }
        )
        vis_args.use_async_vis = cli_args.use_async_vis
        visualizer = create_visualizer(vis_args)
        visualizer.use_pyplot = False
        visualizer.use_pyplot = True
    else:
        visualizer = None

    exp = create_eval_exp(exp_disabled=cli_args.exp_disabled)
    metadata = defaultdict(dict)
    failed_exp_names = {}
    samplers = {}

    exp_stats_path = f"{Path(SAVE_DIR).parent}/exp_stats.json"
    try:
        existing_exp_stats = json.load(open(exp_stats_path))
    except Exception as e:
        print(f"Error loading {exp_stats_path=}: {e}")
        existing_exp_stats = {}

    for exp_name in tqdm(exp_names, desc="Experiments"):

        if cli_args.use_exisiting_exp_stats:
            if exp_name in existing_exp_stats:
                existing_exp_stats_exp = existing_exp_stats.get(exp_name)
                if existing_exp_stats_exp is not None:
                    for ds_name, ds_stats in existing_exp_stats_exp[
                        "exp_stats"
                    ].items():
                        exp_stats[ds_name][exp_name] = ds_stats
                    metadata["avg_forward_path_times"][exp_name] = existing_exp_stats[
                        exp_name
                    ]["metadata"]["avg_forward_path_time"]
                    metadata["weights_size"][exp_name] = existing_exp_stats[exp_name][
                        "metadata"
                    ]["weights_size"]
                    continue

        save_dir = f"{artifact_dir}/{exp_name}/preds"
        if cli_args.force_artifacts_download or cli_args.rm_prev_preds:
            shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir, exist_ok=True)

        ckpt_path = f"{artifact_dir}/{exp_name}/model.pt"
        save_path_args = f"{artifact_dir}/{exp_name}/train_args.yaml"

        if cli_args.force_artifacts_download:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

        try:
            artifacts = load_artifacts_from_comet_v2(
                exp_name, ckpt_path, save_path_args, comet_api
            )
        except Exception as e:
            traceback.print_exc()
            print(f"Error loading {exp_name=}")
            failed_exp_names[exp_name] = traceback.format_exc()
            continue

        args = artifacts["args"]
        args = fix_args(args, cli_args, construction_site_path=f"{ntr_folder}/construction_site")

        map_dim = torch.tensor(
            [args.map_dimension, args.map_dimension, args.map_dimension]
        )
        if args.map_dimension == 64:
            mini_batch_size = 8
        elif args.map_dimension == 96:
            mini_batch_size = 2
        elif args.map_dimension == 128:
            mini_batch_size = 1
        map_resolution = torch.tensor(
            [args.map_resolution_xy, args.map_resolution_xy, args.map_resolution_z],
            device=device,
        )

        network = get_model(
            args,
            device,
            map_dim,
            map_resolution,
            mini_batch_size,
            use_large_model=(
                getattr(args, "use_large_model", False)
                or (
                    True
                    if exp_name in ["grand_column_8877", "renewed_chickadee_7053"]
                    else False
                )
            ),
        )
        network.eval()
        for p in network.parameters():
            p.requires_grad = False
        load_from_checkpoint(ckpt_path, network, device)

        use_points_input = not (
            getattr(args, "model_name", "unet")
            in [
                "unet",
                "sparse_unet",
            ]
        )
        real_ds_cache = ds_cache["points"] if use_points_input else ds_cache["voxels"]
        if map_resolution not in real_ds_cache:
            datasets = create_datasets(
                args=args,
                device=device,
                map_dim=map_dim,
                map_resolution=map_resolution,
                noisifier=None,
                use_real_ds_in_train=False,
                add_test_ds=cli_args.ds_names != ["train"],
            )
            if use_points_input:
                for k, ds in datasets.items():
                    datasets[k] = PointCloudDatasetAdapterPointAttn(
                        ds, gt_pts_scaler=1, use_pc_norm=use_points_input
                    )

            for ds_alias, ds_name in ds_alias_to_name.items():
                if ds_name in datasets:
                    real_ds_cache[map_resolution][ds_alias] = datasets[ds_name]

        # get sampler that picks random idxs from dataset. do it only once
        if not samplers:
            # all datasets have the same length
            for ds_name, ds in real_ds_cache[map_resolution].items():
                # ds_name = ds_alias_to_name[ds_alias]
                samplers[ds_name] = torch.utils.data.RandomSampler(
                    ds, replacement=True, num_samples=len(ds)
                )

        try:
            res = run_exp_on_datasets(
                device,
                mini_batch_size,
                use_dist_loss,
                exp_stats,
                exp_name,
                args,
                map_dim,
                map_resolution,
                network,
                real_ds_cache[map_resolution],
                cli_args=cli_args,
                visualizer=visualizer,
                use_points=use_points_input,
                use_sparse=args.use_sparse,
                dl_slice_len=cli_args.dl_slice_len,
                use_extended_input=cli_args.use_extended_input
                or exp_name in ar_exp_names,
                use_prev_pred_as_input=cli_args.use_prev_pred_as_input
                or args.use_prev_pred_as_input,
                exp_dir=f"{artifact_dir}/{exp_name}",
                samplers=samplers,
            )
            metadata["avg_forward_path_times"][exp_name] = res["avg_forward_path_time"]
            metadata["avg_inference_times"][exp_name] = res["avg_inference_time"]
            metadata["weights_size"][exp_name] = get_size(ckpt_path)
        except Exception as e:
            traceback.print_exc()
            exc_msg = traceback.format_exc()
            print(f"Error running {exp_name=}")
            failed_exp_names[exp_name] = exc_msg
            raise

        # break

    if visualizer and not cli_args.dry_run:
        if hasattr(visualizer, "process"):
            visualizer.stop()
        poke_visualizer(exp, visualizer, save_dir=save_dir)

    save_dir_csv = SAVE_DIR
    if not cli_args.exp_names:
        save_dir_csv = save_dir_csv.replace(
            "results/", f"results/{cli_args.exp_names_pack}_"
        )
    os.makedirs(save_dir_csv, exist_ok=True)

    # METADATA
    metadata_df = pd.DataFrame(metadata)
    # fill in NA with -1
    metadata_df = metadata_df.fillna(-1)
    metadata_df["avg_forward_path_times"] = (
        metadata_df["avg_forward_path_times"] * 1000
    ).astype(int)
    metadata_df["avg_inference_times"] = (
        metadata_df["avg_inference_times"] * 1000
    ).astype(int)
    metadata_df = metadata_df.round(3).rename(
        columns={
            "avg_forward_path_times": "Forward Path Time, ms",
            "weights_size": "Weights Size, MB",
            "avg_inference_times": "Inference Time, ms",
        }
    )
    metadata_df = metadata_df.rename_axis("Experiment")
    # drop Forward Path Time, ms
    metadata_df = metadata_df.drop(columns=["Forward Path Time, ms"])
    print("\n### Metadata ###")
    print(metadata_df.to_markdown())
    metadata_df.to_csv(f"{save_dir_csv}/metadata.csv", index=True)

    print("\n### Metadata per ds ###")
    for ds_name, ds_exp_stats in exp_stats.items():
        df = get_df_from_ds_exp_stats(
            ["avg_nan_pred", "avg_num_pred", "avg_num_gt"], ds_exp_stats
        )
        df = df.round(2)
        print(f"\n### {ds_name_to_full_name[ds_name]} ###")
        print(df.to_markdown())
        df.to_csv(f"{save_dir_csv}/{ds_name}_metadata.csv", index=True)

    # MAIN STATS

    for ds_name, ds_exp_stats in exp_stats.items():
        df = get_df_from_ds_exp_stats(req_stats_keys, ds_exp_stats)
        df = df.rename(
            columns={
                "maes": "MAE",
                "maes_height": "MAE (height)",
                "chamfer_dists": "Chamfer Dist",
                "f1": "F1",
                "recall": "Recall",
                "precision": "Precision",
                "exp_name": "CometML Name",
            }
        )

        df = df.round(4)
        print(f"\n### {ds_name_to_full_name[ds_name]} ###")
        print(df.to_markdown())
        df.to_csv(f"{save_dir_csv}/{ds_name}.csv", index=True)

    if cli_args.do_write_exp_stats:
        exp_stats_reformatted = defaultdict(defaultdict)
        exp_stats_reformatted[exp_name]["exp_stats"] = {}
        for ds_name, ds_exp_stats in exp_stats.items():
            for exp_name, stats in ds_exp_stats.items():
                exp_stats_reformatted[exp_name]["exp_stats"][ds_name] = (
                    format_exp_stats(stats)
                )
        for exp_name in exp_names:
            exp_stats_reformatted[exp_name]["metadata"] = {}
            exp_stats_reformatted[exp_name]["metadata"]["avg_forward_path_time"] = (
                metadata["avg_forward_path_times"][exp_name]
            )
            exp_stats_reformatted[exp_name]["metadata"]["avg_inference_time"] = (
                metadata["avg_inference_times"][exp_name]
            )
            exp_stats_reformatted[exp_name]["metadata"]["weights_size"] = metadata[
                "weights_size"
            ][exp_name]

        with open(exp_stats_path, "w") as f:
            json.dump(exp_stats_reformatted, f, indent=4)

        print(f"Saved stats to {exp_stats_path=}")

    print(f"{save_dir_csv=} {os.path.basename(save_dir_csv)}")

    if cli_args.dry_run:
        shutil.rmtree(save_dir_csv, ignore_errors=True)

    exp.end()

    if failed_exp_names:
        print("\n### Failed experiments ###")
        print(failed_exp_names)
        raise Exception("Some experiments failed")


def get_df_from_ds_exp_stats(cols, ds_exp_stats):
    df = pd.DataFrame(
        {
            k: format_exp_stats({k: v for k, v in es.items() if k in cols})
            for k, es in ds_exp_stats.items()
        }
    ).T
    df["comment"] = (
        df.reset_index()["index"].apply(lambda x: exp_name_to_meaning.get(x, "")).values
    )

    # put comment as index, and index as exp_name
    df = df.rename_axis("exp_name").reset_index(drop=False)
    df = df.set_index("comment")
    df = df.rename_axis("Experiment")

    # sort df cols to match req_stats_keys order
    df = df[cols + ["exp_name"]]
    return df


def run_exp_on_datasets(
    device,
    mini_batch_size,
    use_dist_loss,
    exp_stats,
    exp_name,
    args,
    map_dim,
    map_resolution,
    network,
    datasets,
    visualizer,
    cli_args,
    use_points=False,
    use_sparse=False,
    dl_slice_len=None,
    use_extended_input=False,
    use_prev_pred_as_input=False,
    exp_dir=None,
    samplers=None,
):
    forward_path_times = []  # a single forward path of the net
    inference_times = []  # forward path + all necessary preprocessing
    nan_pred = defaultdict(int)
    num_pred = defaultdict(int)
    num_gt = defaultdict(int)
    use_pc_norm = getattr(args, "use_pc_norm", False)

    for ds_name, ds in datasets.items():
        ds_preds = defaultdict(list)

        sampler = samplers.get(ds_name) if samplers else None
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=mini_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
            drop_last=True,
            sampler=sampler,
        )
        if dl_slice_len is not None:
            dl = islice(dl, dl_slice_len)
            dl_len = dl_slice_len
        else:
            dl_len = len(dl)

        vis_batch_idxs = torch.linspace(0, dl_len - 1, 7).int().tolist()
        avg_stats = defaultdict(list)
        threshold = args.occupancy_threshold

        for batch_idx, new_data_dict in tqdm(enumerate(dl), total=dl_len):

            assert (
                len(new_data_dict["measured"]) == mini_batch_size
            ), f"Network works with fixed batch size: {mini_batch_size=}"

            new_data_dict = transfer_batch_to_device(new_data_dict, device)
            time_prep = 0
            if use_points:
                for k, v in new_data_dict.items():
                    if isinstance(v, list) and all(
                        isinstance(elem, torch.Tensor) for elem in v
                    ):
                        new_data_dict[k] = torch.stack(v)
                data_dicts = [new_data_dict]
            elif use_sparse:
                data_dicts = align_current_data_dict_to_prev_sparse(new_data_dict)
            else:
                start = time.time()
                new_data_dict = convert_batch_of_pts_to_vg(
                    new_data_dict,
                    map_dim,
                    map_resolution,
                    use_xy_centroids=args.use_centroids_for_dist_loss,
                )
                time_prep = time.time() - start
                data_dicts = align_current_data_dict_to_prev_dense(new_data_dict)
            if not use_points:
                network.reset()
            centroids = None
            occupancy = None
            for t, data_dict in enumerate(data_dicts):

                if use_extended_input and not (use_points or use_sparse) and t > 0:
                    if use_prev_pred_as_input:
                        extension_vg = centroids.detach().clone()
                        extension_vg[~occupancy] = -1.0
                    else:
                        extension_vg = data_dicts[t - 1]["measured"]
                    merged_vg = merge_vgs_in_common_frame(
                        data_dict["measured"], extension_vg
                    )
                    measured = merged_vg
                else:
                    measured = data_dict["measured"]
                target = data_dict["gt"]
                target_mesh = data_dict["mesh"]

                if use_points:
                    if args.model_name == "pflow":
                        forward_kwargs = {
                            "optimizer": None,
                            "step": batch_idx,
                            "writer": None,
                        }
                    else:
                        forward_kwargs = {}

                    start = time.time()
                    point_net_forward_res = point_net_forward(
                        args, network, "test", data_dict, do_compute_loss=False, **forward_kwargs
                    )
                    forward_path_time = time.time() - start
                    forward_path_times.append(forward_path_time)
                    inference_times.append(forward_path_time)

                    losses = point_net_forward_res["losses"]
                    pcds_pred = point_net_forward_res["preds"]
                    if args.model_name == "pflow" and (True):
                        pcds_pred = network.reconstruct(measured)
                    if use_pc_norm:
                        data_dict["mean"] = data_dict["mean"].unsqueeze(1)
                        data_dict["max_dist"] = (
                            data_dict["max_dist"].unsqueeze(1).unsqueeze(1)
                        )
                    if use_pc_norm:
                        pcds_pred_denorm = denormalize_pcl(
                            pcds_pred, data_dict["mean"], data_dict["max_dist"]
                        )
                        target_denorm = denormalize_pcl(
                            target_mesh, data_dict["mean"], data_dict["max_dist"]
                        )
                    else:
                        pcds_pred_denorm = pcds_pred
                        target_denorm = target_mesh

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
                        occupancy,
                        target_vg,
                    )
                else:
                    if use_sparse:
                        start = time.time()
                        measured_sparse = point_clouds_to_sparse_tensor(
                            map_resolution, measured, 0
                        )
                        time_prep = time.time() - start
                    start = time.time()
                    with torch.no_grad():
                        if use_sparse:
                            net_outputs = network(
                                measured_sparse,
                                target,
                            )
                        else:
                            net_outputs = network(
                                measured,
                                threshold,
                            )

                    forward_path_time = time.time() - start
                    forward_path_times.append(forward_path_time)
                    inference_times.append(forward_path_time)

                    if use_sparse:
                        common_loss_kwargs = dict(
                            pos_weight=args.pos_weight,
                            centroid_weight=args.centroid_weight,
                            use_mae_loss=args.use_mae_loss and use_dist_loss,
                            use_chamfer_loss=args.use_chamfer_loss and use_dist_loss,
                            added_noise=data_dict["added_noise"],
                        )
                        mesh = data_dict.get("mesh")

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
                        stats = compute_occupancy_metrics_sparse(
                            net_outputs["output"],
                            net_outputs["target_sparse"],
                            net_outputs["kernel_map_out"],
                        )
                    else:
                        occupancy_logits = net_outputs["occupancy_logits"]
                        centroids = net_outputs["centroids"]
                        occupancy = net_outputs["occupancy"]

                        common_loss_kwargs = dict(
                            pos_weight=args.pos_weight,
                            centroid_weight=args.centroid_weight,
                            use_mae_loss=args.use_mae_loss and use_dist_loss,
                            use_chamfer_loss=args.use_chamfer_loss and use_dist_loss,
                        )

                        losses = loss_func_dense(
                            data={
                                "occupancy_logits": occupancy_logits,
                                "occupancy": occupancy,
                                "centroids": centroids,
                                "target": target,
                                "map_dim": map_dim,
                                "map_resolution": map_resolution,
                                "target_mesh": target_mesh,
                                "measured_vg": measured,
                            },
                            **common_loss_kwargs,
                        )

                        stats = compute_occupancy_metrics(
                            occupancy,
                            target if target_mesh is None else target_mesh,
                        )

                area_maps = data_dict.get("area_map")

                # Metrics

                if use_points:
                    pts_gts = target_denorm
                    pts_preds = pcds_pred_denorm
                else:
                    if use_sparse:
                        # note: in training, mesh is used instead of gt
                        pts_gts = target
                        pts_preds = [
                            sparse_tensor_to_point_cloud(
                                net_outputs["output"],
                                batch_idx=i,
                                map_resolution=map_resolution,
                            )
                            for i in range(mini_batch_size)
                        ]
                    else:
                        pts_gts, pts_preds = convert_gt_and_pred_voxel_grids_to_pts(
                            map_dim,
                            map_resolution,
                            net_outputs,
                            target=data_dict["gt"],
                            occupancy_threshold=threshold,
                            use_joint_mask=False,
                        )

                for i in range(len(pts_preds)):
                    if len(pts_preds[i]) == 0:
                        nan_pred[ds_name] += 1
                    num_pred[ds_name] += len(pts_preds[i])
                    num_gt[ds_name] += len(pts_gts[i])

                if use_points:
                    pred_vg = convert_pts_to_vg(
                        pts_preds,
                        map_dim,
                        map_resolution,
                        use_xy_centroids=False,
                    )
                elif use_sparse:
                    pred_vg = convert_pts_to_vg(
                        pts_preds,
                        map_dim,
                        map_resolution,
                        use_xy_centroids=False,
                    )
                    target_vg = convert_pts_to_vg(
                        pts_gts,
                        map_dim,
                        map_resolution,
                        use_xy_centroids=False,
                    )
                else:
                    pred_vg = net_outputs
                    target_vg = data_dict["gt"]

                pts_gts_dist, pts_preds_dist = convert_gt_and_pred_voxel_grids_to_pts(
                    map_dim,
                    map_resolution,
                    pred_vg,
                    target=target_vg,
                    occupancy_threshold=threshold,
                    use_joint_mask=True,
                )
                # dist metrics are computed wrt incomplete ground truth
                dist_metrics = compute_dist_metrics(
                    args, data_dict["pose"], area_maps, pts_gts_dist, pts_preds_dist
                )

                for k, v in stats.items():
                    avg_stats[k].append(v)
                for k, v in dist_metrics.items():
                    if "granular" in k:
                        continue
                    if isinstance(v, list):
                        filtered_v = []
                        for i, vv in enumerate(v):
                            if isinstance(vv, torch.Tensor):
                                if torch.isnan(vv).any():
                                    continue
                            else:
                                if np.isnan(vv).any():
                                    continue
                            filtered_v.append(vv)
                        v = torch.tensor(filtered_v).mean().item()
                    avg_stats[k].append(v)
                for k, v in losses.items():
                    avg_stats[k].append(v)

                if batch_idx not in vis_batch_idxs:
                    continue

                log_img_name = f"{exp_name}/{ds_name}/batch_{batch_idx}_step_{t}"
                ds_preds["pred"].extend([x.detach().cpu().numpy() for x in pts_preds])
                if use_sparse or use_points:
                    if use_pc_norm:
                        measured_pts = denormalize_pcl(
                            measured, data_dict["mean"], data_dict["max_dist"]
                        )
                        mesh_pts = denormalize_pcl(
                            target_mesh, data_dict["mean"], data_dict["max_dist"]
                        )
                    else:
                        measured_pts = measured
                        mesh_pts = target_mesh
                else:
                    measured_pts = [
                        x
                        for x in convert_vg_to_pcl(
                            measured, map_dim=map_dim, map_resolution=map_resolution
                        )
                    ]
                    mesh_pts = [
                        x
                        for x in convert_vg_to_pcl(
                            target_mesh,
                            map_dim=map_dim,
                            map_resolution=map_resolution,
                        )
                    ]

                ds_preds["measured"].extend(measured_pts)
                ds_preds["gt"].extend([x.detach().cpu().numpy() for x in pts_gts])
                ds_preds["mesh"].extend([x.detach().cpu().numpy() for x in mesh_pts])

                if visualizer:
                    for in_pcl, out_pcl in zip(measured_pts, pts_preds):
                        visualizer.display_input_output_pcls(
                            in_pcl, out_pcl, title=f"{log_img_name}_io", step=batch_idx
                        )
                    if use_points:
                        vis_point_based(
                            visualizer,
                            data_dict,
                            measured,
                            pcds_pred_denorm,
                            target_denorm,
                            log_img_name=log_img_name,
                            step=batch_idx,
                            do_vis_entire_batch=args.do_vis_entire_batch,
                            use_pc_norm=use_pc_norm,
                        )
                    else:
                        if use_sparse:
                            vis_sparse_voxel_based(
                                map_resolution=map_resolution,
                                visualizer=visualizer,
                                step=batch_idx,
                                sparse_input=network.input,
                                sparse_output=net_outputs["output"],
                                gt=target_mesh,
                                log_img_name=log_img_name,
                                mesh=net_outputs.get("mesh"),
                            )
                        else:
                            vis_voxel_based(
                                map_dim=map_dim,
                                map_resolution=map_resolution,
                                network=network,
                                visualizer=visualizer,
                                threshold=threshold,
                                step=batch_idx,
                                occupancy_logits=occupancy_logits,
                                data_dict=data_dict,
                                centroids=centroids,
                                log_img_name=log_img_name,
                            )

            torch.cuda.empty_cache()
            # break

        exp_stats[ds_name][exp_name] = avg_stats
        exp_stats[ds_name][exp_name].update(
            {
                "avg_nan_pred": (nan_pred[ds_name] / (dl_len * mini_batch_size)),
                "avg_num_pred": (num_pred[ds_name] / (dl_len * mini_batch_size)),
                "avg_num_gt": (num_gt[ds_name] / (dl_len * mini_batch_size)),
            }
        )

        if not cli_args.dry_run:
            torch.save(ds_preds, f"{exp_dir}/preds/{ds_name}_preds.pt")

    forward_path_times = forward_path_times[len(datasets):]
    inference_times = inference_times[len(datasets):]
    return {
        "avg_forward_path_time": np.mean(forward_path_times),
        "avg_inference_time": np.mean(inference_times),
    }


def get_size(path):
    size_mb = int(os.path.getsize(path) >> 20)
    return size_mb


if __name__ == "__main__":

    ntr_folder = Path(__file__).parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_names",
        nargs="+",
        choices=["hong", "bremg", "hong2", "train", "all"],
        default=["all"],
    )
    parser.add_argument(
        "--exp_names_pack",
        choices=["ablation", "benchmark", "ar", "res", "sparse", "seed_loss", "all"],
        default="ablation",
    )
    parser.add_argument(
        "--exp_names",
        nargs="*",
        default=[],
    )
    parser.add_argument("--copy_df", action="store_true")
    parser.add_argument("--do_vis", action="store_true")
    parser.add_argument("--use_async_vis", action="store_true")
    parser.add_argument("--use_extended_input", action="store_true")
    parser.add_argument("--use_prev_pred_as_input", action="store_true")
    parser.add_argument("--rm_prev_preds", action="store_true")
    parser.add_argument("--exp_enabled", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_exisiting_exp_stats", action="store_true")
    parser.add_argument("--do_write_exp_stats", action="store_true")
    parser.add_argument("--force_artifacts_download", action="store_true")
    parser.add_argument(
        "--dl_slice_len", type=int, help="Limit number of batches in dataloader"
    )
    parser.add_argument(
        "--artifact_dir",
        default=f"{ntr_folder}/artifacts",
    )
    parser.add_argument(
        "--hong_ds_dir",
        default=f"{ntr_folder}/construction_site/real",
    )
    parser.add_argument(
        "--bremg_ds_dir",
        default=f"{ntr_folder}/construction_site/real/armano_bremgarten/parsed/sunavigation",
    )
    parser.add_argument(
        "--hong2_ds_dir",
        default=f"{ntr_folder}/construction_site/real/hong_30_07_24/parsed/subset",
    )
    cli_args, _ = parser.parse_known_args()
    if cli_args.ds_names == ["all"]:
        cli_args.ds_names = ["hong", "bremg", "hong2", "train"]
    cli_args.exp_disabled = not cli_args.exp_enabled

    if "hong2" not in cli_args.ds_names:
        cli_args.hong2_ds_dir = None
    if "bremg" not in cli_args.ds_names:
        cli_args.bremg_ds_dir
    if "hong" not in cli_args.ds_names:
        cli_args.hong_ds_dir = None

    # print cli_args
    print(f"{cli_args=}")

    # print full df instead of truncated
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    main(cli_args)
