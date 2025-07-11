import typing as t

import matplotlib.pyplot as plt
import torch
from terrain_representation.utils.utils import TensorOrArr, transfer_batch_to_device
from terrain_representation.utils.visualization import (
    PointCloudVisualizer,
    PointCloudVisualizerAsync,
)
from terrain_representation.utils.voxel_grid_utils import (
    convert_batch_of_pts_to_vg,
    convert_pred_vg_to_pcl,
    convert_vg_to_pcl,
    voxel_grid_to_point_cloud,
)
from terrain_synthesis.utils.vis_utils import plot_point_cloud, plot_point_cloud_pyplot


def display_model_preds(
    visualizer: t.Union[PointCloudVisualizerAsync, PointCloudVisualizer],
    net_input: t.Any,
    data_dict: dict,
    preds: dict,
    map_dim: TensorOrArr,
    map_resolution: TensorOrArr,
    threshold: float,
    take_idx: int = 1,
    log_img_name: t.Optional[str] = None,
    step=None,
):
    """Helps to display the entire sample: input, output, ground truth, and occupancy.
    Args:
        visualizer: Visualizer object.
        net_input: Network input.
        data_dict: Data dictionary.
        preds: Predictions dictionary containing 'centroids' and 'occupancy_logits'.
        map_dim: Map dimensions.
        map_resolution: Map resolution.
        threshold: Threshold for the occupancy.
        take_idx: Index of the sample to take from the batch.
        log_img_name: Name of the log image.
        step: Step number.
    Returns:
        Rendered image or None (for async visualizer)."""
    centroids = preds["centroids"][take_idx].clone().detach()
    occupancy_logits = preds["occupancy_logits"][take_idx].clone().detach()
    net_output_pcl = convert_pred_vg_to_pcl(
        map_dim,
        map_resolution,
        threshold,
        centroids=centroids,
        occupancy_logits=occupancy_logits,
    )
    measured_vg = data_dict["measured"][take_idx].clone().detach()
    gt_vg = data_dict["gt"][take_idx].clone().detach()

    occupancy_probs = torch.sigmoid(occupancy_logits)[0]
    occupancy_points = convert_vg_to_pcl(
        centroids,
        map_dim,
        map_resolution,
        mask=occupancy_probs >= 0.5,
    )
    occupancy_values = occupancy_probs[occupancy_probs >= 0.5]

    net_input_pcl = convert_net_input_to_pcl(
        net_input, map_dim, map_resolution, take_idx
    )

    # identical original data is different in a batch data_dict["measured"]. why?
    measured_pcl = convert_vg_to_pcl(measured_vg, map_dim, map_resolution)

    gt_pcl = convert_vg_to_pcl(gt_vg, map_dim, map_resolution)

    vis_data_dict = {
        "measurement": measured_pcl,
        "input": net_input_pcl,
        "occupancy": {
            "occupancy_points": occupancy_points,
            "occupancy_values": occupancy_values,
        },
        "output": net_output_pcl,
        "gt": gt_pcl,
        "log_img_name": log_img_name,
        "step": step,
    }
    if data_dict.get("mesh") is not None:
        mesh = convert_vg_to_pcl(data_dict["mesh"][take_idx], map_dim, map_resolution)
        vis_data_dict["mesh"] = mesh

    return visualizer.display_point_clouds(vis_data_dict)


def convert_net_input_to_pcl(
    net_input: t.Any, map_dim: TensorOrArr, map_resolution: TensorOrArr, take_idx: int
) -> torch.Tensor:
    """Converts the 6D input (current measurement and the previous prediction) of the network to a point cloud."""
    inp = net_input[take_idx].clone().unsqueeze(-1).transpose(-1, 0)[0]
    inp[inp <= 0.0] = -1.0
    has_prev = inp.shape[-1] == 6

    if has_prev:
        inp_meas = convert_vg_to_pcl(inp[..., :3], map_dim, map_resolution)
        # inp_prev = convert_vg_to_pcl(inp[..., 3:], map_dim, map_resolution)
        # pc_inp = torch.cat((inp_meas, inp_prev), dim=0)
        pc_inp = inp_meas
    else:
        pc_inp = convert_vg_to_pcl(inp, map_dim, map_resolution)
    return pc_inp


def plot_ds_samples(
    ds,
    num_samples=20,
    do_overlay=False,
    do_random=False,
    do_follow_training=False,
    map_dim=None,
    map_resolution=None,
    device="cpu",
):
    num_cols = min(5, num_samples)
    num_rows = max(1, num_samples // num_cols)
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(max(15, num_samples), max(5, num_samples // 2))
    )
    idxs = range(num_samples)
    if do_random:
        idxs = torch.randperm(len(ds))[:num_samples]
    for i, idx in enumerate(idxs):
        idx = idx.item()
        sample = ds[idx]
        if do_follow_training:
            sample = transfer_batch_to_device(sample, device)
            sample = convert_batch_of_pts_to_vg(sample, map_dim, map_resolution)
        measured = sample["measured"][0]
        mesh = sample["mesh"][0]
        if do_follow_training:
            measured = convert_vg_to_pcl(measured, map_dim, map_resolution)
            mesh = convert_vg_to_pcl(mesh, map_dim, map_resolution)
        if num_rows == 1:
            ax = axs[i % num_cols]
        else:
            ax = axs[i // num_cols, i % num_cols]
        if do_overlay:
            plot_point_cloud_pyplot(mesh, centers=measured, ax=ax)
        else:
            plot_point_cloud_pyplot(measured, ax=ax)
        title = f"{idx=}.{sample['traj_path']}"
        ax.set_title(title)
        if i != 0:
            ax.axis("off")


def vis_point_based(
    visualizer,
    data_dict,
    measured,
    pcds_pred_denorm,
    target_denorm,
    log_img_name,
    do_vis_entire_batch=False,
    use_pc_norm=True,
    step=0,
):
    if do_vis_entire_batch:
        take_idxs = range(len(measured))
    else:
        take_idxs = [0]
    for take_idx in take_idxs:
        pts_mean = data_dict["mean"][take_idx]
        pts_max_dist = data_dict["max_dist"][take_idx]
        if use_pc_norm:
            vis_measured = measured[take_idx] * pts_max_dist + pts_mean
        else:
            vis_measured = measured[take_idx]
        vis_output = pcds_pred_denorm[take_idx].detach()
        vis_target = target_denorm[take_idx]
        vis_data_dict = {
            "measurement": vis_measured,
            "input": vis_measured,
            "output": vis_output,
            "gt": vis_target,
            "log_img_name": f"{log_img_name}_sample_{take_idx}",
            "step": step,
        }
        if data_dict.get("mesh") is not None:
            if use_pc_norm:
                vis_data_dict["mesh"] = (
                    data_dict["mesh"][take_idx] * pts_max_dist + pts_mean
                )
            else:
                vis_data_dict["mesh"] = data_dict["mesh"][take_idx]

        visualizer.display_point_clouds(vis_data_dict)


def vis_gen_based(
    visualizer,
    measured,
    log_img_name,
    do_vis_entire_batch=False,
    step=0,
):
    if do_vis_entire_batch:
        take_idxs = range(len(measured))
    else:
        take_idxs = [0]
    for take_idx in take_idxs:
        visualizer.display_single_point_cloud(
            measured[take_idx], title=f"{log_img_name}_sample_{take_idx}", step=step
        )
