import typing as t

import numpy as np
import torch
from terrain_representation.utils.utils import TensorOrArr, TensorOrArrOrList
from torch import Tensor

convert_pts_to_vg_cache = {}


def occupancy_mask(vg):
    assert vg.shape[-1] == 3
    return vg[..., 0] > 0.0


def point_cloud_to_coords_and_feats(
    pointcloud: TensorOrArrOrList,
    map_dim: TensorOrArrOrList,
    map_resolution: TensorOrArrOrList,
) -> "tuple[Tensor, Tensor]":
    """Converts a point cloud to coordinates and features used for voxel grid creation."""
    if not isinstance(map_dim, torch.Tensor):
        map_dim = torch.tensor(map_dim)
    if not isinstance(map_resolution, torch.Tensor):
        map_resolution = torch.tensor(map_resolution)
    if isinstance(pointcloud, list) or len(pointcloud.shape) == 3:
        return [
            _point_cloud_to_coords_and_feats(p, map_dim, map_resolution)
            for p in pointcloud
        ]
    return _point_cloud_to_coords_and_feats(pointcloud, map_dim, map_resolution)


def _point_cloud_to_coords_and_feats(
    pointcloud: TensorOrArr,
    map_dim: TensorOrArrOrList,
    map_resolution: TensorOrArrOrList,
) -> "tuple[Tensor, Tensor]":
    """Converts a point cloud to coordinates and features used for voxel grid creation."""
    assert len(pointcloud.shape) == 2, "Point cloud must be of shape (N, 3)"
    if not isinstance(pointcloud, torch.Tensor):
        pointcloud = torch.as_tensor(pointcloud)
    if map_dim.device != pointcloud.device:
        map_dim = map_dim.to(pointcloud.device)
    if map_resolution.device != pointcloud.device:
        map_resolution = map_resolution.to(pointcloud.device)
    measured = pointcloud / map_resolution + map_dim / 2
    feats = measured % 1.0
    coords = (measured - feats).long()

    mask = torch.logical_and(coords >= 0, coords < map_dim[0]).all(1)
    coords = coords[mask]
    feats = feats[mask]
    return coords, feats


def postprocess_coords_and_feats(coords, feats, use_xy_centroids=False):
    # returns unique coords of cells with points and the mean of the points that fall into the same cell
    unique_coords, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)
    device = coords.device

    count_points = torch.zeros(
        unique_coords.size(0), dtype=torch.float32, device=device
    )

    count_points = count_points.scatter_add(
        0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32)
    )

    dims_to_agg = 1 if use_xy_centroids else 3
    sum_offsets = torch.zeros(
        (unique_coords.size(0), dims_to_agg), dtype=torch.float32, device=device
    )
    if use_xy_centroids:
        # do the same as for XYZ, but only for Z coordinate. others are set to 0.5
        offsets = feats[:, 2].float().unsqueeze(-1)
    else:
        offsets = feats.float()

    sum_offsets = sum_offsets.scatter_add(
        0, inverse_indices.unsqueeze(-1).expand(-1, dims_to_agg), offsets
    )
    mean_offsets = sum_offsets / count_points.unsqueeze(-1)
    if use_xy_centroids:
        mean_offsets = mean_offsets.repeat(1, 3)
        mean_offsets[:, :2] = 0.5

    return unique_coords, mean_offsets


def convert_pts_to_vg(
    pcl: TensorOrArrOrList,
    map_dim: TensorOrArrOrList,
    map_resolution: TensorOrArrOrList,
    use_xy_centroids: bool = False,
) -> torch.Tensor:
    """Converts a point cloud to a voxel grid."""
    if isinstance(pcl, list) or len(pcl.shape) == 3:
        res = [
            _convert_pts_to_vg(pcl_i, map_dim, map_resolution, use_xy_centroids)
            for pcl_i in pcl
        ]
        if isinstance(pcl, torch.Tensor):
            res = torch.stack(res, dim=0)
        return res
    return _convert_pts_to_vg(pcl, map_dim, map_resolution, use_xy_centroids)


def _convert_pts_to_vg(
    pcl: TensorOrArr,
    map_dim: TensorOrArrOrList,
    map_resolution: TensorOrArrOrList,
    use_xy_centroids: bool,
) -> torch.Tensor:
    """Converts a point cloud to a voxel grid."""

    assert len(pcl.shape) == 2, f"Point cloud must be of shape (N, 3). Got {pcl.shape}"
    if isinstance(pcl, np.ndarray):
        pcl = torch.tensor(pcl)

    if "measured" in convert_pts_to_vg_cache:
        measured = convert_pts_to_vg_cache["measured"].clone()
    else:
        measured = torch.zeros(
            map_dim[0],
            map_dim[1],
            map_dim[2],
            3,
            device=pcl.device,
        )
        measured.fill_(-1.0)
        convert_pts_to_vg_cache["measured"] = measured.clone()
    coords, feats = point_cloud_to_coords_and_feats(pcl, map_dim, map_resolution)
    coords, feats = postprocess_coords_and_feats(coords, feats, use_xy_centroids)
    measured[
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
    ] = feats.float()

    return measured


def convert_pred_vg_to_pcl(
    map_dim: TensorOrArr,
    map_resolution: TensorOrArr,
    threshold: float,
    centroids: torch.Tensor,
    occupancy_logits: torch.Tensor,
):
    """Converts the predicted voxel grid to a point cloud."""
    occupancy_probs = torch.sigmoid(occupancy_logits).squeeze(dim=0)
    pc_out = voxel_grid_to_point_cloud(
        centroids,
        occupancy_probs >= threshold,
        map_dim,
        map_resolution,
    )

    return pc_out


def convert_vg_to_pcl(
    vg: torch.Tensor,
    map_dim: TensorOrArrOrList,
    map_resolution: TensorOrArrOrList,
    mask: t.Any = None,
):
    if not isinstance(vg, torch.Tensor):
        vg = torch.tensor(vg)
    if mask is None:
        mask = vg[..., 0] > 0.0
    pcl = voxel_grid_to_point_cloud(vg, mask, map_dim, map_resolution)

    return pcl


def voxel_grid_to_point_cloud(
    voxel_grid: Tensor,
    is_occupied: Tensor,
    map_dim: TensorOrArrOrList,
    map_resolution: TensorOrArrOrList,
) -> t.Union[Tensor, t.List[Tensor]]:
    """Wraps the _voxel_grid_to_point_cloud function to handle batched inputs."""
    if not isinstance(map_dim, torch.Tensor):
        map_dim = torch.tensor(map_dim)
    if not isinstance(map_resolution, torch.Tensor):
        map_resolution = torch.tensor(map_resolution)
    # Unoccupied voxels have a feature value of -1.0
    # selector = voxel_grid[..., 0] > 0.0001
    if len(voxel_grid.shape) == 5:
        res = []
        for i, vg in enumerate(voxel_grid):
            res.append(
                _voxel_grid_to_point_cloud(vg, is_occupied[i], map_dim, map_resolution)
            )
        if all([r.shape == res[0].shape for r in res[1:]]):
            res = torch.stack(res, dim=0)
    else:
        res = _voxel_grid_to_point_cloud(
            voxel_grid, is_occupied, map_dim, map_resolution
        )
    return res


def _voxel_grid_to_point_cloud(
    voxel_grid: Tensor,
    is_occupied: Tensor,
    map_dim: Tensor,
    map_resolution: Tensor,
) -> Tensor:
    """Converts a voxel grid to a point cloud.
    Args:

        voxel_grid: The voxel grid to convert.
        is_occupied: A boolean mask indicating which voxels are occupied.
        map_dim: The axis dimensions of the voxel grid.
        map_resolution: The cell sizes of the voxel grid.
    """

    assert (
        len(voxel_grid.shape) != 5
    ), "voxel grids with batch_size > 1 not supported. See voxel_grid_to_point_cloud for batched inputs."

    if map_dim.device != voxel_grid.device:
        map_dim = map_dim.to(voxel_grid.device)
    if map_resolution.device != voxel_grid.device:
        map_resolution = map_resolution.to(voxel_grid.device)
    if is_occupied.device != voxel_grid.device:
        is_occupied = is_occupied.to(voxel_grid.device)

    is_occupied = is_occupied.squeeze(0)
    feats = voxel_grid[is_occupied]
    coords = is_occupied.nonzero().float()
    pointcloud = (coords + feats - map_dim / 2) * map_resolution
    return pointcloud


def align_vgs(*args, target):
    if isinstance(args[0], list):
        return [_align_vgs(*args_, target=me) for args_, me in zip(args, target)]
    return _align_vgs(*args, target=target)


def _align_vgs(*args, target):
    assert target.ndim == 4
    for vg in args:
        vg[target[..., 0] < 0] = -1.0
    return args


def convert_batch_of_pts_to_vg(batch, map_dim, map_resolution, use_xy_centroids=False):
    for traj_idx in range(len(batch["measured"])):
        for k in ["measured"]:
            batch[k][traj_idx] = convert_pts_to_vg(
                batch[k][traj_idx], map_dim, map_resolution, use_xy_centroids=False
            )
        for k in ["gt", "mesh"]:
            batch[k][traj_idx] = convert_pts_to_vg(
                batch[k][traj_idx],
                map_dim,
                map_resolution,
                use_xy_centroids=use_xy_centroids,
            )
    for traj_idx in range(len(batch["measured"])):
        align_vgs(
            batch["measured"][traj_idx],
            batch["gt"][traj_idx],
            target=batch["mesh"][traj_idx],
        )
    return batch


def merge_vgs_in_common_frame(m1_vg, m2_vg):
    m1_vg_mask = m1_vg > 0
    m2_vg_mask = m2_vg > 0
    merged_vg = m1_vg.clone()
    merged_vg[m2_vg_mask & ~m1_vg_mask] = m2_vg[m2_vg_mask & ~m1_vg_mask]
    return merged_vg
