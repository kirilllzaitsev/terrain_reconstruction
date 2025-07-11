import gc
import typing as t
from collections import defaultdict

import numpy as np
import torch
from terrain_representation.utils.pointcloud_utils import match_point_clouds
from terrain_representation.utils.utils import TensorOrArr, TensorOrArrOrList
from terrain_synthesis.utils.elev_map_utils import fetch_pts_by_area_map_coords


def import_chamfer_dist(use_l1=True) -> t.Optional[t.Any]:
    try:
        from pointr.extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

        cd_l1 = ChamferDistanceL1()
        cd_l2 = ChamferDistanceL2()
    except ImportError:
        print("ChamferDistanceL1/L2 is not available")
        cd_l1 = None
        cd_l2 = None
    if use_l1:
        return cd_l1
    return cd_l2


try:
    from terrain_representation.losses.chamfer_distance import ChamferDistance
except Exception as e:
    print(f"Failed to import a custom ChamferDistance: {e}. Loading an alternative.")
    ChamferDistanceL1 = import_chamfer_dist()
    ChamferDistance = None


@torch.no_grad()
def compute_abs_errors_btw_point_clouds(
    x: TensorOrArrOrList, y: TensorOrArrOrList
) -> t.Dict[str, t.Any]:
    """Computes the mean absolute error (MAE) and Chamfer distance between two point clouds.
    Returns:
    - mae: mean absolute error
    - mae_height: mean absolute error of the height (third coordinate)
    - mae_granular: MAE for each point
    - mae_height_granular: MAE for each point's height
    - chamfer_dist: Chamfer distance
    - chamfer_dist_granular: Chamfer distance for each point
    """

    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y):
        y = torch.tensor(y)

    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    if len(y.shape) == 1:
        y = y.unsqueeze(0)

    has_batch_dim = len(x.shape) == 3
    try:
        matched_x, matched_y = match_point_clouds(x, y)
    except Exception as e:
        print(f"{e}.\nReturning NaN for metrics.")
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "mae": torch.nan,
            "mae_height": torch.nan,
            "mae_granular": torch.nan,
            "mae_height_granular": torch.nan,
            "chamfer_dist": torch.nan,
            "chamfer_dist_granular": torch.nan,
        }
    if has_batch_dim:
        batch_size = x.shape[0]
        dists = []
        height_diffs = []
        chamfer_dists = []
        for i in range(batch_size):
            dist = _compute_mae(matched_x[i], matched_y[i])
            dists.append(dist)
            height_diff = compute_height_error(matched_x[i], matched_y[i])
            chamfer_dist = compute_chamfer_dist(x[i], y[i])
            height_diffs.append(height_diff)
            chamfer_dists.append(chamfer_dist)
        dists = torch.stack(dists)
        height_diffs = torch.stack(height_diffs)
        chamfer_dists = torch.stack(chamfer_dists)
    else:
        dists = _compute_mae(matched_x, matched_y)
        height_diffs = compute_height_error(matched_x, matched_y)
        chamfer_dists = compute_chamfer_dist(x, y)

    return {
        "mae": torch.mean(dists),
        "mae_height": torch.mean(height_diffs),
        "mae_granular": dists,
        "mae_height_granular": height_diffs,
        "chamfer_dist": torch.mean(chamfer_dists),
        "chamfer_dist_granular": chamfer_dists,
    }


def _compute_mae(matched_x: torch.Tensor, matched_y: torch.Tensor) -> torch.Tensor:
    assert len(matched_x.shape) == len(matched_y.shape) == 2
    return torch.sqrt(torch.sum((matched_x - matched_y) ** 2, dim=1))


def compute_height_error(
    matched_x: torch.Tensor, matched_y: torch.Tensor
) -> torch.Tensor:
    assert len(matched_x.shape) == len(matched_y.shape) == 2
    return torch.abs(matched_x[:, 2] - matched_y[:, 2])


def compute_chamfer_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if ChamferDistance is None:
        if ChamferDistanceL1 is not None:
            return ChamferDistanceL1(x, y)
        return torch.tensor(torch.nan)
    chamfer_dist = ChamferDistance()
    has_batch_dim = len(x.shape) == 3
    if not has_batch_dim:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    dist1, dist2 = chamfer_dist(x, y)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    return loss


def compute_metrics_for_known_elevated_areas(
    pts_pred: torch.Tensor,
    pts_gt: torch.Tensor,
    area_map: t.Optional[np.ndarray],
    elev_area_names: t.List[str],
    min_pts_for_metrics: int = 90,
) -> t.Dict[str, t.Any]:
    """Computes metrics for elevated areas as defined by the area map. min_pts_for_metrics is the minimum number of points in the ground truth point cloud for the area to be considered as present. If an area has fewer points, it is skipped.
    Args:
        - pts_pred: predicted point cloud
        - pts_gt: ground truth point cloud
        - area_map: area map that accompanies the terrain from which the point clouds were obtained
        - elev_area_names: names of elevated areas in the area map (e.g., ["piles", "ramp"])
        - min_pts_for_metrics: minimum number of points in the ground truth point cloud for the area to be considered as present
    Returns:
        See `compute_abs_errors_btw_point_clouds` for the return values.
    """

    pts_pred = pts_pred.detach()
    if len(pts_pred.shape) == 1:
        pts_pred = pts_pred.unsqueeze(0)
    metrics = {
        "mae_granular": [],
        "mae_height_granular": [],
        "chamfer_dist_granular": [],
        "num_pts_pred": torch.tensor(0.0),
        "num_pts_gt": torch.tensor(0.0),
    }
    for area_name in elev_area_names:
        area_pts_pred = fetch_pts_by_area_map_coords(pts_pred, area_map, area_name)
        area_pts_gt = fetch_pts_by_area_map_coords(pts_gt, area_map, area_name)
        if len(area_pts_gt) < min_pts_for_metrics:
            # print(f"Skipping {area_name} due to too few points ({len(area_pts_gt)})")
            continue
        abs_errors = compute_abs_errors_btw_point_clouds(area_pts_pred, area_pts_gt)
        metrics["mae_granular"].append(abs_errors["mae_granular"])
        metrics["mae_height_granular"].append(abs_errors["mae_height_granular"])
        metrics["chamfer_dist_granular"].append(abs_errors["chamfer_dist_granular"])
        metrics["num_pts_pred"] += len(area_pts_pred)
        metrics["num_pts_gt"] += len(area_pts_gt)

    metrics["mae"] = torch.mean(torch.stack(metrics["mae_granular"]))
    metrics["mae_height"] = torch.mean(torch.stack(metrics["mae_height_granular"]))
    metrics["chamfer_dist"] = torch.mean(torch.stack(metrics["chamfer_dist_granular"]))
    return metrics


def is_elevated_known(
    pts_gt: torch.Tensor,
    area_map: t.Optional[np.ndarray],
    elev_area_names: t.List[str],
    min_pts_for_metrics: int = 90,
) -> bool:
    """Checks if the ground truth point cloud contains any known elevated areas."""
    if not area_map:
        return False

    for area_name in elev_area_names:
        area_pts_gt = fetch_pts_by_area_map_coords(pts_gt, area_map, area_name)
        if len(area_pts_gt) >= min_pts_for_metrics:
            return True
    return False
