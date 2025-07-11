import typing as t

import numpy as np
import torch
from terrain_synthesis.utils.elev_map_utils import slice_pts


def contains_elev_area(
    pts: np.ndarray,
    elev_elems: t.List[t.Tuple[int, int, int, int]],
    min_elems_to_be_considered_in_area: int = 1,
    area_region_z_upper_limit: float = 1.0,
    area_region_z_lower_limit: float = 0.0,
    threshold_percentage_of_elev_pts: float = 0.9,
) -> bool:
    """Estimates via known elev_elems if the given point cloud contains a reasonable portion of elevated areas.
    Args:
        pts: The point cloud.
        elev_elems: The list of elevated elements in the form of (bottom_left_x, bottom_left_y, width, height).
        min_elems_to_be_considered_in_area: The minimum number of elevated elements to be considered an elevated area.
        area_region_z_upper_limit: The upper limit of the z-coordinate for the area to be considered elevated.
        area_region_z_lower_limit: The lower limit of the z-coordinate for the area to be considered elevated.
        threshold_percentage_of_elev_pts: The threshold percentage of elevated points with each elevated element.
    """

    counter = 0
    for x, y, width, height in elev_elems:
        mask = slice_pts(pts, x, y, width, height, 10)
        if is_elevated(
            pts[mask],
            flat_region_z_upper_limit=area_region_z_upper_limit,
            flat_region_z_lower_limit=area_region_z_lower_limit,
            threshold_percentage_of_pts=threshold_percentage_of_elev_pts,
        ):
            counter += 1
            if counter >= min_elems_to_be_considered_in_area:
                return True
    return False


def is_elevated(
    pts: np.ndarray,
    flat_region_z_upper_limit: float = 1.0,
    flat_region_z_lower_limit: float = 0.0,
    threshold_percentage_of_pts: float = 0.2,
    area_map: t.Optional[t.Dict[str, t.List[t.Tuple[int, int, int, int]]]] = None,
    elev_region_names: t.Optional[t.List[str]] = None,
    min_elems_to_be_considered_in_area: int = 1,
) -> bool:
    """Estimates via heuristic if the given point cloud is elevated. Optionally, it can use an area map to check if the point cloud contains a reasonable portion of elevated areas (see contains_elev_area).
    Args:
        See contains_elev_area.
    """

    mask = (pts[:, 2] > flat_region_z_upper_limit) | (
        pts[:, 2] < flat_region_z_lower_limit
    )
    if area_map is not None:
        assert (
            elev_region_names is not None
        ), "elev_region_names must be provided if area_map is provided"
        for area_name in elev_region_names:
            if contains_elev_area(
                pts,
                area_map[area_name],
                min_elems_to_be_considered_in_area=min_elems_to_be_considered_in_area,
                # should be area-specific
                area_region_z_upper_limit=flat_region_z_upper_limit,
                area_region_z_lower_limit=flat_region_z_lower_limit,
                threshold_percentage_of_elev_pts=0.9,
            ):
                print(f"elevated area: {area_name}")
                return True
    if len(pts[mask]) == 0:
        return False
    if isinstance(pts, torch.Tensor):
        lib = torch
    else:
        lib = np
    if lib.sum(mask) / pts.shape[0] > threshold_percentage_of_pts:
        return True
    return False
