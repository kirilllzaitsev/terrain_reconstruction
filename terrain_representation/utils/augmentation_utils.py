import copy
import typing as t

import numpy as np
import torch
from terrain_representation.utils.utils import TensorOrArrOrList
from terrain_synthesis.utils.elev_map_utils import slice_pts


def slice_height_of_area_map_region(
    pts: np.ndarray,
    area_map: t.Dict[str, t.List[t.Tuple[int, int, int, int]]],
    area_name: str,
    em_resolution: int = 10,
) -> np.ndarray:
    """Slices the height of the given point cloud within the given area map region.
    Args:
        pts: The point cloud.
        area_map: The area map in the format of {area_name: [(x, y, width, height)]}.
        area_name: The name of the area map region. See ds_synthesis module for more details.
        em_resolution: The resolution of the elevation map, "how many cells to cover 1 meter".
    """
    new_pts = copy.deepcopy(pts)
    for x, y, width, height in area_map[area_name]:
        mask = slice_pts(new_pts, x, y, width, height, em_resolution)
        if len(new_pts[mask]) > 0:
            new_pts[mask, 2] = torch.clip(new_pts[mask, 2], min=0, max=2.5)
    return new_pts


def mask_out_part_of_area_map_region(
    pts: np.ndarray,
    area_map: t.Dict[str, t.List[t.Tuple[int, int, int, int]]],
    area_name: str,
    blob_radius: int = 3,
    num_blobs: int = 1,
    em_resolution: int = 10,
) -> np.ndarray:
    """Masks out part of the given point cloud within the given area map region.
    Args:
        pts: The point cloud.
        area_map: The area map in the format of {area_name: [(x, y, width, height)]}.
        area_name: The name of the area map region. See ds_synthesis module for more details.
        blob_radius: The radius of the blob to be masked out.
        num_blobs: The number of blobs to be masked out.
        em_resolution: The resolution of the elevation map, "how many cells to cover 1 meter".
    """
    new_pts = copy.deepcopy(pts)
    blob_masks = []
    region_masks = []
    for x, y, width, height in area_map[area_name]:
        mask = slice_pts(new_pts, x, y, width, height, em_resolution)
        new_pts_slice = new_pts[mask]
        if len(new_pts_slice) == 0:
            continue
        _, blob_mask = mask_out_pts(new_pts_slice, blob_radius, num_blobs)
        blob_masks.append(blob_mask)
        region_masks.append(mask)
    mask = np.ones(len(new_pts), dtype=bool)
    for region_mask, blob_mask in zip(region_masks, blob_masks):
        mask[region_mask] = mask[region_mask] & blob_mask
    new_pts = new_pts[mask]
    return new_pts


def mask_out_pts(
    new_pts_slice: np.ndarray, blob_radius: int, num_blobs: int
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Masks out part of the given point cloud by adding random blobs.
    Returns the masked point cloud and the mask."""

    blob_centers = []
    for _ in range(num_blobs):
        random_blob_center = np.array(
            (
                np.random.uniform(
                    new_pts_slice[:, 0].min() + blob_radius,
                    new_pts_slice[:, 0].max() - blob_radius,
                    size=1,
                ),
                np.random.uniform(
                    new_pts_slice[:, 1].min() + blob_radius,
                    new_pts_slice[:, 1].max() - blob_radius,
                    size=1,
                ),
            )
        ).squeeze()
        blob_centers.append(random_blob_center)

    return mask_out_pts_with_known_blobs(new_pts_slice, blob_centers, blob_radius)


def mask_out_pts_with_known_blobs(
    new_pts_slice: np.ndarray, blob_centers: TensorOrArrOrList, blob_radius: int
) -> t.Tuple[np.ndarray, np.ndarray]:
    blob_masks = []
    for blob_center in blob_centers:
        if len(blob_center) == 2:
            dist_to_lidar_center = np.linalg.norm(
                new_pts_slice[:, :2] - blob_center, axis=1
            )
        else:
            dist_to_lidar_center = np.linalg.norm(
                new_pts_slice[:, :3] - blob_center, axis=1
            )
        blob_mask = dist_to_lidar_center > blob_radius
        blob_masks.append(blob_mask)
    mask = np.ones(len(new_pts_slice), dtype=bool)
    for blob_mask in blob_masks:
        mask = mask & blob_mask
    new_pts_slice_masked = new_pts_slice[mask]
    return new_pts_slice_masked, mask


def find_orthogonal_vectors(
    V: np.ndarray, global_up: np.ndarray = np.array([0, 0, 1])
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Finds two orthogonal vectors U and W to the given vector V."""

    # Check if V is parallel or anti-parallel to global_up
    if np.isclose(np.abs(np.dot(V, global_up)), 1.0, atol=1e-6):
        # Use an alternative up vector
        alternative_up = (
            np.array([1, 0, 0])
            if np.isclose(V[0], 0, atol=1e-6)
            else np.array([0, 1, 0])
        )
        U = np.cross(V, alternative_up)
    else:
        # Project global_up onto the plane perpendicular to V
        projection = global_up - np.dot(global_up, V) * V
        U = projection

    U = U / np.linalg.norm(U)
    W = np.cross(V, U)
    return U, W


def gen_pts_on_arc(
    center: np.ndarray,
    arc_angle_deg: float,
    radius: float,
    viewing_dir: np.ndarray,
    num_pts: int,
) -> t.List[np.ndarray]:
    """Generates random points on an arc with the given center, radius, and viewing direction."""
    arc_angle = np.deg2rad(arc_angle_deg)
    U, W = find_orthogonal_vectors(viewing_dir)

    random_points = []
    for _ in range(num_pts):
        theta = np.random.uniform(0, arc_angle)  # Random angle in [0, pi]
        P = center + radius * (np.cos(theta) * U + np.sin(theta) * W)
        random_points.append(P)
    return random_points


def conv_xyxy_to_center(sample_obstacles: np.ndarray, z: float = 4.5) -> np.ndarray:
    """Converts the given obstacles from (bottom_left_x, bottom_left_y, top_right_x, top_right_y) to (center_x, center_y, z)."""

    if not isinstance(sample_obstacles, np.ndarray):
        sample_obstacles = np.array(sample_obstacles)
    if len(sample_obstacles.shape) < 2:
        sample_obstacles = np.atleast_2d(sample_obstacles)
    return np.vstack(
        [
            (sample_obstacles[:, 0] + sample_obstacles[:, 2]) / 2,
            (sample_obstacles[:, 1] + sample_obstacles[:, 3]) / 2,
            np.ones(len(sample_obstacles)) * z,
        ]
    ).T
