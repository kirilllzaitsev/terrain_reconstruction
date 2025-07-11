from pathlib import Path

import numpy as np
from terrain_representation.utils.pointcloud_utils import (
    downsample_point_cloud,
    filter_pts_within_azimuth_angle_from_center,
    get_dist_stats_of_pts,
    slice_pcl_by_azimuth_angle,
)
from terrain_synthesis.utils.postprocessing_utils import generate_scan_angles_from_to


def find_lidar_camera_paths_pair_by_idx(
    base_dir, ts_mapping, cam_path=None, lidar_path=None
):
    """
    print(ts_mapping['matching_pairs'][0])
    {'pcl timestamp': 1646744641245,
    'pcl filename': '1646744641.245.bin',
    'image timestamp': 1646744641257,
    'image filename': '1646744641.257.png'}
    """
    assert cam_path is not None or lidar_path is not None
    if cam_path is not None:
        if "." not in cam_path:
            cam_path = f"{cam_path}.png"
        cam_path = Path(cam_path)
        # idx = ts_mapping['image_filenames'].index(cam_path.name)
        idx = [
            i
            for i, d in enumerate(ts_mapping["matching_pairs"])
            if d["image filename"] == cam_path.name
        ][0]
    else:
        if "." not in lidar_path:
            lidar_path = f"{lidar_path}.bin"
        lidar_path = Path(lidar_path)
        idx = [
            i
            for i, d in enumerate(ts_mapping["matching_pairs"])
            if d["pcl filename"] == lidar_path.name
        ][0]
    lidar_path, camera_path = fetch_paths_by_idx_from_ts_mapping(
        base_dir, ts_mapping, idx
    )
    return lidar_path, camera_path


def fetch_paths_from_ts_mapping(base_dir, ts_mapping):
    idxs = list(range(len(ts_mapping["matching_pairs"])))
    return fetch_paths_by_idx_from_ts_mapping(base_dir, ts_mapping, idxs=idxs)


def fetch_paths_by_idx_from_ts_mapping(base_dir, ts_mapping, idx=None, idxs=None):
    if idx is not None:
        return _fetch_paths_by_idx_from_ts_mapping(base_dir, ts_mapping, idx)
    elif idxs is not None:
        return [
            _fetch_paths_by_idx_from_ts_mapping(base_dir, ts_mapping, idx)
            for idx in idxs
        ]
    else:
        raise ValueError("Either idx or idxs should be provided")


def _fetch_paths_by_idx_from_ts_mapping(base_dir, ts_mapping, idx):
    base_dir = Path(base_dir)
    lidar_path = base_dir / "lidar" / ts_mapping["matching_pairs"][idx]["pcl filename"]
    camera_path = (
        base_dir / "camera" / ts_mapping["matching_pairs"][idx]["image filename"]
    )

    return lidar_path, camera_path


def parse_real_lidar_sample(s: np.ndarray) -> tuple:
    # Returns downsampled point cloud and the intensity values
    return downsample_point_cloud(s[:, :3], 0.01), s[:, 3]


def parse_path(p: str) -> tuple:
    # See parse_real_lidar_sample
    return parse_real_lidar_sample(np.fromfile(p, dtype=np.float32).reshape(-1, 4))


def slice_frontal_scan(pts_in_range, lidar_center):
    mean_pt = np.mean(pts_in_range, axis=0)
    view_vector = mean_pt - lidar_center
    front_angle_rad = np.deg2rad(180)

    chunk_pts = filter_pts_within_azimuth_angle_from_center(
        pts_in_range, lidar_center, view_vector, front_angle_rad
    )
    utmost_left_pt, utmost_right_pt = get_dist_stats_of_pts(chunk_pts)[
        "most_distant_pts"
    ]
    utmost_left_pt_angle = (
        np.arctan2(utmost_left_pt[1], utmost_left_pt[0]) * 180 / np.pi
    )
    utmost_right_pt_angle = (
        np.arctan2(utmost_right_pt[1], utmost_right_pt[0]) * 180 / np.pi
    )
    if utmost_left_pt_angle > utmost_right_pt_angle:
        utmost_left_pt_angle, utmost_right_pt_angle = (
            utmost_right_pt_angle,
            utmost_left_pt_angle,
        )
        #

    scan_angles = generate_scan_angles_from_to(
        num_scans=3,
        start_angle=utmost_left_pt_angle,
        end_angle=utmost_right_pt_angle,
        margin_angle=30,
    )
    angle_slice_deg = 120
    t = lidar_center
    real_slices = slice_pcl_by_azimuth_angle(
        pts_in_range, angle_slice_deg, t, chunk_angles=scan_angles
    )

    return real_slices
