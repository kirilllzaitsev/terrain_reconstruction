import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from terrain_representation.utils.utils import TensorOrArr


def slice_random_lidar_scan_from_pts(
    pts: TensorOrArr,
    lidar_range: float = 10.0,
    lidar_blind_range: float = 5.0,
    robot_height: float = 4.0,
) -> t.Tuple[TensorOrArr, TensorOrArr]:
    """Creates a random slice of the point cloud that would be seen by a lidar sensor. The sensor is placed at a random position and the points which fall within its range, while being outside of the blind range, are returned.
    Returns:
        scan: The slice of the point cloud that would be seen by the lidar sensor.
        lidar_sensor_pos_closest_pt: The closest to the lidar sensor position point from the point cloud.
    """
    lidar_sensor_pos = np.array(
        (
            np.random.uniform(
                pts[:, 0].min() + lidar_range, pts[:, 0].max() - lidar_range, size=1
            ),
            np.random.uniform(
                pts[:, 1].min() + lidar_range, pts[:, 1].max() - lidar_range, size=1
            ),
        )
    ).squeeze()
    lidar_sensor_pos_closest_pt = pts[
        np.linalg.norm(pts[:, :2] - lidar_sensor_pos, axis=1).argmin()
    ]
    lidar_sensor_pos_closest_pt[2] = robot_height

    scan = fetch_pts_in_lidar_range(
        pts, lidar_sensor_pos, lidar_range, lidar_blind_range
    )
    return scan, lidar_sensor_pos_closest_pt


def fetch_pts_in_lidar_range(
    pts: TensorOrArr,
    lidar_center: TensorOrArr,
    lidar_range: float = 10.0,
    lidar_blind_range: float = 5.0,
    use_2d=True,
) -> TensorOrArr:
    """Returns the points that are within the range and outside of the blind range of the lidar sensor."""
    if use_2d:
        axis = 2
    else:
        axis = 3
    if len(lidar_center) == 3:
        lidar_center = lidar_center.squeeze()[:axis]
    if isinstance(pts, torch.Tensor):
        if not isinstance(lidar_center, torch.Tensor):
            lidar_center = torch.tensor(
                lidar_center, dtype=pts.dtype, device=pts.device
            )
        dist_to_lidar_center = torch.linalg.norm(pts[:, :axis] - lidar_center, dim=1)
    else:
        dist_to_lidar_center = np.linalg.norm(pts[:, :axis] - lidar_center, axis=1)
    pts_in_range = pts[
        (dist_to_lidar_center < lidar_range)
        & (dist_to_lidar_center > lidar_blind_range)
    ]
    return pts_in_range


def fetch_pts_in_lidar_range_natural(
    pts: TensorOrArr,
    lidar_center: TensorOrArr,
    lidar_range: float = 10.0,
    lidar_blind_range: float = 5.0,
    show_extended_info: bool = False,
) -> TensorOrArr:
    """Similar to fetch_pts_in_lidar_range, but the points are sampled in a way that resembles the natural distribution of points that would be seen by a lidar sensor. Some heuristics are used to achieve this, instead of a rigorous physical model."""

    # not only selects points within the range, but also:
    # 1. point density is higher near the center of the lidar
    # 2. the slice consists of circles of increasing radius
    if len(lidar_center) == 3:
        lidar_center = lidar_center[:2]
    dist_to_lidar_center = np.linalg.norm(pts[:, :2] - lidar_center, axis=1)
    pts_in_range = pts[
        (dist_to_lidar_center < lidar_range)
        & (dist_to_lidar_center > lidar_blind_range)
    ]

    num_circles = 7
    min_radius = lidar_blind_range
    # gap_between_circles = 0.5
    circle_radius_delta = (lidar_range - min_radius) / (num_circles - 1)
    circle_radius = np.linspace(
        min_radius, lidar_range - circle_radius_delta, num_circles
    )

    dist_to_lidar_center = np.linalg.norm(pts_in_range[:, :2] - lidar_center, axis=1)
    natural_pts_in_range = []
    sampling_rates = []
    for i, r in enumerate(circle_radius):
        # sampling_rate = 1 / (i + 1)
        # sampling_rate = np.exp(-i/num_circles)
        sampling_rate = 1 / (np.exp(i**0.5 / num_circles))
        # gap_between_circles directly proportional to the radius
        gap_between_circles = r / (10 + num_circles)
        if show_extended_info:
            print(f"{sampling_rate=}")
            print(f"{r=}")
            print(f"{gap_between_circles=}")
        pts_in_circle = pts_in_range[
            (dist_to_lidar_center < r + circle_radius_delta)
            & (dist_to_lidar_center > r + gap_between_circles)
        ]
        sampled_pts = pts_in_circle[
            np.random.choice(
                pts_in_circle.shape[0],
                int(pts_in_circle.shape[0] * sampling_rate),
                replace=False,
            )
        ]
        natural_pts_in_range.append(sampled_pts)
        sampling_rates.append(sampling_rate)

    # double y-axis plot with: x-axis for circle index, y-axis (right) for sampling rate, y-axis (left) for number of points
    if show_extended_info:

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(range(num_circles), [len(pts) for pts in natural_pts_in_range], "g")
        ax2.plot(range(num_circles), sampling_rates, "r")
        ax1.set_ylabel("Number of points", color="green")
        ax2.set_ylabel("Sampling rate", color="red")
        ax1.set_xlabel("Circle index")
        ax1.set_label("Number of points")
        ax2.set_label("Sampling rate")
        plt.title("Number of points and sampling rate per circle")
        plt.show()

    natural_pts_in_range = np.vstack(natural_pts_in_range)

    return natural_pts_in_range
