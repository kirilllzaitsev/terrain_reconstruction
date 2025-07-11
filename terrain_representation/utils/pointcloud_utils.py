import math
import typing as t

import scipy

try:
    import MinkowskiEngine as ME
except ImportError:
    pass
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Open3D is not installed")
import torch
from scipy.spatial.transform import Rotation as R
from terrain_representation.utils.sensor_utils import fetch_pts_in_lidar_range
from terrain_representation.utils.utils import (
    DeviceType,
    TensorOrArr,
    TensorOrArrOrList,
    pick_library,
)
from terrain_synthesis.utils.preprocessing_utils import fuse_point_clouds_in_world
from terrain_synthesis.utils.vis_utils import plot_point_cloud
from torch import Tensor


def transform_point_cloud(pointcloud_A: Tensor, pose_BA: Tensor) -> Tensor:
    """Transforms the pointcloud from frame B to A"""
    pointcloud_B = (
        torch.mm(pointcloud_A, pose_BA[:3, :3].transpose(0, 1)) + pose_BA[:3, -1]
    )
    return pointcloud_B


def sparse_tensor_to_point_cloud(
    sparse_tensor: "ME.SparseTensor", batch_idx: int, map_resolution: Tensor
) -> Tensor:
    """Converts a sparse tensor to a point cloud."""
    assert len(map_resolution) == 3, "map_resolution must be a 3x1 tensor"
    return (
        sparse_tensor.coordinates_at(batch_idx)[:, :3].float()
        + sparse_tensor.features_at(batch_idx)
    ) * map_resolution


def point_clouds_to_sparse_tensor(
    map_resolution: Tensor,
    point_clouds: t.List[Tensor],
    time_idx: int,
    coordinate_manager: "ME.CoordinateManager" = None,
) -> "ME.SparseTensor":
    """Converts a list of point clouds to a sparse tensor."""
    device = point_clouds[0].device
    feats = [(point_cloud / map_resolution) % 1 for point_cloud in point_clouds]
    coords = [
        torch.cat(
            (
                (point_cloud / map_resolution).floor().int(),
                time_idx
                * torch.ones((len(point_cloud), 1), dtype=torch.int, device=device),
            ),
            dim=1,
        )
        for point_cloud in point_clouds
    ]
    coords, feats = ME.utils.sparse_collate(coords=coords, feats=feats, device=device)
    tensor = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        coordinate_manager=coordinate_manager,
        device=device,
    )

    return tensor


def xyzrpy_to_transform(xyz: Tensor, rpy: Tensor) -> Tensor:
    """Compute the transformation matrix from the tuple x,y,z, roll, pitch, yaw"""
    cr, sr = torch.cos(rpy[0]), torch.sin(rpy[0])
    cp, sp = torch.cos(rpy[1]), torch.sin(rpy[1])
    cy, sy = torch.cos(rpy[2]), torch.sin(rpy[2])
    rot_mat = torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
            [-sp, cp * sr, cp * cr, xyz[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=rpy.device,
    )

    return rot_mat


def normalize(x: Tensor, eps: float = 1e-9) -> Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw component of a quaternion.

    Args:
        quat (torch.Tensor): Input orientation to extract yaw from.

    Returns:
        torch.Tensor: quat.
    """
    quat_yaw = quat.clone().view(-1, 4)
    qx = quat_yaw[:, 0]
    qy = quat_yaw[:, 1]
    qz = quat_yaw[:, 2]
    qw = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:, :2] = 0.0
    quat_yaw[:, 2] = torch.sin(yaw / 2)
    quat_yaw[:, 3] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw


def random_box(device: DeviceType) -> Tensor:
    """Creates a randomly sized cube."""
    box_len = torch.randint(2, 15, (1,)).item()
    box_width = torch.randint(1, 5, (1,)).item()
    box_height = torch.randint(15, 30, (1,)).item()
    xx, yy, zz = torch.meshgrid(
        torch.arange(-box_len // 2, box_len // 2, device=device),
        torch.arange(-box_width // 2, box_width // 2, device=device),
        torch.arange(-box_height // 2, box_height // 2, device=device),
        indexing="xy",
    )
    box = torch.stack((xx, yy, zz), dim=-1).view(-1, 3).float().repeat(2, 1) * 0.05
    # perturb the points uniformly
    box_noise = torch.zeros((len(box), 3), device=device).uniform_(-0.025, 0.025)
    box += box_noise
    # give the cube in a random orientation
    box_orientation = torch.zeros((3,), device=device).uniform_(-3.1415, 3.1415)
    box_position = torch.zeros((3,), device=device)

    transform = xyzrpy_to_transform(box_position, box_orientation)
    box = transform_point_cloud(box, transform)
    return box


def point_cloud_to_heightmap(
    points: Tensor,
    map: Tensor,
    map_dim: Tensor = torch.tensor([64, 64, 64]),
    map_resolution_xyz: TensorOrArrOrList = [0.05, 0.05, 0.05],
) -> Tensor:
    """Converts a point cloud to a heightmap."""
    # map = torch.zeros((map_dim[0], map_dim[1]), device=points.device)
    map[:] = float("nan")
    # points = points.long()
    # flip the x-y axes, so that x shows up in the image
    points[:, 0] = map_dim[0] - 1 - points[:, 0]
    points[:, 1] = map_dim[1] - 1 - points[:, 1]
    # find the point with the highest z value at an x-y voxel without a for loop
    indices = torch.argsort(
        points[:, 2] * map_resolution_xyz[2]
        + map_dim[0] * points[:, 0].long()
        + map_dim[0] * map_dim[1] * points[:, 1].long(),
        descending=True,
    )
    points = points[indices]
    points_uv = points[:, 0].long() + map_dim[0] * points[:, 1].long()
    selector = (points_uv[1:] - points_uv[:-1]) != 0
    # scale the pixel values to a metric range. Invalid pixels have a value of -1
    map.view(-1)[points_uv[1:][selector]] = (
        points[:, 2][1:][selector] * map_resolution_xyz[2]
    )
    map[1, 1] = map[2, 2]
    return map


def l1_loss_pixel_space(
    output: Tensor,
    target: Tensor,
    map_dim: Tensor = torch.tensor([64, 64, 64]),
    map_resolution_xyz: TensorOrArrOrList = [0.05, 0.05, 0.03],
) -> Tensor:
    "Computes the l1 loss of the heightmaps of the pointclouds"

    output_map = point_cloud_to_heightmap(
        output, map_dim=map_dim, map_resolution_xyz=map_resolution_xyz
    )
    target_map = point_cloud_to_heightmap(
        target, map_dim=map_dim, map_resolution_xyz=map_resolution_xyz
    )

    # take only valid pixels
    # mask = torch.logical_and(output_map > -0.99, target_map > -0.99)
    mask = ~torch.logical_or(torch.isnan(output_map), torch.isnan(target_map))
    output_map = output_map[mask]
    target_map = target_map[mask]

    # un-normalize the height value from [0,1] to meters
    output_map = output_map  # *map_dim[2] #*map_resolution_xyz[2]
    target_map = target_map  # *map_dim[2] #*map_resolution_xyz[2]

    loss = torch.mean(torch.abs(output_map - target_map))

    return loss


def compute_point_cloud_statistics(
    point_cloud: Tensor, true_point_cloud: Tensor
) -> dict:
    """Computes the precision, recall and f1 score for two point clouds."""
    # voxelize the target point cloud. the reconstruction is already voxelized.
    target_coords = torch.cat(
        (
            true_point_cloud.floor(),
            torch.zeros(
                (len(true_point_cloud), 1),
                dtype=torch.int,
                device=true_point_cloud.device,
            ),
        ),
        dim=1,
    )
    target_feats = true_point_cloud % 1
    batched_target_coords, batched_target_feats = ME.utils.sparse_collate(
        coords=[target_coords], feats=[target_feats], device=true_point_cloud.device
    )
    batched_target = ME.SparseTensor(
        features=batched_target_feats,
        coordinates=batched_target_coords,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        device=true_point_cloud.device,
    )
    true_point_cloud = batched_target.coordinates_at(0)[
        :, :3
    ].float() + batched_target.features_at(0)

    indices = point_cloud.floor()
    indices_true = true_point_cloud.floor()
    tp = torch.sum((indices_true == indices.unsqueeze(1)).all(dim=-1).any(dim=1))
    fp = len(indices) - tp
    fn = len(indices_true) - tp
    precision = tp / (tp + fp)
    recall = tp / (fn + tp)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision.item(), "recall": recall.item(), "f1": f1.item()}


def rand_perlin_masks(
    num_masks: int,
    shape: t.Tuple[int, int],
    res: t.Tuple[int, int],
    threshold: float,
    fade: t.Callable = lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3,
    device: DeviceType = torch.device("cuda"),
):
    # Adapted from https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0], device=device),
                torch.arange(0, res[1], delta[1], device=device),
            ),
            dim=-1,
        )
        % 1
    )
    angles = 2 * math.pi * torch.rand(num_masks, res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile_grads(slice1, slice2):
        return (
            gradients[:, slice1[0] : slice1[1], slice2[0] : slice2[1]]
            .repeat_interleave(d[0], 1)
            .repeat_interleave(d[1], 2)
        )

    def dot(grad, shift):
        return (
            torch.stack(
                (
                    grid[: shape[0], : shape[1], 0] + shift[0],
                    grid[: shape[0], : shape[1], 1] + shift[1],
                ),
                dim=-1,
            )
            * grad[:, : shape[0], : shape[1]]
        ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    perlin = math.sqrt(2) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]
    )

    perlin += (
        torch.abs(torch.amin(perlin, (1, 2)))
        .view(-1, 1, 1)
        .repeat(1, shape[0], shape[1])
    )
    perlin.div_(
        torch.abs(torch.amax(perlin, (1, 2)))
        .view(-1, 1, 1)
        .repeat(1, shape[0], shape[1])
    )
    perlin[perlin < threshold] = 0
    perlin[perlin > threshold] = 1
    return perlin.int()


def convert_quat_and_t_to_pose(qt: TensorOrArr) -> TensorOrArr:
    """
    Convert a 7D quaternion+translation into a 4x4 pose matrix.
    Args:
        qt: A 4-element quaternion [w, x, y, z] and a 3-element translation [x, y, z]
    Returns:
        A 4x4 homogeneous transformation matrix
    """

    is_tensor = isinstance(qt, torch.Tensor)
    if is_tensor:
        device = qt.device
        qt = qt.cpu().numpy()

    q = qt[:4]
    t = qt[4:]
    x, y, z, w = q
    rotation_matrix = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = t

    if is_tensor:
        pose = torch.from_numpy(pose).float().to(device)
    return pose


def convert_pose_to_quat_and_t(pose: TensorOrArr) -> TensorOrArr:
    """Analogoous to convert_quat_and_t_to_pose, but in the opposite direction."""
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose)
    rotation_matrix = pose.cpu()[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    quaternion = torch.tensor(rotation.as_quat())
    # Extract translation as a 3D vector (x, y, z)
    translation_vector = pose.cpu()[:3, 3]
    pose = torch.vstack([quaternion.unsqueeze(1), translation_vector.unsqueeze(1)]).T[0]
    return pose


def fuse_centered_pts_in_world(
    measured: TensorOrArr, poses: TensorOrArr, axes_means: TensorOrArr
) -> TensorOrArr:
    """Fuses the centered point clouds in the world frame."""
    new_measured = []
    for i, (m, m_means) in enumerate(zip(measured, axes_means)):
        new_measured.append(m + m_means)
    new_measured = fuse_point_clouds_in_world(new_measured, poses)
    return new_measured


def fuse_centered_pts(measured: TensorOrArr, axes_means: TensorOrArr) -> TensorOrArr:
    """Fuses the centered point clouds."""
    new_measured = []
    for i, (m, m_means) in enumerate(zip(measured, axes_means)):
        new_measured.append(m + m_means)
    lib = pick_library(measured)
    new_measured = lib.concatenate(new_measured, axis=0)
    return new_measured


def radius_outlier_removal(
    point_cloud: np.ndarray, radius: float = 0.8, min_neighbors_in_radius: int = 2
) -> np.ndarray:
    """
    Filters outliers using the Radius Outlier Removal method.

    Args:
        point_cloud: Input point cloud.
        radius: The radius of the sphere to consider for neighborhood.
        min_neighbors_in_radius: Minimum number of neighbors within the sphere radius.
    Returns:
        Filtered point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    filtered_cloud, _ = pcd.remove_radius_outlier(
        nb_points=min_neighbors_in_radius, radius=radius
    )
    return np.asarray(filtered_cloud.points)


def radius_outlier_removal_pclpy(
    point_cloud: np.ndarray, radius: float, min_neighbors_in_radius: int
) -> np.ndarray:
    """
    Filters outliers using the Radius Outlier Removal method from PCL.
    """
    import pclpy.pcl as pcl

    if isinstance(point_cloud, np.ndarray):
        point_cloud = pcl.PointCloud.PointXYZ(point_cloud)
    outrem = pcl.filters.RadiusOutlierRemoval.PointXYZ()
    outrem.setInputCloud(point_cloud)
    outrem.setRadiusSearch(radius)
    outrem.setMinNeighborsInRadius(min_neighbors_in_radius)
    filtered_cloud = pcl.PointCloud.PointXYZ()
    outrem.filter(filtered_cloud)
    return filtered_cloud.xyz


def downsample_point_cloud(xyz: TensorOrArr, voxel_size: float) -> TensorOrArr:
    """Downsample point cloud by averaging points within every voxel of size voxel_size."""
    pcd = o3d.geometry.PointCloud()
    is_tensor = isinstance(xyz, torch.Tensor)
    if is_tensor:
        device = xyz.device
        xyz = xyz.detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    new_pts = np.asarray(downpcd.points)
    if is_tensor:
        new_pts = torch.from_numpy(new_pts).to(device)
    return new_pts


def match_point_clouds(x: Tensor, y: Tensor) -> t.Tuple[Tensor, Tensor]:
    """Wraps the _match_point_clouds function to handle batched inputs."""
    has_batch_dim = len(x.shape) == 3
    if has_batch_dim:
        matched_x = []
        matched_y = []
        batch_size = x.shape[0]
        for i in range(batch_size):
            mx, my = _match_point_clouds(x[i], y[i])
            matched_x.append(mx)
            matched_y.append(my)
        matched_x = torch.stack(matched_x, dim=0)
        matched_y = torch.stack(matched_y, dim=0)
    else:
        matched_x, matched_y = _match_point_clouds(x, y)
    return matched_x, matched_y


def _match_point_clouds(x: Tensor, y: Tensor) -> t.Tuple[Tensor, Tensor]:
    """Matches the points in x to the nearest neighbors in y."""
    dist = torch.cdist(x, y, p=2)
    _, nns = torch.topk(-dist, k=1, dim=1)
    x_idxs = list(range(x.shape[0]))
    y_idxs = nns.squeeze().tolist()
    matched_x = x[x_idxs]
    matched_y = y[y_idxs]
    if len(matched_x.shape) == 1:
        matched_x = matched_x.unsqueeze(0)
    if len(matched_y.shape) == 1:
        matched_y = matched_y.unsqueeze(0)
    return matched_x, matched_y


def gen_filled_sphere(radius: float = 10.0, num_points: int = 1000):
    """Generates a filled sphere point cloud."""

    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.arccos(1 - 2 * np.random.uniform(0, 1, num_points))
    r = radius * np.cbrt(np.random.uniform(0, 1, num_points))

    # Spherical to Cartesian conversion
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z]).T


def angle_btw_vectors(
    vector1: np.ndarray, vector2: np.ndarray, signed: bool = True
) -> np.ndarray:
    """Computes the angle between two vectors. Inputs may be 2D."""
    vector1 = np.atleast_2d(vector1)
    vector2 = np.atleast_2d(vector2)
    cos_angle = np.einsum("ij,kj->i", vector1, vector2) / (
        np.linalg.norm(vector1, axis=1) * np.linalg.norm(vector2)
    )
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    if signed:
        is_2d = vector1.shape[1] == 2
        if is_2d:
            det = vector1[:, 0] * vector2[:, 1] - vector1[:, 1] * vector2[:, 0]
            angle = np.where(det >= 0, angle, -angle)
        else:
            cross = np.cross(vector1, vector2)
            angle = np.where(cross[:, 2] < 0, -angle, angle)
    return angle


def filter_pts_within_azimuth_angle_from_center(
    pts: TensorOrArr,
    center_pos: np.ndarray,
    view_vector: np.ndarray,
    angle_rad: float,
    inner_angle_rad: t.Optional[float] = None,
) -> np.ndarray:
    """Filters points that are within the sector defined by the azimuth angle in the view direction from the center position."""

    mask = get_pts_mask_within_azimuth_angle(
        pts, center_pos, view_vector, angle_rad, inner_angle_rad
    )
    selected_points_3d_slice = pts[mask]
    return selected_points_3d_slice


def get_pts_mask_within_azimuth_angle(
    pts, center_pos, view_vector, angle_rad, inner_angle_rad=None
):
    pts_centered = pts - center_pos

    # only x and y for determining the angle
    pts_centered = pts_centered[:, :2]
    view_vector = view_vector[:2]

    pts_centered_normalized = (
        pts_centered / np.linalg.norm(pts_centered, axis=1)[:, np.newaxis]
    )

    angle_rad_half = angle_rad / 2
    angles = angle_btw_vectors(pts_centered_normalized, view_vector)
    mask = np.abs(angles) <= angle_rad_half

    if len(angles[mask]) == 0:
        print("No points within the outer slice")
        # print spec
        print(f"{angles=}")
        if len(angles) > 0:
            print(f"{np.min(angles)=}, {np.max(angles)=}")
        print(f"{angle_rad_half=}")
        print(f"{view_vector=}")
        return mask

    if inner_angle_rad is None:
        return mask

    min_angle = np.min(angles[mask])
    max_angle = np.max(angles[mask])
    # assert min_angle <= 0, f"{min_angle=} should be negative"
    inner_angle_rad_half = inner_angle_rad / 2
    inner_angle_center = np.random.uniform(
        min_angle + inner_angle_rad_half, max_angle - inner_angle_rad_half
    )

    inner_mask_1 = angles > inner_angle_center - inner_angle_rad_half
    inner_mask_2 = angles < inner_angle_center + inner_angle_rad_half
    inner_mask = inner_mask_1 & inner_mask_2

    return mask & ~inner_mask


def shift_centered_pts(pts: TensorOrArr, translation: TensorOrArr) -> TensorOrArr:
    return pts + translation


def slice_pcl_by_azimuth_angle(
    pts: TensorOrArr,
    angle: float,
    center_pos: np.ndarray,
    chunk_angles: t.Optional[np.ndarray] = None,
    inner_angle_rad: t.Optional[float] = None,
) -> t.List[np.ndarray]:
    """Slices the point cloud by azimuth angles.
    Args:
        pts: The input point cloud.
        angle: The azimuth angle.
        center_pos: The origin position.
        chunk_angles: The angles to slice the point cloud by.
        inner_angle_rad: The angle to use for filtering points within the outer slice.
    Returns:
        A list of point clouds sliced by the azimuth angle.
    """

    angle_rad = np.deg2rad(angle)
    num_chunks = 360 // angle
    chunk_angles = chunk_angles or np.linspace(angle // 2, 360 - angle // 2, num_chunks)

    merged_pts = []
    for chunk_angle in chunk_angles:
        view_vector = np.array(
            [
                np.cos(np.deg2rad(chunk_angle)),
                np.sin(np.deg2rad(chunk_angle)),
                0,
            ]
        )
        chunk_pts = filter_pts_within_azimuth_angle_from_center(
            pts, center_pos, view_vector, angle_rad, inner_angle_rad
        )
        if len(chunk_pts) == 0:
            continue
        merged_pts.append(chunk_pts)
    return merged_pts


def slice_pcl_traj_with_arm_occlusion(
    pts: np.ndarray,
    angle: float,
    center_pos: np.ndarray,
    scan_angles: t.Optional[np.ndarray] = None,
) -> t.List[np.ndarray]:
    """Slices the point cloud by azimuth angles and filters points that would be occluded (as determined by heuristics) by the robot arm."""

    from terrain_synthesis.utils.postprocessing_utils import (
        filter_pts_occluded_by_arm,
        generate_scan_starting_angles,
    )

    num_scans = 10
    total_coverage = 360  # Total coverage in degrees
    arm_length = 5

    scan_angles = (
        generate_scan_starting_angles(num_scans, total_coverage)
        if scan_angles is None
        else scan_angles
    )
    merged_pts = []
    for i, chunk_angle in enumerate(scan_angles):
        angle_rad = np.deg2rad(angle)
        view_vector = np.array(
            [
                np.cos(np.deg2rad(chunk_angle)),
                np.sin(np.deg2rad(chunk_angle)),
                0,
            ]
        )
        x2 = filter_pts_within_azimuth_angle_from_center(
            pts, center_pos, view_vector, angle_rad
        )
        arm_endeffector_pos = center_pos + view_vector * arm_length
        filter_arm_res = filter_pts_occluded_by_arm(
            x2,
            center_pos,
            arm_endeffector_pos=arm_endeffector_pos,
            arm_length=arm_length,
        )
        x2 = filter_arm_res["pts"]
        if len(x2) == 0:
            continue
        merged_pts.append(x2)
    return merged_pts


def random_sample_pts(
    pts: np.ndarray,
    n_random_spawns: int,
    plot_individual: bool = False,
    robot_height: float = 3.8,
    lidar_range: float = 10.0,
    lidar_blind_range: float = 5.0,
    terrain_wh: float = 50,
) -> dict:
    """Randomly samples points in the point cloud as if they were lidar scans.
    Args:
        pts: The input point cloud.
        n_random_spawns: The number of random spawns.
        plot_individual: Whether to plot the individual point clouds.
        robot_height: The height of the robot.
        lidar_range: The range of the lidar (largest distance from which points are visible).
        lidar_blind_range: The blind range of the lidar (points are not visible).
        terrain_wh: The width/height of the square terrain.
    Returns:
        A dictionary containing the point clouds, the downsampled point clouds, and the random translations.
    """

    random_translations = []
    for spawn_idx in range(n_random_spawns):
        random_translation = np.random.uniform(0, terrain_wh, size=2)
        random_translation = np.append(random_translation, robot_height)
        random_translations.append(random_translation)

    pointCloudDatas = sample_pts_in_lidar_range(
        pts,
        random_translations,
        lidar_range=lidar_range,
        lidar_blind_range=lidar_blind_range,
        plot_individual=plot_individual,
        output_as_list=True,
    )

    pointCloudDatas_downsampled = downsample_point_cloud(
        np.concatenate(pointCloudDatas, axis=0), voxel_size=0.01
    )
    return {
        "pointCloudDatas": pointCloudDatas,
        "pointCloudDatas_downsampled": pointCloudDatas_downsampled,
        "random_translations": random_translations,
    }


def sample_pts_in_lidar_range(
    pts: np.ndarray,
    random_translations: list,
    lidar_range: float = 10.0,
    lidar_blind_range: float = 5.0,
    plot_individual: bool = False,
    output_as_list: bool = False,
) -> t.Union[np.ndarray, list]:
    """Samples points in the point cloud as if they were lidar scans.
    Args:
        pts: The input point cloud.
        random_translations: The random translations.
        lidar_range: The range of the lidar (largest distance from which points are visible).
        lidar_blind_range: The blind range of the lidar (points are not visible).
        plot_individual: Whether to plot the individual point clouds.
        output_as_list: Whether to output the point clouds as a list.
    Returns:
        The point cloud data.
    """
    pointCloudDatas = []
    for lidar_center in random_translations:
        random_pts = fetch_pts_in_lidar_range(
            pts,
            lidar_center,
            lidar_range=lidar_range,
            lidar_blind_range=lidar_blind_range,
        )
        pointCloudDatas.append(random_pts)
        if plot_individual:
            plot_point_cloud(random_pts)
    if output_as_list:
        return pointCloudDatas
    return np.concatenate(pointCloudDatas, axis=0)


def xyz_to_cylindrical(xyz: TensorOrArrOrList) -> TensorOrArrOrList:
    """Wraps the _xyz_to_cylindrical function to handle batched inputs."""
    if isinstance(xyz, list):
        return [_xyz_to_cylindrical(x) for x in xyz]
    return _xyz_to_cylindrical(xyz)


def _xyz_to_cylindrical(xyz: TensorOrArr) -> TensorOrArr:
    """Converts a point cloud from Cartesian to cylindrical coordinates."""
    lib = pick_library(xyz)
    shape = xyz.shape
    assert shape[-1] == 3, "Input must have size (N, 3) or (B, N, 3)"
    if len(shape) == 3:
        xyz = xyz.reshape(-1, 3)
    rho = lib.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    phi = lib.arctan2(xyz[:, 1], xyz[:, 0])
    z = xyz[:, 2]
    res = lib.column_stack((rho, phi, z))
    if len(shape) == 3:
        res = res.reshape(shape)
    return res


def cylindrical_to_xyz(cylindrical: TensorOrArrOrList) -> TensorOrArrOrList:
    """Wraps the _cylindrical_to_xyz function to handle batched inputs."""
    if isinstance(cylindrical, list):
        return [_cylindrical_to_xyz(c) for c in cylindrical]
    return _cylindrical_to_xyz(cylindrical)


def _cylindrical_to_xyz(cylindrical: TensorOrArr) -> TensorOrArr:
    """Converts a point cloud from cylindrical to Cartesian coordinates."""
    lib = pick_library(cylindrical)
    shape = cylindrical.shape
    assert shape[-1] == 3, "Input must have size (N, 3) or (B, N, 3)"
    if len(shape) == 3:
        cylindrical = cylindrical.reshape(-1, 3)
    x = cylindrical[:, 0] * lib.cos(cylindrical[:, 1])
    y = cylindrical[:, 0] * lib.sin(cylindrical[:, 1])
    z = cylindrical[:, 2]
    res = lib.column_stack((x, y, z))
    if len(shape) == 3:
        res = res.reshape(shape)
    return res


def read_pcl(input_file: str) -> np.ndarray:
    """Reads a point cloud from a file."""
    if input_file.endswith(".npy"):
        return np.load(input_file)
    return np.asarray(o3d.io.read_point_cloud(input_file).points)


def get_dist_stats_of_pts(pts: TensorOrArr) -> dict:
    """Wraps _get_dist_stats_of_pts and includes axis-wise statistics."""

    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()

    # axis-wise
    axes_stats = {}
    dim_to_ax = {0: "x", 1: "y", 2: "z"}
    for dim in range(3):
        dist_stats_axis = _get_dist_stats_of_pts(pts[:, dim])
        axes_stats[dim_to_ax[dim]] = {
            "max_dist": dist_stats_axis["max_dist"],
            "min_dist": dist_stats_axis["min_dist"],
            "mean_dist": dist_stats_axis["mean_dist"],
        }
    # over all axes
    dist_stats = _get_dist_stats_of_pts(pts)
    most_distant_pts = find_most_distant_points(pts, dist_stats["pairwise_dists"])
    return {
        "max_dist": dist_stats["max_dist"],
        "min_dist": dist_stats["min_dist"],
        "mean_dist": dist_stats["mean_dist"],
        "axes": axes_stats,
        "most_distant_pts": most_distant_pts,
    }


def _get_dist_stats_of_pts(pts: TensorOrArr) -> dict:
    """Computes the maximum, minimum, and mean distances between points in a point cloud."""
    if len(pts.shape) == 1:
        pts = pts[:, None]
    dists = scipy.spatial.distance.cdist(pts, pts, "euclidean")
    pairwise_dists = np.triu(dists, k=1)
    dists = pairwise_dists[pairwise_dists > 0]
    max_dist = np.max(dists)
    min_dist = np.min(dists[dists > 0])
    mean_dist = np.mean(dists[dists > 0])
    return {
        "max_dist": np.round(max_dist, 3),
        "min_dist": np.round(min_dist, 3),
        "mean_dist": np.round(mean_dist, 3),
        "pairwise_dists": pairwise_dists,
    }


def find_most_distant_points(
    pts: TensorOrArr, pairwise_distances: np.ndarray = None
) -> list:

    if pairwise_distances is None:
        # Compute the pairwise distances
        pairwise_distances = scipy.spatial.distance.cdist(pts, pts, "euclidean")

    # Find the indices of the maximum distance
    i, j = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)

    # Retrieve the coordinates of the two points
    return [pts[i], pts[j]]


def load_point_cloud_o3d(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    return pcd


def align_pcls(pcl, target_pcl, dist_threshold):
    # dist_threshold for ICP
    pcd_tilted = load_point_cloud_o3d(pcl)
    pcd_flat = load_point_cloud_o3d(target_pcl)
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_tilted,
        pcd_flat,
        dist_threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )
    pcd_tilted.transform(reg_p2p.transformation)
    point_cloud_aligned = np.asarray(pcd_tilted.points)
    return point_cloud_aligned


def get_min_max_pts_from_center(pts, center, use_2d=False):
    if use_2d:
        pts = pts[:, :2]
        center = center[:2]
    distances = np.linalg.norm(pts - center, axis=1)
    pt_min = pts[np.argmin(distances)]
    pt_max = pts[np.argmax(distances)]
    return {
        "pt_min": pt_min,
        "pt_max": pt_max,
        "dist_min": np.min(distances),
        "dist_max": np.max(distances),
    }


def flip_pts_along_y_axis(pts, center=None):
    aux_rotation = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
    )
    means = np.mean(pts, axis=0) if center is None else center
    pts = np.dot(pts - means, aux_rotation)
    pts += means
    return pts


def denormalize_pcl(pcl, mean, max_dist):
    return pcl * max_dist + mean


def normalize_pcl(pc, mean=None, max_dist=None):
    """pc: NxC, return NxC"""
    centroid = torch.mean(pc, 0) if mean is None else mean
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, 1))) if max_dist is None else max_dist
    pc = pc / m
    return pc


def downsample_pcl(pcl, num_pts, idx=None, use_randperm=True):
    assert len(pcl.shape) == 2, f"{pcl.shape=}"
    if pcl.shape[0] > num_pts:
        if idx is None:
            if use_randperm:
                idx = torch.randperm(pcl.shape[0])[:num_pts]
            else:
                idx = torch.arange(num_pts)
        pcl = pcl[idx, :]
    return {
        "pcl": pcl,
        "idx": idx,
    }
