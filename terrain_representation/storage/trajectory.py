import copy
import os
import pickle
import typing as t
from typing import List, Union

from torch import Tensor


class PointCloudTrajectory:
    """A point cloud trajectory
    Stores:
    - (centered) Measured point cloud scan
    - (centered) Ground truth point cloud scan
    - (centered) mesh point cloud of the terrain
    - Pose of the robot in the world frame W
    - The mean of the axes of the measured point cloud before centering
    - The mean of the axes of the ground truth point cloud before centering
    - The mean of the axes of the mesh point cloud before centering
    - The pose of the robot in the body frame B
    - The joint positions of the robot
    - The name of the terrain
    The measurement frame M is the coordinate system in which inference takes place.
    """

    def __init__(self):
        self.measured: List[Tensor] = list()
        self.gt: List[Tensor] = list()
        self.labels: List[Tensor] = list()

        self.pose: List[Tensor] = list()
        # Used for visualization
        self.pose_WB: List[Tensor] = list()
        self.joint_positions: List[Tensor] = list()

        self._measured_axes_means: List[Tensor] = list()
        self._gt_axes_means: List[Tensor] = list()
        self.mesh: List[Tensor] = list()
        self.mesh_axes_means: List[Tensor] = list()

        self.is_synthetic = True

        self.terrain_name = None

    def append(
        self,
        measured: Tensor,
        gt: Tensor,
        pose: Tensor,
        measured_axes_means: Tensor = None,
        gt_axes_means: Tensor = None,
        pose_WB: t.Optional[Tensor] = None,
        joint_positions: t.Optional[Tensor] = None,
        terrain_name: t.Optional[str] = None,
        mesh: t.Optional[Tensor] = None,
        mesh_axes_means: t.Optional[Tensor] = None,
        is_synthetic: bool = True,
        labels: t.Optional[Tensor] = None,
    ):
        """Append new sample to the trajectory"""
        self.measured.append(measured)
        self.gt.append(gt)
        # self.measured_2.append(measured_2)
        # self.gt_2.append(gt_2)
        self.pose.append(pose)
        if measured_axes_means is not None:
            self._measured_axes_means.append(measured_axes_means)
        if gt_axes_means is not None:
            self._gt_axes_means.append(gt_axes_means)
        if pose_WB is not None:
            self.pose_WB.append(pose_WB)
        if joint_positions is not None:
            self.joint_positions.append(joint_positions)
        if mesh is not None:
            self.mesh.append(mesh)
        if mesh_axes_means is not None:
            self.mesh_axes_means.append(mesh_axes_means)
        if labels is not None:
            self.labels.append(labels)
        self.terrain_name = terrain_name
        self.is_synthetic = is_synthetic

    def __getitem__(self, idx: int):
        """Returns a copy of the data at a specific time index"""
        if getattr(self, "labels", None) is not None and len(self.labels) > 0:
            labels = self.labels[idx]
        else:
            labels = None
        result = {
            "measured": (self.measured[idx]).clone(),
            "gt": (self.gt[idx]).clone(),
            "pose": (self.pose[idx]).clone(),
            "is_synthetic": getattr(self, "is_synthetic", True),
            "labels": labels,
        }
        for attr in ["mesh", "standalone"]:
            if not hasattr(self, attr):
                continue
            result[attr] = getattr(self, attr)[idx]
            if isinstance(result[attr], Tensor):
                result[attr] = result[attr].clone()
        if hasattr(self, "terrain_name"):
            result["terrain_name"] = self.terrain_name
        return result

    def create_slice(
        self, start_idx: int = 0, end_idx: t.Optional[int] = None
    ) -> "PointCloudTrajectory":
        if end_idx is None:
            end_idx = len(self.measured)

        result = self.create_subsample(list(range(start_idx, end_idx)))

        return result

    def create_subsample(self, idxs: t.List[int]) -> "PointCloudTrajectory":
        """Picks samples at specific indices and returns a new trajectory with those samples."""

        result = PointCloudTrajectory()
        result.measured = [self.measured[i] for i in idxs]
        result.gt = [self.gt[i] for i in idxs]
        result.pose = [self.pose[i] for i in idxs]
        for attr in [
            "mesh",
        ]:
            if hasattr(self, attr):
                setattr(result, attr, [getattr(self, attr)[i] for i in idxs])
        if hasattr(self, "terrain_name"):
            result.terrain_name = self.terrain_name
        result.is_synthetic = getattr(self, "is_synthetic", True)
        if getattr(self, "labels", None) is not None:
            result.labels = [self.labels[i] for i in idxs]

        return result

    def __len__(self):
        return len(self.measured)


class PointCloudTrajectoryStorage:
    """A point cloud trajectory storage
    Stores point cloud trajectories, which can be dumped and loaded from disk.

    """

    def __init__(self):
        self.point_cloud_trajectories: List[PointCloudTrajectory] = list()

    def append(
        self,
        point_cloud_trajectory: Union[PointCloudTrajectory, List[PointCloudTrajectory]],
    ):
        """Append one or multiple PointCloudTrajectory objects to the storage"""
        if isinstance(point_cloud_trajectory, list):
            self.point_cloud_trajectories.extend(point_cloud_trajectory)
        else:
            self.point_cloud_trajectories.append(point_cloud_trajectory)

    def dump(self, path: Union[str, bytes, os.PathLike]):
        """Dump point cloud trajectories to disk"""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)

    def merge(self, other: "PointCloudTrajectoryStorage"):
        """Merge another PointCloudTrajectoryStorage object into this one"""
        self.point_cloud_trajectories.extend(other.point_cloud_trajectories)

    @classmethod
    def load(
        cls, path: Union[str, bytes, os.PathLike]
    ) -> "PointCloudTrajectoryStorage":
        """Load point cloud trajectories from disk"""
        with open(path, "rb") as f:
            return pickle.load(f)
