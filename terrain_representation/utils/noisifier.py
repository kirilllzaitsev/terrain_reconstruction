import typing as t

import numpy as np
import torch
from terrain_representation.utils.pointcloud_utils import (
    filter_pts_within_azimuth_angle_from_center,
    get_pts_mask_within_azimuth_angle,
    rand_perlin_masks,
    random_box,
)
from terrain_representation.utils.utils import DeviceType, TensorOrArrOrList
from torch import Tensor


class Noisifier:
    """A data noisifier
    Adds random perturbations to the point clouds and poses.
    Args:
        map_dim: The dimensions of the grid
        map_resolution: The resolution of the grid cell in meters
        device: The device to use
        num_blob_points: The number of points in each blob
        noise_scale: (deprecated) The strength of the noise to add to XYZ coordinates
        num_blobs: The number of blobs to add to the point cloud
        blob_extent: The range of the blobs to add to the point cloud in meters
        subsampling_ratio_min/max: The lower/upper bound on random share of points to keep in the point cloud
    """

    def __init__(
        self,
        map_dim: Tensor,
        map_resolution: Tensor,
        device: DeviceType = torch.device("cuda"),
        num_blob_points: t.Optional[float] = None,
        noise_scale: t.Optional[float] = None,
        num_blobs: t.Optional[int] = None,
        blob_extent: t.Optional[float] = None,
        subsampling_ratio_min: float = 1.0,
        subsampling_ratio_max: float = 1.0,
        noise_prob: float = 1.0,
        add_blobs_prob: float = 1.0,
        add_boxes_prob: float = 1.0,
    ):
        self.device = device
        self.map_dim = map_dim.to(device)
        self.map_resolution = map_resolution.to(device)

        if isinstance(noise_scale, float):
            self.noise_scale = torch.tensor(
                [noise_scale, noise_scale, noise_scale], device=device
            )
        elif noise_scale is not None:
            if not isinstance(noise_scale, torch.Tensor):
                noise_scale = torch.tensor(noise_scale, device=device)

        self.noise_prob = noise_prob
        self.num_blob_points = num_blob_points
        self.num_blobs = num_blobs
        self.blob_range_m = blob_extent
        self.subsampling_ratio_min = subsampling_ratio_min
        self.subsampling_ratio_max = subsampling_ratio_max
        self.add_blobs_prob = add_blobs_prob
        self.add_boxes_prob = add_boxes_prob

        # # generate random distractors
        self.num_distractors = 1000
        self.distractors = []
        for i in range(self.num_distractors):
            self.distractors.append(random_box(self.device))

        self.height_offsets = dict()
        self.flips = dict()

    def __call__(
        self,
        measured_point_cloud: Tensor,
        gt: Tensor,
        add_noise: bool = True,
        pose=None,
        labels=None,
        noise_label=-1,
    ) -> "dict[str, Tensor]":
        """Noisifies the measured and gt point clouds"""

        added_noise = False
        if np.random.rand() > self.noise_prob:
            return {
                "measured": measured_point_cloud,
                "gt": gt,
                "added_noise": added_noise,
            }

        # subsample
        if self.subsampling_ratio_min < 1.0:
            num_points = measured_point_cloud.shape[0]
            subsampling_ratio = (
                torch.FloatTensor(1)
                .uniform_(self.subsampling_ratio_min, self.subsampling_ratio_max)
                .item()
            )

            num_points_to_keep = int(num_points * subsampling_ratio)
            indices = torch.randperm(num_points, device=self.device)[
                :num_points_to_keep
            ]
            measured_point_cloud = measured_point_cloud[indices]
            if labels is not None:
                labels = labels[indices]

        has_batch_dim = measured_point_cloud.dim() == 3
        N_dim = -2

        if add_noise:
            min_xyz = torch.min(measured_point_cloud, dim=N_dim)[0]
            max_xyz = torch.max(measured_point_cloud, dim=N_dim)[0]

            # random blobs
            if np.random.rand() < self.add_blobs_prob:
                blob_centers = (max_xyz - min_xyz) * torch.rand(
                    (self.num_blobs, 1, 3), device=self.device
                ) + min_xyz
                blob_elems = torch.zeros(
                    (self.num_blobs, self.num_blob_points, 3), device=self.device
                )
                for i in range(self.num_blobs):
                    blob_extent = max(
                        self.blob_range_m * 0.1,
                        self.blob_range_m * torch.rand(1).item(),
                    )
                    blob_elems[i].normal_(0, blob_extent)

                blob_noise = (blob_centers + blob_elems).flatten(0, 1)
                if has_batch_dim:
                    blob_noise = blob_noise.unsqueeze(0)
                measured_point_cloud = torch.cat(
                    (measured_point_cloud, blob_noise), dim=N_dim
                )
                if labels is not None:
                    labels = torch.cat(
                        (
                            labels,
                            torch.ones(
                                (blob_noise.shape[0], ), device=self.device
                            ).long() * noise_label,
                        ),
                        dim=0,
                    )
                added_noise = True

            # random boxes
            if np.random.rand() < self.add_boxes_prob:
                for i in range(torch.randint(3, 6, (1,)).item()):
                    box_idx = torch.randint(self.num_distractors, (1,)).item()
                    box = self.distractors[box_idx].clone()
                    box_position = (max_xyz - min_xyz) * torch.rand(
                        (3,), device=self.device
                    ) + min_xyz
                    # gen random box_position
                    # box_position = torch.zeros((3,), device=self.device).uniform_(
                    #     min_xyz[0], max_xyz[0]
                    # )
                    box += box_position
                    if has_batch_dim:
                        box = box.unsqueeze(0)
                    measured_point_cloud = torch.cat(
                        (measured_point_cloud, box), dim=N_dim
                    )
                    if labels is not None:
                        labels = torch.cat(
                            (
                                labels,
                                torch.ones(
                                    (box.shape[0], ), device=self.device
                                ).long()
                                * noise_label,
                            ),
                            dim=0,
                        )
                added_noise = True

        # from pytorch3d import transforms
        # rpy = torch.zeros((3,), device=self.device)
        # rpy[0] = torch.zeros((1,), device=self.device).uniform_(-0.1, 0.1)
        # rpy[1] = torch.zeros((1,), device=self.device).uniform_(-0.1, 0.1)
        # rpy[2] = torch.zeros((1,), device=self.device).uniform_(-0.1, 0.1)
        # delta_rot = transforms.euler_angles_to_matrix(rpy, convention='XYZ')
        # measured_point_cloud = torch.mm(delta_rot, measured_point_cloud.transpose(1, 0)).transpose(1, 0)

        # measured_point_cloud += torch.zeros((3,), device=self.device).uniform_(-0.04, 0.04)

        # pose perturbation
        # delta_pos = torch.zeros((3,), device=self.device).uniform_(-0.04, 0.04)
        # pose[:3] += delta_pos

        #
        # rpy = torch.zeros((3,), device=self.device)
        # rpy[0] = torch.zeros((1,), device=self.device).uniform_(-0.1, 0.1)
        # rpy[1] = torch.zeros((1,), device=self.device).uniform_(-0.1, 0.1)
        # rpy[2] = torch.zeros((1,), device=self.device).uniform_(-0.1, 0.1)
        # delta_rot = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(rpy, convention='XYZ'))
        # rot_start = pose[3:].clone()
        # rot_start[0] = pose[-1]
        # rot_start[1:] = pose[3:6]
        # noised_rot = transforms.quaternion_multiply(rot_start, delta_rot)
        # pose[-1] = noised_rot[0]
        # pose[3:6] = noised_rot[1:]

        # measured_point_cloud = torch.cat((measured_point_cloud, initial_measured_point_cloud))

        # center the point cloud in z direction along the whole trajectory
        # if time_idx == 0:
        #     self.height_offsets[batch_idx] = center[2] + \
        #         (2*torch.rand((1,), device=self.device)-1)*self.map_resolution[2]

        # measured_point_cloud[:, 2] -= self.height_offsets[batch_idx]
        # gt[:, 2] -= self.height_offsets[batch_idx]

        # if self.flips[batch_idx] == 0:
        #     pass
        # elif self.flips[batch_idx] == 1:
        #     measured_point_cloud[:, 0] *= -1
        #     gt[:, 0] *= -1
        #     pose[0] *= -1
        #     pose[-1] *= -1
        # elif self.flips[batch_idx] == 2:
        #     measured_point_cloud[:, 1] *= -1
        #     gt[:, 1] *= -1
        #     pose[1] *= -1
        #     pose[-1] *= -1
        # else:
        #     measured_point_cloud[:, 0:2] *= -1
        #     gt[:, 0:2] *= -1
        #     pose[0:2] *= -1

        return {
            "measured": measured_point_cloud,
            "gt": gt,
            "added_noise": added_noise,
            "labels": labels,
        }
