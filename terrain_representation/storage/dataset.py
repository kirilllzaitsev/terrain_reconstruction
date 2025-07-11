# legacy code for compatibility purposes

import typing as t

import torch
from terrain_representation.storage.dataset_torch import (
    PointCloudDataset as PointCloudDatasetTorch,
)
from terrain_representation.storage.trajectory import (
    PointCloudTrajectory as PointCloudTrajectory_,
)
from terrain_representation.storage.trajectory import (
    PointCloudTrajectoryStorage as PointCloudTrajectoryStorage_,
)


class PointCloudTrajectory(PointCloudTrajectory_):
    pass


class PointCloudTrajectoryStorage(PointCloudTrajectoryStorage_):
    pass


class PointCloudDataset(PointCloudDatasetTorch):
    """A point cloud dataset created for compatibility purposes. See `dataset_torch` module for more details."""

    def __init__(
        self,
        base_folder: str,
        map_dim,
        map_resolution,
        device: t.Union[torch.device, str] = torch.device("cuda"),
        overfit_config: t.Optional[dict] = None,
        use_cylindrical_coords: bool = False,
        use_sparse: bool = False,
        **kwargs,
    ):
        super().__init__(
            base_folder=base_folder,
            map_dim=map_dim,
            map_resolution=map_resolution,
            device=device,
            overfit_config=overfit_config,
            use_cylindrical_coords=use_cylindrical_coords,
            use_sparse=use_sparse,
            **kwargs,
        )

    def __len__(self):
        return int(self.len)
