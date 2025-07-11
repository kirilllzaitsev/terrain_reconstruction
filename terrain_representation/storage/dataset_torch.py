import json
import os
import re
import traceback
import typing as t
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from terrain_representation.storage.isaac_scans_to_traj import (
    assemble_trajectory_from_path,
)
from terrain_representation.storage.trajectory import (
    PointCloudTrajectory as PointCloudTrajectory_,
)
from terrain_representation.storage.trajectory import (
    PointCloudTrajectoryStorage as PointCloudTrajectoryStorage_,
)
from terrain_representation.utils.pointcloud_utils import (
    convert_pose_to_quat_and_t,
    downsample_pcl,
    xyz_to_cylindrical,
)
from terrain_representation.utils.sensor_utils import fetch_pts_in_lidar_range
from terrain_representation.utils.voxel_grid_utils import convert_pts_to_vg
from terrain_synthesis.utils.utils import (
    convert_tr_traj_dir_to_metadata_key,
    get_ordered_paths,
)
from tqdm import tqdm


# for compatibility purposes
class PointCloudTrajectory(PointCloudTrajectory_):
    pass


class PointCloudTrajectoryStorage(PointCloudTrajectoryStorage_):
    pass


class PointCloudDataset(torch.utils.data.Dataset):
    """A dataset for synthetic point clouds."""

    def __init__(
        self,
        base_folder: str,
        map_dim: torch.Tensor,
        map_resolution: torch.Tensor,
        device: t.Union[torch.device, str] = torch.device("cuda"),
        overfit_config: t.Optional[dict] = None,
        use_cylindrical_coords: bool = False,
        use_sparse: bool = False,
        do_convert_pose_to_quat_and_t: bool = True,
        mesh_cell_resolution: t.Optional[float] = None,
        add_noise_to_real_data: bool = False,
        split="train",
        robot_params=None,
        traj_metadata_key=None,
        apply_rotation_prob=0.0,
        apply_mask_prob=0.0,
        apply_shift_prob=0.0,
        noisifier: t.Optional[t.Callable] = None,
        sequence_length: int = 1,
        step_skip: int = 1,
        seq_start: t.Optional[int] = None,
        num_pts: t.Optional[int] = None,
        use_labels: bool = False,
    ):
        self.apply_rotation_prob = apply_rotation_prob
        self.apply_mask_prob = apply_mask_prob
        self.apply_shift_prob = apply_shift_prob
        # do_convert_pose_to_quat_and_t is False for sparse dataset
        self.device = device
        self.use_labels = use_labels
        self.mesh_cell_resolution = mesh_cell_resolution
        self.use_sparse = use_sparse
        self.do_convert_pose_to_quat_and_t = do_convert_pose_to_quat_and_t
        self.use_cylindrical_coords = use_cylindrical_coords
        self.map_dim = map_dim.to(device)
        self.map_resolution = map_resolution.to(device)
        self.overfit_config = {} if overfit_config is None else overfit_config
        self.add_noise_to_real_data = add_noise_to_real_data
        self.noisifier = noisifier
        self.sequence_length = sequence_length
        self.step_skip = step_skip
        self.seq_start = seq_start
        self.num_pts = num_pts

        self.base_folder = base_folder
        self.traj_metadata_key = (
            traj_metadata_key
            or f"{Path(base_folder).parent.name}/{Path(base_folder).name}"
        )
        self.robot_params = (
            self.try_to_load_robot_params(base_folder)
            if robot_params is None
            else robot_params
        )

        # assemble metadata
        terrain_paths = get_ordered_paths(base_folder, ".*terrain_(\d+)")
        if self.robot_params is not None:
            if split == "val":
                terrain_paths = [
                    x
                    for x in terrain_paths
                    if Path(x).name in self.robot_params["terrain_names_val"]
                ]
        self.terrain_traj_paths = [
            get_ordered_paths(tp, ".*traj_(\d+).npy") for tp in terrain_paths
        ]
        self.traj_metadata = {}
        if self.robot_params is not None:
            if "traj_metadata" in self.robot_params:
                self.traj_metadata = self.robot_params["traj_metadata"][
                    self.traj_metadata_key
                ]
                self.terrain_traj_paths = self.filter_invalid_trajs(
                    self.terrain_traj_paths
                )

        # adjust metadata based on overfit_config
        if overfit_config is not None:
            self.terrain_traj_paths = self.adjust_trajs_to_overfit_config(
                self.terrain_traj_paths, overfit_config
            )
            self.scan_idxs_in_traj = overfit_config.get("scan_idxs_in_traj", [0])
            self.num_scans_in_traj = len(self.scan_idxs_in_traj)
        else:
            self.num_scans_in_traj = None
            self.scan_idxs_in_traj = None

            # filter out traj that are too short
            self.terrain_traj_paths = [
                [
                    traj_path
                    for traj_path in trajs
                    if self.traj_metadata[
                        convert_tr_traj_dir_to_metadata_key(traj_path)
                    ]["num_scans"]
                    > self.sequence_length * self.step_skip
                ]
                for trajs in self.terrain_traj_paths
            ]

        self.terrain_traj_paths_metadata_keys = [
            convert_tr_traj_dir_to_metadata_key(tr)
            for trs in self.terrain_traj_paths
            for tr in trs
        ]
        self.trajectory_lengths = [
            v["num_scans"]
            for k, v in self.traj_metadata.items()
            if k in self.terrain_traj_paths_metadata_keys
        ]
        self.terrain_traj_paths = [
            [
                traj_path
                for traj_path in trajs
                if self.traj_metadata[convert_tr_traj_dir_to_metadata_key(traj_path)][
                    "num_scans"
                ]
                > sequence_length * step_skip
            ]
            for trajs in self.terrain_traj_paths
        ]

        # flatten terrain_traj_paths
        self.traj_paths = [x for tr in self.terrain_traj_paths for x in tr]

    def __len__(self) -> int:
        # each trajectory also stores a sequence of point clouds
        return len(self.traj_paths)

    def __getitem__(self, idx: int) -> t.Dict[str, t.Any]:
        """Sample a trajectory from the dataset.
        In our case, a trajectory is a sequence of 120-degree point clouds.
        """
        traj_path = self.traj_paths[idx]
        apply_rotation = self.apply_rotation_prob > np.random.rand()
        apply_mask = self.apply_mask_prob > np.random.rand()
        apply_shift = self.apply_shift_prob > np.random.rand()
        terrain_name = traj_path_to_terrain_name(traj_path)
        if self.use_labels:
            # area map provides bounding boxes for obstacles. it was created when generating the dataset
            area_map = get_area_map(
                terrain_name,
                robot_params=(
                    self.robot_params["terrains"] if self.robot_params else None
                ),
            )
        else:
            area_map = None
        traj = assemble_trajectory_from_path(
            traj_path,
            num_scans=self.num_scans_in_traj,
            scan_idxs=self.scan_idxs_in_traj,
            min_gt_pts=300,  # minimum number of points for a pcl to be included
            apply_rotation=apply_rotation,
            apply_mask=apply_mask,
            apply_shift=apply_shift,
            area_map=area_map,
        )
        if len(traj) == 0:
            raise RuntimeError(f"Trajectory {self.traj_paths[idx]} is empty")
        parsed_traj = parse_traj(
            traj,
            self.sequence_length,
            self.step_skip,
            self.device,
            noisifier=self.noisifier,
            seq_start=self.seq_start,
            num_pts=self.num_pts,
        )
        parsed_traj.update(
            {
                "traj_path": "/".join(
                    traj_path.replace(".npy", "").rsplit("/", maxsplit=3)[-2:]
                )
            }
        )
        return parsed_traj

    def get_vg_from_pcl(self, pcl: torch.Tensor) -> torch.Tensor:
        return convert_pts_to_vg(pcl, self.map_dim, self.map_resolution)

    def filter_invalid_trajs(self, terrain_traj_paths):
        # some trajs may be too short due to insufficient points in the scan
        new_tr_traj_paths = []
        invalid_trajs = []
        for tr_trajs in terrain_traj_paths:
            new_tr_paths = []
            for tr_traj in tr_trajs:
                short_name = convert_tr_traj_dir_to_metadata_key(tr_traj)

                if short_name in self.traj_metadata:

                    traj_metadata = self.traj_metadata[short_name]
                    if traj_metadata["num_scans"] > 1:
                        new_tr_paths.append(tr_traj)
                    else:
                        invalid_trajs.append(
                            "/".join(tr_traj.rsplit("/", maxsplit=2)[-2:])
                        )
            new_tr_traj_paths.append(new_tr_paths)
        return new_tr_traj_paths

    def try_to_load_robot_params(self, base_folder):
        robot_params_path = Path(base_folder) / "robot_params.json"
        if not robot_params_path.exists():
            robot_params_path = Path(base_folder).parent / "robot_params.json"
            if not os.path.exists(robot_params_path):
                robot_params_path = (
                    Path(base_folder).parent.parent / "robot_params.json"
                )
                if not os.path.exists(robot_params_path):
                    print(f"{robot_params_path=} not found")
                    return None
            print(
                f"Using parent folder {robot_params_path.parent.name} for robot params"
            )

        if os.path.exists(robot_params_path):
            robot_params = json.load(open(robot_params_path, "r"))
        else:
            robot_params = None
            print(f"Robot params not found at {robot_params_path}")
        return robot_params

    def adjust_trajs_to_overfit_config(self, terrain_traj_paths, overfit_config):
        # helper for setting up ds for overfitting
        num_terrains = overfit_config.get("num_terrains", len(terrain_traj_paths))

        new_terrain_traj_paths = terrain_traj_paths[:num_terrains]

        traj_idxs = overfit_config.get("traj_idxs")
        if traj_idxs is not None and len(traj_idxs) > 0:
            new_terrain_traj_paths = [
                [traj_paths[i] for i in traj_idxs]
                for traj_paths in new_terrain_traj_paths
            ]
        else:
            num_traj_per_terrain = overfit_config.get("num_traj_per_terrain")
            assert (
                num_traj_per_terrain is not None
            ), f"num_traj_per_terrain or traj_idxs must be provided.\n{overfit_config=}"
            new_terrain_traj_paths = [
                tr[:num_traj_per_terrain] for tr in new_terrain_traj_paths
            ]

        new_terrain_traj_paths = [x for tr in new_terrain_traj_paths for x in tr]

        return [new_terrain_traj_paths]


class PointCloudDatasetPickle:
    """A legacy dataset used for loading data from .pkl files."""

    def __init__(
        self,
        base_folder: str,
        device,
        file_suffix: str = ".pkl",
        overfit_config=None,
        noisifier: t.Optional[t.Callable] = None,
        sequence_length: int = 1,
        step_skip: int = 1,
        seq_start: t.Optional[int] = None,
        num_pts: t.Optional[int] = None,
        ds_portion=1.0,
        ds_portion_from_start=None,
        ds_portion_from_end=None,
    ):
        self.base_folder = base_folder
        self.device = device
        self.file_suffix = file_suffix
        self.ds_portion = ds_portion
        self.ds_portion_from_start = ds_portion_from_start
        self.ds_portion_from_end = ds_portion_from_end
        self.overfit_config = {} if overfit_config is None else overfit_config
        self.trajectory_file_paths = [
            p
            for p in get_ordered_paths(self.base_folder, p=r".*terrains?_(\d+).*")
            if p.endswith(file_suffix)
        ]
        assert (
            len(self.trajectory_file_paths) > 0
        ), f"{self.trajectory_file_paths=} is empty"
        self.trajectories: t.List[PointCloudTrajectory] = list()
        self.len = 0
        self.do_convert_pose_to_quat_and_t = False
        self.add_noise_to_real_data = False
        self.use_cylindrical_coords = False
        self.noisifier = noisifier
        self.sequence_length = sequence_length
        self.step_skip = step_skip
        self.seq_start = seq_start
        self.num_pts = num_pts

    def __len__(self) -> int:
        # each trajectory also stores a sequence of point clouds
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> t.Dict[str, t.Any]:
        """Sample a trajectory from the dataset.
        In our case, a trajectory is a sequence of 120-degree point clouds.
        """
        traj = self.trajectories[idx]
        return parse_traj(
            traj,
            sequence_length=self.sequence_length,
            step_skip=self.step_skip,
            device=self.device,
            noisifier=self.noisifier,
            seq_start=self.seq_start,
            num_pts=self.num_pts,
        )

    def load_new_trajectories(self, portion: float = 1.0):
        """Loads a new portion of trajectories from the pickled files."""
        self.trajectories.clear()

        if self.overfit_config is not None:
            # load only one terrain
            num_files_to_load = 1
        else:
            num_files_to_load = int(len(self.trajectory_file_paths) * portion)
        print("Loading dataset")
        progress = tqdm(total=num_files_to_load)
        num_traj = self.overfit_config.get("num_traj_per_terrain")
        for file_path in self.trajectory_file_paths[:num_files_to_load]:
            try:
                point_cloud_storage = PointCloudTrajectoryStorage.load(file_path)
            except Exception as e:
                print(f"Error loading {file_path}")
                raise e
            new_trajs = []
            for traj in point_cloud_storage.point_cloud_trajectories:
                traj.measured = traj.measured_point_cloud_M
                traj.gt = traj.ground_truth_point_cloud_M
                traj.pose = traj.pose_WM
                if len(traj) > 2:
                    new_trajs.append(traj)
            self.trajectories += new_trajs
            torch.cuda.empty_cache()
            progress.update(1)
            if num_traj is not None and len(self.trajectories) >= num_traj:
                break
        progress.close()

        if self.overfit_config is not None and len(self.overfit_config) > 0:
            traj_idxs = self.overfit_config.get("traj_idxs") or []
            if len(traj_idxs) > 0:
                self.trajectories = [self.trajectories[i] for i in traj_idxs]
            else:
                assert (
                    num_traj is not None
                ), f"num_traj or traj_idxs must be provided.\n{self.overfit_config=}"
                self.trajectories = self.trajectories[:num_traj]

            scan_idxs = self.overfit_config.get("scan_idxs", [])
            if len(scan_idxs) > 0:
                for i in range(len(self.trajectories)):
                    self.trajectories[i] = self.trajectories[i].create_subsample(
                        scan_idxs
                    )
            else:
                num_scans_in_traj = self.overfit_config.get("num_scans_in_traj")
                assert (
                    num_scans_in_traj is not None
                ), f"num_scans_in_traj or scan_idxs must be provided.\n{self.overfit_config=}"
                for i in range(len(self.trajectories)):
                    self.trajectories[i] = self.trajectories[i].create_slice(
                        end_idx=min(num_scans_in_traj, len(self.trajectories[i])) - 1
                    )

        self.trajectories = subsample_ds_arr(
            self.trajectories,
            ds_portion=self.ds_portion,
            ds_portion_from_start=self.ds_portion_from_start,
            ds_portion_from_end=self.ds_portion_from_end,
        )

        self.trajectory_lengths = torch.zeros((len(self.trajectories),))
        for i, trajectory in enumerate(self.trajectories):
            self.trajectory_lengths[i] = len(trajectory)
            self.len += self.trajectory_lengths[i].item()

        print(
            f"Read {self.len} point clouds from {len(self.trajectories)} trajectories from .pkl"
        )


class PointCloudDatasetRos(PointCloudDataset):
    """A dataset for point clouds extracted from ROS bags."""

    def __init__(
        self,
        base_folder: str,
        map_dim: torch.Tensor,
        map_resolution: torch.Tensor,
        device: t.Union[torch.device, str] = torch.device("cuda"),
        use_cylindrical_coords: bool = False,
        use_sparse: bool = False,
        do_convert_pose_to_quat_and_t: bool = False,
        ds_portion=1.0,
        ds_portion_from_start=None,
        ds_portion_from_end=None,
        noisifier: t.Optional[t.Callable] = None,
        sequence_length: int = 1,
        step_skip: int = 1,
        num_pts: t.Optional[int] = None,
    ):
        self.base_folder = base_folder
        self.device = device
        self.use_sparse = use_sparse
        self.map_dim = map_dim.to(device)
        self.map_resolution = map_resolution.to(device)
        self.do_convert_pose_to_quat_and_t = do_convert_pose_to_quat_and_t
        self.use_cylindrical_coords = use_cylindrical_coords
        self.scans_paths = get_ordered_paths(
            f"{self.base_folder}/lidar/*.npy", p=".*/(\d+).\d+.npy"
        )
        self.scans_paths = subsample_ds_arr(
            self.scans_paths,
            ds_portion=ds_portion,
            ds_portion_from_start=ds_portion_from_start,
            ds_portion_from_end=ds_portion_from_end,
        )

        self.noisifier = noisifier
        self.trajectory_lengths = [
            1
        ]  # not meaningful. for compatibility with PointCloudDataset
        self.sequence_length = sequence_length
        self.step_skip = step_skip
        self.num_pts = num_pts

    def __len__(self) -> int:
        return len(self.scans_paths)

    def __getitem__(self, idx):
        sample_traj_seq = defaultdict(list)

        pose = torch.ones(7) if self.do_convert_pose_to_quat_and_t else torch.ones(4, 4)
        try:
            measured = np.load(self.scans_paths[idx])
        except Exception as e:
            print(f"Error loading {self.scans_paths[idx]}. Loading another random scan")
            traceback.print_exc()
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())
        center = np.array([0.0, 0.0, 0.0])
        measured = fetch_pts_in_lidar_range(
            measured,
            center,
            lidar_range=10.0,
            lidar_blind_range=4.0,  # this blind range allows not having holes in ground truth occupancy
        )

        measured = torch.tensor(measured, dtype=torch.float32)
        gt = measured.clone()

        sample = {
            "measured": measured,
            "gt": gt,
            "pose": pose,
            "is_synthetic": False,
        }

        parsed_sample = parse_sample(
            sample,
            self.noisifier,
            device=self.device,
            num_pts=self.num_pts,
            do_downsample_target=False,
        )

        measured = parsed_sample["measured"]
        gt = parsed_sample["gt"]
        mesh = parsed_sample["mesh"]

        sample_traj_seq["measured"].append(measured)
        sample_traj_seq["gt"].append(gt)
        if mesh is not None:
            sample_traj_seq["mesh"].append(mesh)
        for k in [
            "pose",
            "is_synthetic",
            "added_noise",
            "area_map",
        ]:
            sample_traj_seq[k].append(parsed_sample[k])

        return sample_traj_seq


def custom_collate_fn(batch):
    batch_traj_seq = defaultdict(list)
    known_keys = [
        "measured",
        "gt",
        "pose",
        "mesh",
        "area_map",
        "is_synthetic",
        "added_noise",
        "mean",
        "max_dist",
        "labels",
    ]
    for data_dict in batch:
        for k, v in data_dict.items():
            if k in known_keys:
                batch_traj_seq[k].append(v)
    return batch_traj_seq


def parse_sample(
    sample,
    noisifier,
    device,
    use_cylindrical_coords=False,
    robot_params=None,
    num_pts=None,
    do_downsample_target=False,
):
    measured = sample["measured"]
    gt = sample["gt"]
    pose = sample["pose"]
    is_synthetic = sample["is_synthetic"]
    mesh = sample.get("mesh")
    labels = sample.get("labels")
    added_noise = False

    if num_pts is not None:
        measured_res = downsample_pcl(measured, num_pts, use_randperm=True)
        measured = measured_res["pcl"]
        if labels is not None:
            labels = labels[measured_res["idx"]]
        if do_downsample_target:
            # unclear what are the positive effects of downsampling the complete point cloud
            idx = torch.randperm(measured.shape[0])[:num_pts]
            gt_res = downsample_pcl(gt, num_pts, idx=idx, use_randperm=False)
            gt = gt_res["pcl"]
            if mesh is not None:
                mesh_res = downsample_pcl(mesh, num_pts, idx=idx, use_randperm=False)
                mesh = mesh_res["pcl"]

    if noisifier:
        try:
            noise_res = noisifier(
                measured,
                gt,
                add_noise=True,
                pose=pose,
                labels=labels,
            )
            measured = noise_res["measured"]
            gt = noise_res["gt"]
            added_noise = noise_res["added_noise"]
            labels = noise_res["labels"]
        except Exception as e:
            print(f"Error noisifying {e}")
            traceback.print_exc()

    if use_cylindrical_coords:
        measured = xyz_to_cylindrical(measured)
        gt = xyz_to_cylindrical(gt)

    if mesh is None:
        mesh = gt.clone()

    # if pose.shape[0] == 4 and do_convert_pose_to_quat_and_t:
    #     pose = convert_pose_to_quat_and_t(pose)
    area_map = get_area_map(sample, robot_params=robot_params)

    added_noise = torch.tensor(added_noise, dtype=torch.float32, device=device)

    res = {
        "pose": pose,
        "area_map": area_map,
        "measured": measured,
        "gt": gt,
        "mesh": mesh,
        "is_synthetic": is_synthetic,
        "added_noise": added_noise,
        "labels": labels,
    }
    return res


def parse_traj(
    traj,
    sequence_length,
    step_skip,
    device,
    noisifier=None,
    use_cylindrical_coords=False,
    seq_start=None,
    num_pts=None,
):
    # parses a trajectory into a sequence of point clouds
    if seq_start is None:
        seq_start = torch.randint(
            0,
            max(1, len(traj) + 1 - sequence_length * step_skip),
            (1,),
        ).item()
    assert step_skip > 0, f"{step_skip=}"
    sample_traj_seq = defaultdict(list)

    for ts in range(sequence_length):
        time_idx = seq_start + ts * step_skip
        sample = traj[time_idx]

        parsed_sample = parse_sample(
            sample,
            noisifier=noisifier,
            device=device,
            use_cylindrical_coords=use_cylindrical_coords,
            num_pts=num_pts,
            do_downsample_target=False,
        )

        measured = parsed_sample["measured"]
        gt = parsed_sample["gt"]
        mesh = parsed_sample["mesh"]
        sample_traj_seq["measured"].append(measured)
        sample_traj_seq["gt"].append(gt)
        if mesh is not None:
            sample_traj_seq["mesh"].append(mesh)
        for k in [
            "pose",
            "is_synthetic",
            "added_noise",
            "labels",
        ]:
            sample_traj_seq[k].append(parsed_sample[k])

    return sample_traj_seq


def get_area_map(terrain_name, robot_params=None):
    """Get the structure of the terrain for the given sample if available."""
    area_map = {}
    if robot_params is not None:
        if terrain_name in robot_params:
            area_map = robot_params[terrain_name]["area_map"]
    return area_map


def subsample_ds_arr(arr, ds_portion, ds_portion_from_start, ds_portion_from_end):
    assert not ds_portion < 1.0 and (
        ds_portion_from_end or ds_portion_from_start
    ), "Either ds_portion or (ds_portion_from_end | ds_portion_from_start) must be provided"
    if ds_portion < 1.0:
        random_idxs = torch.randperm(len(arr))[: int(len(arr) * ds_portion)]
        arr = [arr[i] for i in random_idxs]
    elif ds_portion_from_start:
        arr = arr[: int(len(arr) * ds_portion_from_start)]
    elif ds_portion_from_end:
        arr = arr[-int(len(arr) * ds_portion_from_end) :]
    return arr


def traj_path_to_terrain_name(traj_path):
    assert "terrain_" in traj_path, f"{traj_path=}"
    p = re.compile(r".*/(terrain_\d+)/.*")
    return p.match(traj_path).group(1)
