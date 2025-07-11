import glob
import json
import os
import shutil
import typing as t
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from terrain_representation.storage.trajectory import (
    PointCloudTrajectory,
    PointCloudTrajectoryStorage,
)
from terrain_representation.utils.comet_utils import root_dir
from terrain_representation.utils.pointcloud_utils import (
    flip_pts_along_y_axis,
    get_pts_mask_within_azimuth_angle,
)
from terrain_synthesis.utils.postprocessing_utils import (
    conv_to_meters,
    conv_xyhw_to_xyxy,
)
from terrain_synthesis.utils.utils import get_ordered_paths
from tqdm.auto import tqdm

base_save_dir = f"{root_dir}/data/train"
base_save_dir_val = base_save_dir.replace("/train", "/validation")


def assemble_trajectory(
    measureds: t.List[np.ndarray],
    poses: t.List[np.ndarray],
    terrain_name: t.Optional[str] = None,
    meshes: t.Optional[t.List[np.ndarray]] = None,
    gts: t.Optional[t.List[np.ndarray]] = None,
    is_synthetic: bool = True,
    apply_rotation: bool = False,
    apply_mask: bool = False,
    apply_shift: bool = False,
    area_map: t.Optional[dict] = None,
) -> PointCloudTrajectory:
    traj = PointCloudTrajectory()
    for i in range(len(measureds)):
        mesh = None if meshes is None else meshes[i]
        gt = None if gts is None else gts[i]
        add_new_traj_sample(
            traj,
            measureds[i],
            poses[i],
            terrain_name,
            mesh=mesh,
            gt=gt,
            is_synthetic=is_synthetic,
            apply_rotation=apply_rotation,
            apply_mask=apply_mask,
            apply_shift=apply_shift,
            area_map=area_map,
        )
    return traj


def load_scan(measured_path, pose_path, mesh_path, min_gt_pts=200):
    measured = np.load(measured_path)
    pose = np.load(pose_path)
    mesh = np.load(mesh_path) if mesh_path.exists() else None
    if measured.shape[0] < min_gt_pts:
        print(
            f"Not enough points in the ground truth: {len(measured)=} < {min_gt_pts}. Skipping {measured_path=}"
        )
        return None
    return {"measured": measured, "pose": pose, "mesh": mesh}


def assemble_trajectory_from_path(
    traj_path,
    is_synthetic=True,
    num_scans=None,
    scan_idxs=None,
    min_gt_pts=300,
    apply_rotation=False,
    apply_mask=False,
    apply_shift=False,
    area_map=None,
):
    if ".npy" not in traj_path:
        traj_path = f"{traj_path}.npy"
    result = np.load(traj_path, allow_pickle=True).item()
    measureds = [m for m in result["measured"]]
    poses = [p for p in result["pose"]]
    meshes = [m for m in result["mesh"]]
    if scan_idxs is not None:
        measureds = [measureds[i] for i in scan_idxs]
        poses = [poses[i] for i in scan_idxs]
        meshes = [meshes[i] for i in scan_idxs]
    if num_scans is not None:
        measureds = measureds[:num_scans]
        poses = poses[:num_scans]
        meshes = meshes[:num_scans]
    idxs_to_remove = []
    for i, measured in enumerate(measureds):
        if measured.shape[0] < min_gt_pts:
            idxs_to_remove.append(i)
    measureds = [m for i, m in enumerate(measureds) if i not in idxs_to_remove]
    poses = [p for i, p in enumerate(poses) if i not in idxs_to_remove]
    meshes = [m for i, m in enumerate(meshes) if i not in idxs_to_remove]
    return assemble_trajectory(
        measureds,
        poses,
        Path(traj_path).parent.name,
        meshes,
        gts=None,
        is_synthetic=is_synthetic,
        apply_rotation=apply_rotation,
        apply_mask=apply_mask,
        apply_shift=apply_shift,
        area_map=area_map,
    )


def assemble_trajectory_from_path_v1(
    traj_path, is_synthetic=True, num_scans=None, scan_idxs=None, min_gt_pts=200
):
    traj_dir = Path(traj_path)
    measured_paths = sorted(list(traj_dir.glob("measured/*.npy")))
    pose_paths = sorted(list(traj_dir.glob("pose/*.npy")))
    mesh_paths = sorted(list(traj_dir.glob("mesh/*.npy")))
    measureds = []
    poses = []
    meshes = []

    for measured_path, pose_path, mesh_path in zip(
        measured_paths, pose_paths, mesh_paths
    ):
        result = load_scan(measured_path, pose_path, mesh_path, min_gt_pts)
        if result is not None:
            if num_scans is not None and len(measureds) >= num_scans:
                break
            if scan_idxs is not None and len(measureds) not in scan_idxs:
                continue
            measureds.append(result["measured"])
            poses.append(result["pose"])
            meshes.append(result["mesh"])
    return assemble_trajectory(
        measureds,
        poses,
        traj_dir.parent.name,
        meshes,
        gts=None,
        is_synthetic=is_synthetic,
    )


def add_new_traj_sample(
    traj: PointCloudTrajectory,
    measured: np.ndarray,
    pose: np.ndarray,
    terrain_name: t.Optional[str] = None,
    mesh: t.Optional[np.ndarray] = None,
    is_synthetic: bool = True,
    gt: t.Optional[np.ndarray] = None,
    apply_rotation: bool = False,
    apply_mask: bool = False,
    apply_shift: bool = False,
    area_map: t.Optional[dict] = None,
) -> None:
    """Adds a new sample to the trajectory. The sample includes:
    - (centered) ground truth point cloud
    - (centered) measured point cloud
    - (centered) mesh
    - axes_means used to center the point clouds
    - sensor pose
    - name of the terrain (e.g., "terrain_0")
    """

    t = pose[:3, 3]
    if apply_mask:
        scan_mean_pt = measured.mean(0)
        view_vector = scan_mean_pt - t
        azimuth_scan_span_deg = 140
        mask = get_pts_mask_within_azimuth_angle(
            measured,
            t,
            view_vector,
            np.deg2rad(azimuth_scan_span_deg),
            inner_angle_rad=np.deg2rad(np.random.uniform(0, 40)),
        )
        measured = measured[mask]
        if gt is not None:
            gt = gt[mask]

    labels = None
    if area_map is not None:
        labels = np.zeros(measured.shape[0]) - 1
        # get the labels for the points
        measured_flipped = flip_pts_along_y_axis(measured, center=t)
        for shape_id, (shape_type, shape_instance_coords) in enumerate(
            area_map.items()
        ):
            shape_instance_coords_xyxy = conv_xyhw_to_xyxy(
                conv_to_meters(shape_instance_coords), is_y_axis_top_down=False
            )

            for shape_instance in shape_instance_coords_xyxy:
                # where pt.x >= x1 and pt.x <= x2 and pt.y >= y1 and pt.y <= y2, assign the shape_id
                shape_mask = np.logical_and(
                    np.logical_and(
                        measured_flipped[:, 0] >= shape_instance[0],
                        measured_flipped[:, 0] <= shape_instance[2],
                    ),
                    np.logical_and(
                        measured_flipped[:, 1] >= shape_instance[1],
                        measured_flipped[:, 1] <= shape_instance[3],
                    ),
                )

                labels[shape_mask] = shape_id

    pose = torch.from_numpy(pose).float()
    R = pose[:3, :3]
    sensor_translation = pose[:3, 3]

    measured = torch.from_numpy(measured).float()
    measured -= sensor_translation

    if mesh is not None:
        mesh = torch.from_numpy(mesh).float()
        mesh -= sensor_translation
    if gt is None:
        gt = measured.clone()
    else:
        gt = torch.from_numpy(gt).float()
        gt -= sensor_translation

    if apply_rotation:
        measured = torch.matmul(R.T, measured.T).T
        mesh = torch.matmul(R.T, mesh.T).T
        gt = torch.matmul(R.T, gt.T).T

    if apply_shift:
        # shift to center
        shift = get_shift_of_pts_in_sensor_frame_to_center(measured)
        measured -= shift
        mesh -= shift
        gt -= shift

    traj.append(
        measured=measured,
        gt=gt,
        pose=pose,
        terrain_name=terrain_name,
        mesh=mesh,
        is_synthetic=is_synthetic,
        labels=torch.from_numpy(labels) if labels is not None else None,
    )


def split_list_into_n_chunks(lst: t.List, n: int) -> t.List[t.List]:
    """Splits a list into n chunks."""
    k, m = divmod(len(lst), n)
    chunks = [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]
    return [c for c in chunks if c]


def convert_numpy_to_trajectory(
    scan_dir: str,
    save_dir_name: str,
    val_pkls_num: int = 0,
    val_pkls_synt_num: int = 0,
    val_pkls_idxs: t.Optional[t.List[int]] = None,
) -> None:
    """Parses the numpy files in the given directory and saves them as a set of trajectories. Trajectories are ignored if they have less than 3 scans. save_dir_name is the name of the directory where .pkl files will be saved. If val_pkls_num is not 0, the last val_pkls_num .pkl files will be saved in the validation directory.
    Expected directory structure:
    scan_dir
    ├── terrain_0
    │   ├── metadata.json
    │   ├── traj_0
    │   │   ├── gt
    │   │   │   ├── 0.npy
    │   │   │   ├── 1.npy
    │   │   │   └── ...
    │   │   ├── measured
    │   │   │   ├── 0.npy
    │   │   │   ├── 1.npy
    │   │   │   └── ...
    │   │   ├── pose
    │   │   │   ├── 0.npy
    │   │   │   ├── 1.npy
    │   │   │   └── ...
    │   │   └── mesh
    │   │       ├── 0.npy
    │   │       ├── 1.npy
    │   │       └── ...
    ├── terrain_1
    │   ...
    └── terrain_N
    """

    terrain_dirs = get_ordered_paths(scan_dir, p=r".*terrain_(\d+).*")

    is_synthetic_masks = []
    metadata_found = False
    for terrain_dir in terrain_dirs:
        metadata_path = f"{terrain_dir}/metadata.json"
        if os.path.exists(metadata_path):
            metadata = json.load(open(metadata_path))
            is_synthetic = metadata["is_synthetic"]
            metadata_found = True

        else:
            is_synthetic = True
        is_synthetic_masks.append(is_synthetic)
    if not metadata_found:
        print("metadata.json file not found. Assuming all terrains are synthetic.")

    val_save_dir = f"{base_save_dir_val}/{save_dir_name}"
    train_save_dir = f"{base_save_dir}/{save_dir_name}"
    if args.do_reload:
        shutil.rmtree(val_save_dir, ignore_errors=True)
        shutil.rmtree(train_save_dir, ignore_errors=True)

    processed_terrain_names = [
        Path(x).name
        for x in get_ordered_paths(f"{val_save_dir}/terrain_*", p=r".*terrain_(\d+).*")
    ] + [
        Path(x).name
        for x in get_ordered_paths(
            f"{train_save_dir}/terrain_*", p=r".*terrain_(\d+).*"
        )
    ]

    terrain_dirs = [
        d for d in terrain_dirs if Path(d).name not in processed_terrain_names
    ]

    ignored_trajectories = {
        val_save_dir: defaultdict(list),
        train_save_dir: defaultdict(list),
    }

    num_workers = args.num_workers

    total_samples = {val_save_dir: 0, train_save_dir: 0}
    terrain_dirs_idx_chunks = split_list_into_n_chunks(
        range(len(terrain_dirs)), len(terrain_dirs) // args.pkl_size
    )

    if val_pkls_num > 0:
        val_pkls_idxs = (
            val_pkls_idxs
            if val_pkls_idxs is not None
            else np.random.choice(
                range(len(terrain_dirs_idx_chunks)), val_pkls_num, replace=False
            ).tolist()
        )
    for chunk_idx, terrain_dir_idxs in enumerate(
        tqdm(terrain_dirs_idx_chunks, desc="Chunks")
    ):
        save_dir = val_save_dir if chunk_idx in val_pkls_idxs else train_save_dir
        data_src = "synt" if is_synthetic else "real"
        chunk_name = (
            f"terrains_{min(terrain_dir_idxs)}_{max(terrain_dir_idxs)}_{data_src}"
        )
        if os.path.exists(f"{save_dir}/{chunk_name}.pkl"):
            print(f"Skipping the chunk {chunk_name=} because it already exists.")
            continue

        terrain_dirs_chunk = [terrain_dirs[i] for i in terrain_dir_idxs]
        is_synthetic_masks_chunk = [is_synthetic_masks[i] for i in terrain_dir_idxs]
        total_samples_in_pack = 0
        storage = PointCloudTrajectoryStorage()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for result in tqdm(
                executor.map(
                    worker_function_p,
                    zip(terrain_dirs_chunk, is_synthetic_masks_chunk),
                    chunksize=len(terrain_dirs) // args.pkl_size // num_workers,
                ),
                total=len(terrain_dirs_chunk),
                desc="Terrains",
            ):
                ignored_trajectories[save_dir][result["terrain_name"]] = result[
                    "ignored_trajectories"
                ]
                total_samples_in_pack += result["total_samples"]
                storage.merge(PointCloudTrajectoryStorage.load(result["storage_path"]))
        print(f"Chunk {chunk_idx}. Merged the results.")
        total_samples[save_dir] += total_samples_in_pack

        is_synthetic = all(is_synthetic_masks_chunk)
        assert any(is_synthetic_masks_chunk) == all(
            is_synthetic_masks_chunk or [False]
        ), "All terrains in a pack should be either synthetic or real"

        os.makedirs(save_dir, exist_ok=True)

        storage.dump(f"{save_dir}/{chunk_name}.pkl")
        print(f"Chunk {chunk_idx}. Stored the results.")

        robot_params_path_dest = f"{save_dir}/robot_params.json"
        if is_synthetic and not os.path.exists(robot_params_path_dest):
            shutil.copy(
                f"{Path(scan_dir).parent}/robot_params.json", robot_params_path_dest
            )

        print(f"Total samples in a pack: {total_samples_in_pack}")

    print(f"Total samples: {total_samples}")
    for d in [val_save_dir, train_save_dir]:
        print(f"{ignored_trajectories[d]=}")
        print(f"{total_samples[d]=}")

        with open(f"{d}/metadata.json", "w") as f:
            metadata = {
                "total_samples": total_samples[d],
                "ignored_trajectories": ignored_trajectories[d],
            }
            json.dump(metadata, f)


def worker_function_p(inputs):
    terrain_dir, is_synthetic = inputs
    storage = PointCloudTrajectoryStorage()
    terrain_name = Path(terrain_dir).name
    ignored_trajectories = []

    total_samples = 0
    for traj_dir in glob.glob(f"{terrain_dir}/*"):
        if not os.path.isdir(traj_dir):
            continue
        traj_dir = Path(traj_dir)
        gt_paths = sorted(list(traj_dir.glob("gt/*.npy")))
        measured_paths = sorted(list(traj_dir.glob("measured/*.npy")))
        pose_paths = sorted(list(traj_dir.glob("pose/*.npy")))
        mesh_paths = sorted(list(traj_dir.glob("mesh/*.npy")))
        gts = []
        measureds = []
        poses = []
        meshes = []
        for gt_path, measured_path, pose_path, mesh_path in zip(
            gt_paths, measured_paths, pose_paths, mesh_paths
        ):
            gt = np.load(gt_path)
            measured = np.load(measured_path)
            pose = np.load(pose_path)
            mesh = np.load(mesh_path) if mesh_path.exists() else None
            if gt.shape[0] == 0 or measured.shape[0] == 0:
                print(
                    f"Skipping {gt_path=} and {measured_path=}. Provided shapes are invalid: {gt.shape=} and {measured.shape=}"
                )
                continue
            min_pts_in_gt = 200
            if len(gt) < min_pts_in_gt:
                print(
                    f"Not enough points in the ground truth: {len(gt)=} < {min_pts_in_gt}. Skipping {gt_path=}"
                )
                continue
            gts.append(gt)
            measureds.append(measured)
            poses.append(pose)
            meshes.append(mesh)
        if len(gts) < 2:
            print(
                f"Skipping {traj_dir} of the terrain {terrain_name} because it has less than 3 samples"
            )
            ignored_trajectories.append(str(traj_dir.name))
            continue
        traj = assemble_trajectory(
            gts,
            measureds,
            poses,
            terrain_name=terrain_name,
            meshes=meshes,
            is_synthetic=is_synthetic,
        )

        total_samples += len(traj)

        storage.append(traj)

    tmp_dir = "/tmp/storages"
    os.makedirs(tmp_dir, exist_ok=True)
    storage_path = f"{tmp_dir}/{terrain_name}.pkl"
    storage.dump(storage_path)

    if len(ignored_trajectories) > 0:
        print(f"{ignored_trajectories=}")
    return {
        "ignored_trajectories": ignored_trajectories,
        "total_samples": total_samples,
        "storage_path": storage_path,
        "terrain_name": terrain_name,
    }


def get_shift_of_pts_in_sensor_frame_to_center(
    pts, mins=None, maxs=None
) -> t.Union[np.ndarray, torch.Tensor]:
    if isinstance(pts, torch.Tensor):
        mins = pts.min(0)[0] if mins is None else mins
        maxs = pts.max(0)[0] if maxs is None else maxs
        random_shift = torch.zeros(3)
    else:
        mins = pts.min(0) if mins is None else mins
        maxs = pts.max(0) if maxs is None else maxs
        random_shift = np.zeros(3)
    if (mins[0] < 0 and maxs[0] < 0) or (mins[0] > 0 and maxs[0] > 0):
        # shift x axis
        if mins[0] < 0 and maxs[0] < 0:
            max_lim = maxs[0]
        else:
            max_lim = mins[0]
        random_shift[0] = 2 * np.random.uniform(0, max_lim)
        random_shift[0] = 2 * max_lim
    elif (mins[1] < 0 and maxs[1] < 0) or (mins[1] > 0 and maxs[1] > 0):
        # shift y axis
        if mins[1] < 0 and maxs[1] < 0:
            max_lim = maxs[1]
        else:
            max_lim = mins[1]
        random_shift[1] = 2 * np.random.uniform(0, max_lim)
        random_shift[1] = 2 * max_lim

    return random_shift


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_ds_name",
        required=True,
        help="Name of the saved dataset. E.g., 'terrain_ds_v10'",
    )
    parser.add_argument(
        "--ds_base_dir",
        required=True,
        help="Base directory where Isaac scans are stored",
    )
    parser.add_argument(
        "--ds_subdir",
        default="collected_data",
        help="Subdirectory in the base directory",
    )
    parser.add_argument(
        "--val_pkls_num",
        type=int,
        default=0,
        help="Number of .pkl files to save in the validation directory",
    )
    parser.add_argument(
        "--val_pkls_synt_num",
        type=int,
        default=0,
        help="Number of .pkl files with synthetic data to use for validation",
    )
    parser.add_argument(
        "--pkl_size",
        type=int,
        default=1,
        help="Number of terrains to save in each .pkl file",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Number of workers to use for processing terrains",
    )
    parser.add_argument(
        "--val_pkls_idxs",
        nargs="+",
        type=int,
        default=[2],
        help="Indices of .pkl files to save in the validation directory",
    )
    parser.add_argument(
        "--do_reload",
        action="store_true",
    )
    args, _ = parser.parse_known_args()
    if args.val_pkls_num > 0:
        print(
            f"Use --val_pkls_idxs to specify the indices of .pkl files to save in the validation directory. {args.val_pkls_idxs=}"
        )
    args.val_pkls_num = len(args.val_pkls_idxs)
    if args.val_pkls_num > 0:
        assert (
            args.val_pkls_synt_num <= args.val_pkls_num
        ), f"{args.val_pkls_synt_num=} > {args.val_pkls_num=} is not allowed"
    args.val_pkls_synt_num = args.val_pkls_synt_num or args.val_pkls_num
    storage = convert_numpy_to_trajectory(
        scan_dir=f"{args.ds_base_dir}/{args.ds_subdir}",
        save_dir_name=args.save_ds_name,
        val_pkls_num=args.val_pkls_num,
        val_pkls_synt_num=args.val_pkls_synt_num,
        val_pkls_idxs=args.val_pkls_idxs,
    )
