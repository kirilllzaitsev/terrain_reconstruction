import typing as t

import meshio
import numpy as np
import open3d as o3d


def save_npy_to_pcd(npy_path: str, pcd_path: t.Optional[str] = None) -> str:
    """Having a .npy file, saves it as a .pcd file."""

    xyz = np.load(npy_path)
    save_path = npy_path.replace(".npy", ".pcd") if pcd_path is None else pcd_path
    save_pcl(xyz, save_path, format="pcd")
    return save_path


def save_pcl(
    pcl: t.Union[np.ndarray, o3d.geometry.PointCloud],
    path: str,
    format: t.Optional[str] = None,
) -> None:
    """Saves a point cloud to a file. The format is inferred from the path or can be specified. PLY files are saved in float (32-bit) format."""

    if format is None:
        format = path.split(".")[-1]
    if format in ["pcd", "ply"]:
        if "." not in path:
            path = f"{path}.{format}"
        if not isinstance(pcl, o3d.geometry.PointCloud):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcl)
            pcl = pcd
        o3d.io.write_point_cloud(path, pcl)
        if format == "ply":
            mesh = meshio.read(path)
            mesh.points = mesh.points.astype(np.float32)
            mesh.write(path)
    elif format == "npy":
        np.save(path, pcl)
    else:
        raise ValueError(f"Unsupported format: {format}")
