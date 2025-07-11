import matplotlib.pyplot as plt
import numpy as np
import scipy


def estimate_density(point_cloud: np.ndarray) -> list:
    """
    Estimate the density of points along each dimension in a point cloud.

    Parameters:
        point_cloud: The point cloud represented as a 2D array where each row is a point.

    Returns:
        A list containing density estimates along each dimension.
    """
    # Calculate histograms along each dimension
    densities = []
    for dim in range(point_cloud.shape[1]):
        hist, bins = np.histogram(point_cloud[:, dim], bins=50, density=True)
        densities.append(hist)

    return densities


def est_and_plot_pcl_density(pcl: np.ndarray) -> None:
    density_estimates = estimate_density(pcl)

    # Plot density estimates
    fig, axs = plt.subplots(1, pcl.shape[1], figsize=(15, 5))
    for i, density in enumerate(density_estimates):
        axs[i].bar(range(len(density)), density, width=1)
        axs[i].set_title(f"Dimension {i}")
    plt.tight_layout()
    plt.show()


def compute_pt_distances(pts: np.ndarray) -> dict:
    cdists = scipy.spatial.distance.cdist(pts, pts, "euclidean")
    return {
        "min": cdists.min(),
        "max": cdists.max(),
        "mean": cdists.mean(),
        "std": cdists.std(),
    }


def augm_pcl_by_perturbing_pts(pcl, n=10, std=0.01):
    pcls = []
    for i in range(n):
        pcls.append(pcl + np.random.normal(0, std, pcl.shape))
    return np.concatenate(pcls, axis=0)


def get_rotation_matrix_from_degrees(rotation_degrees: tuple) -> np.ndarray:
    """
    Get a 3x3 rotation matrix from the given rotation angles in degrees.

    Parameters:
        rotation_degrees: The rotation angles in degrees for the x, y, and z axes.

    Returns:
        The 3x3 rotation matrix.
    """
    # Convert degrees to radians
    rotation_radians = np.radians(rotation_degrees)

    # Get the rotation matrix
    rotation_matrix = np.array(
        [
            [
                np.cos(rotation_radians[1]) * np.cos(rotation_radians[2]),
                np.cos(rotation_radians[1]) * np.sin(rotation_radians[2]),
                -np.sin(rotation_radians[1]),
            ],
            [
                np.sin(rotation_radians[0])
                * np.sin(rotation_radians[1])
                * np.cos(rotation_radians[2])
                - np.cos(rotation_radians[0]) * np.sin(rotation_radians[2]),
                np.sin(rotation_radians[0])
                * np.sin(rotation_radians[1])
                * np.sin(rotation_radians[2])
                + np.cos(rotation_radians[0]) * np.cos(rotation_radians[2]),
                np.sin(rotation_radians[0]) * np.cos(rotation_radians[1]),
            ],
            [
                np.cos(rotation_radians[0])
                * np.sin(rotation_radians[1])
                * np.cos(rotation_radians[2])
                + np.sin(rotation_radians[0]) * np.sin(rotation_radians[2]),
                np.cos(rotation_radians[0])
                * np.sin(rotation_radians[1])
                * np.sin(rotation_radians[2])
                - np.sin(rotation_radians[0]) * np.cos(rotation_radians[2]),
                np.cos(rotation_radians[0]) * np.cos(rotation_radians[1]),
            ],
        ]
    )

    return rotation_matrix
