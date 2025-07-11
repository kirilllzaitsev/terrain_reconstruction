import queue
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import vedo
from PIL import Image
from terrain_representation.utils.pointcloud_utils import (
    cylindrical_to_xyz,
    point_cloud_to_heightmap,
)
from torch.multiprocessing import set_start_method


class PointCloudVisualizer:
    """Class for asynchronous visualization of point clouds."""

    def __init__(
        self,
        extra_axes_params=None,
        figure_name="",
        input_in_cylindrical=False,
        use_pyplot=False,
        pts_radius=10,
    ):
        self.extra_plot_params = extra_axes_params or dict(
            interactive=True,
            resetcam=False,
        )
        self.res_queue = queue.Queue()
        self.pts_radius = pts_radius

        self.figure_name = figure_name
        self.input_in_cylindrical = input_in_cylindrical
        self.data_key_to_axes_name = {
            "measurement": "measurement",
            "input": "input",
            "output": "output",
            "occupancy": "occupancy",
            "gt": "gt_scan",
            "mesh": "gt_mesh",
        }
        self.offscreen = self.extra_plot_params.pop("offscreen", False)
        self.use_pyplot = use_pyplot

    def display_point_clouds(self, point_cloud_dict: dict, use_pyplot=None):
        """Puts the point cloud data into the queue for rendering."""
        return self._display_point_clouds(point_cloud_dict, use_pyplot)

    def _display_point_clouds(self, data_dict, use_pyplot=None):
        use_pyplot = use_pyplot or self.use_pyplot
        if use_pyplot:
            vis_res = plot_data_dict_pyplot(
                data_dict,
                self.data_key_to_axes_name,
                data_dict["log_img_name"],
                data_dict["step"],
            )
        else:
            vis_res = plot_data_dict(
                data_dict,
                self.offscreen,
                self.figure_name,
                self.extra_plot_params,
                self.input_in_cylindrical,
                self.data_key_to_axes_name,
                data_dict["log_img_name"],
                data_dict["step"],
                pts_radius=self.pts_radius,
            )
        if vis_res is not None:
            self.res_queue.put(vis_res)
        return vis_res

    def display_single_point_cloud(self, point_cloud, title="", step=0):
        """Display a single point cloud."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig = display_single_point_cloud_pyplot(point_cloud, title, ax)
        vis_res = get_vis_res(title, step, fig)
        self.res_queue.put(vis_res)
        return vis_res

    def display_input_output_pcls(self, input_pcl, output_pcl, title="", step=0):
        """Display a single point cloud."""
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        display_single_point_cloud_pyplot(input_pcl, "", ax[0])
        display_single_point_cloud_pyplot(output_pcl, "", ax[1])
        vis_res = get_vis_res(title, step, fig)
        self.res_queue.put(vis_res)
        return vis_res

    def stop(self):
        """Stop the visualization process."""
        pass


class PointCloudVisualizerAsync(PointCloudVisualizer):
    """Class for asynchronous visualization of point clouds."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self.queue = mp.Queue()
        self.res_queue = mp.Queue()
        self.process = mp.Process(
            target=self._render_loop, args=(self.queue,), daemon=True
        )
        self.process.start()

    def display_point_clouds(self, point_cloud_dict: dict):
        """Puts the point cloud data into the queue for rendering."""
        self.queue.put(("update", point_cloud_dict))

    def _render_loop(self, queue):
        """Render loop for the visualizer. This function is called in a separate process.
        The mechanics of this loop is as follows:
        - The visualizer waits for the queue to be filled with data (at least one element).
        - When the queue is filled, the visualizer processes the data and optionally displays it.
        - The visualizer then puts the rendered image in the result queue.
        - The main process can then get the result from the result queue and use it as needed.
        """
        while True:
            if queue.empty():
                time.sleep(0.04)
                continue
            try:

                cmd, data_dict = queue.get_nowait()
                if cmd == "exit":
                    break

                self._display_point_clouds(data_dict)

            except Exception as e:
                print(e)
                raise e

            time.sleep(0.001)

    def stop(self):
        """Stop the visualization process."""
        self.queue.put(("exit", None))
        self.process.join(timeout=5)
        return super().stop()

    def display_point_clouds_as_heightmap(
        self,
        point_cloud_array,
        map_dim=[64, 64, 64],
        map_resolution_xyz=[0.05, 0.05, 0.05],
        window_name="Heightmap",
        display_speed_ms=1,
    ):
        """Display point clouds as elevation maps."""
        height_maps = list()
        device = point_cloud_array[0].device
        map_dim = torch.tensor(map_dim, device=device)
        map_resolution_xyz = torch.tensor(map_resolution_xyz, device=device)
        for i, point_cloud in enumerate(point_cloud_array):
            height_map = point_cloud_to_heightmap(
                point_cloud.clone(),
                None,
                map_dim=map_dim,
                map_resolution_xyz=map_resolution_xyz,
            )
            height_maps.append(height_map.cpu().numpy() - 0.8)

        display_images(window_name, height_maps, display_speed_ms)


def plot_data_dict(
    data_dict: dict,
    offscreen,
    figure_name,
    extra_plot_params,
    input_in_cylindrical,
    data_key_to_axes_name,
    log_img_name,
    step,
    pts_radius=10,
):
    log_img_name = data_dict.pop("log_img_name")
    step = data_dict.pop("step")
    num_axes = len(data_dict) if len(data_dict) % 2 == 0 else len(data_dict) + 1
    plotter = vedo.Plotter(
        N=num_axes,
        offscreen=offscreen,
    )

    pointclouds = list()
    legends = list()

    for key, elem in data_dict.items():
        if key == "occupancy":
            data = elem["occupancy_points"].detach().cpu().numpy().squeeze()
            if input_in_cylindrical:
                data = cylindrical_to_xyz(data)
            scale = elem["occupancy_values"].detach().cpu().numpy().squeeze()
            pointcloud = vedo.Points(data)
            pointcloud.cmap("jet", scale, vmin=0.5, vmax=1)
            pointcloud.addScalarBar3D(title="occupancy prob")
        else:
            data = elem.detach().cpu().numpy().squeeze()
            if input_in_cylindrical:
                data = cylindrical_to_xyz(data)
            pointcloud = vedo.Points(data)
            # pointcloud.cmap("rainbow", data[:, 2], vmin=-2, vmax=2)
            pointcloud.cmap("rainbow", data[:, 2], vmin=None, vmax=None)
            pointcloud.addScalarBar3D(title="height")
        pointclouds.append(pointcloud)
        legends.append(data_key_to_axes_name.get(key, key))

    legends += [""] * (num_axes - len(legends))
    pointclouds.extend([[]] * (num_axes - len(pointclouds)))
    assert len(pointclouds) == len(legends), f"{len(pointclouds)} != {len(legends)}"
    assert len(legends) % 2 == 0 and len(pointclouds) % 2 == 0

    vis_res = None
    for idx, pointcloud in enumerate(pointclouds):
        fig = plotter.show(
            pointcloud,
            legends[idx],
            at=idx,
            axes=dict(
                c="black",
                xrange=(-pts_radius, pts_radius),
                yrange=(-pts_radius, pts_radius),
                zrange=(-pts_radius, pts_radius),
                xyShift=0.5,
                numberOfDivisions=10,
            ),
            title=figure_name,
            camera=dict(
                position=(
                    5.233653722575609,
                    4.805790731006791,
                    21.428978640333078,
                ),
            ),
            **extra_plot_params,
        )
        if idx == len(pointclouds) - 1:
            vis_res = {
                "fig": (
                    np.asarray(fig)
                    if isinstance(fig, Image.Image)
                    else fig.topicture().tonumpy()
                ),
                "log_img_name": log_img_name,
                "step": step,
            }
    return vis_res


def plot_single_point_cloud_vedo(
    point_cloud,
    offscreen=False,
    extra_plot_params=None,
    log_img_name="",
    step=0,
    pts_radius=10,
):
    extra_plot_params = extra_plot_params or dict(
        interactive=True,
        resetcam=False,
    )
    plotter = vedo.Plotter(
        offscreen=offscreen,
    )

    point_cloud = point_cloud.detach().cpu().numpy().squeeze()
    pointcloud = vedo.Points(point_cloud)
    pointcloud.cmap("rainbow", point_cloud[:, 2], vmin=None, vmax=None)
    pointcloud.addScalarBar3D(title="height", pos=(10, 0, 0))

    fig = plotter.show(
        pointcloud,
        axes=dict(
            c="black",
            xrange=(-pts_radius, pts_radius),
            yrange=(-pts_radius, pts_radius),
            zrange=(-pts_radius, pts_radius),
            xyShift=0.5,
            numberOfDivisions=10,
        ),
        camera=dict(
            position=(
                5.233653722575609,
                4.805790731006791,
                21.428978640333078,
            ),
        ),
        **extra_plot_params,
    )

    vis_res = {
        "fig": (
            np.asarray(fig)
            if isinstance(fig, Image.Image)
            else fig.topicture().tonumpy()
        ),
        "log_img_name": log_img_name,
        "step": step,
    }
    return vis_res


def display_single_point_cloud_pyplot(point_cloud, title="", ax=None):
    """Display a single point cloud."""
    ax_is_none = ax is None
    if ax_is_none:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    sc = ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        c=point_cloud[:, 2],
        s=10,
        cmap="rainbow",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("height", fontsize=14)

    ax.set_title(title)
    ax.set_axis_off()
    if ax_is_none:
        return fig


def plot_data_dict_pyplot(
    data_dict: dict,
    data_key_to_axes_name,
    log_img_name,
    step,
):
    log_img_name = data_dict.pop("log_img_name")
    step = data_dict.pop("step")

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for idx, (key, elem) in enumerate(data_dict.items()):
        ax = axs[idx // 3, idx % 3]
        if key == "occupancy":
            data = elem["occupancy_points"].detach().cpu().numpy()
            scale = elem["occupancy_values"].detach().cpu().numpy()
            sc = ax.scatter(
                data[:, 0],
                data[:, 1],
                c=scale,
                cmap="coolwarm",
                vmin=0.5,
                vmax=1,
                s=10,
            )
            plt.colorbar(sc, ax=ax, label="occupancy prob")
        else:
            data = elem.detach().cpu().numpy()
            sc = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap="coolwarm", s=10)
            plt.colorbar(sc, ax=ax, label="height")
        ax.set_title(data_key_to_axes_name.get(key, key))
        # ax.set_facecolor("lightgrey")

    # Adjust layout
    plt.tight_layout()
    vis_res = get_vis_res(log_img_name, step, fig)
    return vis_res


def get_vis_res(log_img_name, step, fig):
    vis_res = {
        "log_img_name": log_img_name,
        "step": step,
    }
    if fig is None:
        vis_res["fig"] = None
    else:
        width, height = fig.canvas.get_width_height()
        fig.canvas.draw()
        vis_res["fig"] = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype="uint8"
        ).reshape(height, width, 3)
        plt.close(fig)
    return vis_res


def display_images(name: str, images: list, speed: int = 1) -> None:
    """Display images in a single window via opencv."""
    if isinstance(images[0], list):
        concats = None
        for i in range(len(images)):
            rows = np.concatenate(images[i], axis=1)
            if concats is None:
                concats = rows
            else:
                concats = np.concatenate((concats, rows), axis=0)
    else:
        concats = np.concatenate(images, axis=1)

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, concats)
    cv2.waitKey(speed)
