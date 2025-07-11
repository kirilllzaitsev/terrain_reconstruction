import typing as t
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import rosbag
from terrain_representation.utils.pointcloud_utils import downsample_point_cloud
from tqdm.auto import tqdm


@lru_cache(maxsize=10)
def vis_rosbag(path: str, topic_name: str, take_n: t.Optional[int] = None) -> list:
    """Parses and visualizes the messages from the rosbag file."""
    msgs = read_rosbag_msgs(path, topic_name, take_n)
    print_em_messages(msgs)
    return msgs


def read_rosbag_msgs(
    path: str, topic_name: str, take_n: t.Optional[int] = None, decoder_fn=None
) -> list:
    """Reads messages from the rosbag file.
    Args:

        path: Path to the rosbag file.
        topic_name: Name of the topic to read messages from.
        take_n: Number of messages to take. All messages are taken if None.
    Returns:
        List of messages from the rosbag file.
    """

    bag = rosbag.Bag(path)
    info = bag.get_type_and_topic_info()
    print(info)
    msgs = []
    for topic, msg, _ in tqdm(
        bag.read_messages(topics=[topic_name]),
        desc="Reading messages",
        total=take_n or info.topics[topic_name].message_count,
    ):
        if topic == topic_name:
            if decoder_fn is not None:
                msg = decoder_fn(msg)
            msgs.append(msg)
            if take_n is not None:
                take_n -= 1
                if take_n == 0:
                    break
    bag.close()
    return msgs


def print_em_messages(msgs: list, layer_idx: int = 0, take_n: t.Optional[int] = None):
    """Displays images (elevation maps) stored in the messages.
    Args:
        msgs: List of messages to show.
        layer_idx: Index of the layer inside the message to use.
        take_n: Number of messages to take. All messages are taken if None.
    Returns:
        A figure with the images parsed from the messages.
    """
    assert len(msgs) > 0, "No messages to print"
    take_n = take_n or len(msgs)

    fig, axs = plt.subplots(1, take_n, figsize=(15, 5))
    for i, msg in enumerate(msgs[:take_n]):
        elevation_msg = msg.data[layer_idx]
        width = elevation_msg.layout.dim[0].size
        height = elevation_msg.layout.dim[1].size
        elevation_map_data = np.array(elevation_msg.data).reshape(width, height)
        if take_n > 1:
            ax = axs[i]
        else:
            ax = axs
        ax.imshow(elevation_map_data, cmap="viridis", origin="lower")
        ax.set_title("Elevation Map")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
    return fig


def parse_sample(s, voxel_size=0.1):
    assert s.shape[1] == 4, f"Expected 4 columns, got {s.shape[1]}"
    return downsample_point_cloud(s[:, :3], voxel_size), s[:, 3]


def parse_lidar_scan_path(p):
    return parse_sample(np.fromfile(p, dtype=np.float32).reshape(-1, 4))
