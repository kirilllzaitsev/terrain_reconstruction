import argparse
import os
import random
import typing as t

import numpy as np
import torch
from sympy import Eq, solve, symbols

TensorOrArr = t.Union[torch.Tensor, np.ndarray]
TensorOrArrOrList = t.Union[list, torch.Tensor, np.ndarray]
DeviceType = t.Union[str, torch.device]


def pick_library(x: TensorOrArr) -> t.Any:
    if isinstance(x, torch.Tensor):
        lib = torch
    else:
        lib = np
    return lib


def load_from_checkpoint(
    ckpt_path: str,
    network: t.Any,
    device: DeviceType,
    optimizer: t.Any = None,
    scheduler: t.Any = None,
) -> None:
    """Load states of the network, optimizer, and scheduler from the checkpoint."""
    state_dicts = torch.load(
        ckpt_path,
        map_location=device,
    )
    network.load_state_dict(state_dicts["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state_dicts["optimizer"])
        for g in optimizer.param_groups:
            g["lr"] = 0.0005
    if scheduler is not None:
        scheduler.load_state_dict(state_dicts["scheduler"])

    print(f"Loaded checkpoint from {ckpt_path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def update_args(
    args: argparse.Namespace, kv_map: t.Dict[str, float]
) -> argparse.Namespace:
    """Update existing keys in args with new values for these keys from kv_map."""
    for k, v in kv_map.items():
        assert hasattr(args, k)
        setattr(args, k, v)
    return args


def align_current_data_dict_to_prev_sparse(current_data_dict: dict) -> t.List[dict]:
    """Same as align_current_data_dict_to_prev_dense but for sparse data."""
    return align_current_data_dict_to_prev_dense(
        current_data_dict, convert_entries_to_tensor=False
    )


def align_current_data_dict_to_prev_dense(
    current_data_dict: dict, convert_entries_to_tensor: bool = True
) -> t.List[t.Dict[str, t.Any]]:
    """Helps to align the data dictionary obtained via __getitem__ of a dataset to the previous format."""
    # current: k: v, where v is a list of batch_size that contains data for the entire sequence of length seq_len
    # prev: seq_len X (k: v), where v is a list of batch_size that contains data for each ts in a traj for a given batch
    prev_data_dict = [
        {k: [] for k in current_data_dict.keys()}
        for _ in range(len(current_data_dict[list(current_data_dict.keys())[0]][0]))
    ]
    for k, v in current_data_dict.items():
        for batch_idx, batch_data in enumerate(v):
            for time_idx, time_data in enumerate(batch_data):

                prev_data_dict[time_idx][k].append(time_data)

        if k not in ["area_map"]:
            for time_idx, _ in enumerate(prev_data_dict):
                time_data = prev_data_dict[time_idx][k]
                if not isinstance(time_data, torch.Tensor):
                    if k == "is_synthetic":
                        time_data = torch.tensor(time_data).bool()
                    elif convert_entries_to_tensor and k not in ["labels"]:
                        time_data = (
                            torch.stack(time_data)
                            if len(time_data) > 0
                            else torch.tensor([])
                        ).float()
                prev_data_dict[time_idx][k] = time_data
    return prev_data_dict


def transfer_batch_to_device(batch: t.Union[dict, list], device):
    if isinstance(batch, dict):
        batch = _transfer_batch_to_device(batch, device)
    elif isinstance(batch, list):
        batch = [_transfer_batch_to_device(b, device) for b in batch]
    return batch


def _transfer_batch_to_device(batch: dict, device: DeviceType) -> dict:
    """Transfer the batch of data to the device. The data can be a dictionary, list, or a tensor."""
    for k, v in batch.items():
        if is_tensor(v):
            batch[k] = to(v, device)
        elif isinstance(v, list):
            if len(v) == 0:
                continue
            if is_tensor(v[0]):
                batch[k] = [to(x, device) for x in v]
            elif isinstance(v[0], list):
                batch[k] = [
                    [to(x, device) if is_tensor(x) else x for x in y] for y in v
                ]
    return batch


def to(x: torch.Tensor, device: DeviceType) -> torch.Tensor:
    return x.to(device, non_blocking=True)


def is_tensor(v):
    return isinstance(v, torch.Tensor)


def find_map_res(map_dim, max_range=20):
    # Define the variable
    x = symbols("x")

    # Define the equation
    # equation = Eq(x * 0.2, 20)
    equation = Eq(map_dim * x, max_range)

    # Solve the equation for x
    solution = solve(equation, x)[0]
    return round(float(solution), 4)
