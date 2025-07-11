import typing as t
from collections import defaultdict

from terrain_representation.losses.metrics import import_chamfer_dist
from terrain_representation.utils.pointcloud_utils import sparse_tensor_to_point_cloud

try:
    import MinkowskiEngine as ME
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
from terrain_representation.utils.voxel_grid_utils import (
    align_vgs,
    voxel_grid_to_point_cloud,
)

loss_func_dense_cache = {}

ChamferLoss = import_chamfer_dist()


def loss_func_dense(
    data: dict,
    pos_weight: torch.Tensor,
    centroid_weight: float,
    use_mae_loss: bool = True,
    use_chamfer_loss: bool = False,
    use_occupancy_loss_with_input: bool = False,
    added_noise: t.Optional[torch.Tensor] = None,
) -> dict:
    """Computes the weighted sum of BCE and distance (MAE or Chamfer) losses.
    Args:
        data: dict containing the following
            occupancy_logits: tensor of shape (B, 1, H, W, D)
            centroids: tensor of shape (B, 3, N)
            target: tensor of shape (B, 3, H, W, D)
            target_mesh: tensor of shape (B, 3, H, W, D)
            map_dim: list of 3 ints
            map_resolution: list of 3 floats
        pos_weight: weight for positive class in BCE loss for occupancy.
        centroid_weight: weight for distance loss.
        occupancy_thresh: threshold for occupancy logits.
        use_mae_loss: whether to use MAE loss.
        use_chamfer_loss: whether to use Chamfer loss instead of MAE loss.
        use_occupancy_loss_with_input: whether to compute additional occupancy loss between predictions and inputs
        added_noise: whether noise was added to samples in the batch
    Returns:
            A dictionary containing the losses.
    """

    pred_occupancy_logits = data["occupancy_logits"]
    pred_occupancy = data["occupancy"]
    centroids = data["centroids"]
    target_gt = data["target"]
    target_mesh = data["target_mesh"]
    measured_vg = data["measured_vg"]

    pos_weight = (
        torch.ones(
            (pred_occupancy_logits.shape[0], 1, 1, 1, 1),
            device=pred_occupancy_logits.device,
        )
        * pos_weight
    )
    # add extra dims to pos_weight to match pred_occupancy_logits shape
    if added_noise is not None and added_noise.any():
        # pos_weight for samples with added noise cannot be higher than 1 (noise occupies cells, cannot penalize marking these cells as unoccupied)
        pos_weight[added_noise.squeeze() > 0] = 1.0

    occupancy_mask_in_mesh = target_mesh.unsqueeze(1)[..., 0] >= 0

    target_probs = torch.zeros_like(pred_occupancy_logits)
    target_probs[occupancy_mask_in_mesh] = 1.0

    weight = torch.ones_like(occupancy_mask_in_mesh).float()
    is_synthetic = data.get("is_synthetic")
    if is_synthetic is not None and not is_synthetic.all():
        # do not penalize cells that are missing in real data yet should be actually occupied
        real_samples_mask = (
            ~is_synthetic.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        ).to(weight.device)
        weight[real_samples_mask & ~occupancy_mask_in_mesh] = 0.0

    loss_occupancy_mesh = F.binary_cross_entropy_with_logits(
        pred_occupancy_logits,
        target_probs,
        pos_weight=pos_weight,
    )
    loss = loss_occupancy_mesh

    # compute occupancy btw occupancy_logits and measured_vg
    res = {
        "loss": loss,
        "loss_occupancy_mesh": loss_occupancy_mesh,
    }
    if use_occupancy_loss_with_input:
        occupancy_mask_in_vg = measured_vg.unsqueeze(1)[..., 0] >= 0
        measured_occupancy_vg = torch.zeros_like(pred_occupancy_logits)
        measured_occupancy_vg[occupancy_mask_in_vg] = 1.0
        loss_occupancy_measured = F.binary_cross_entropy_with_logits(
            pred_occupancy_logits[measured_occupancy_vg > 0.0],
            measured_occupancy_vg[measured_occupancy_vg > 0.0],
            pos_weight=pos_weight,
        )
        loss += loss_occupancy_measured
        res["loss_occupancy_measured"] = loss_occupancy_measured

    assert not (
        use_mae_loss and use_chamfer_loss
    ), "At most one of use_mae_loss and use_chamfer_loss must be True"
    if use_mae_loss or use_chamfer_loss:
        target2 = target_gt.clone()
        target2_occupancy = get_vg_occupancy_mask(target2)
        centroid_mask = target2_occupancy & pred_occupancy
        if use_mae_loss:
            loss_mae = torch.abs(centroids - target2)[centroid_mask].mean()
            loss += centroid_weight * loss_mae
        elif use_chamfer_loss:
            centroid_pts = voxel_grid_to_point_cloud(
                centroids,
                centroid_mask,
                data["map_dim"],
                data["map_resolution"],
            )
            target_pts = voxel_grid_to_point_cloud(
                target2,
                target2[..., 0] > 0.0,
                data["map_dim"],
                data["map_resolution"],
            )
            loss_chamfer = get_chamfer_loss(centroid_pts, target_pts)
            loss += centroid_weight * loss_chamfer
    else:
        # caching for small efficiency gain
        if "loss_mae_default" in loss_func_dense_cache:
            loss_mae = loss_func_dense_cache["loss_mae_default"].clone()
        else:
            loss_mae = torch.tensor(-1.0, device=pred_occupancy_logits.device)
            loss_func_dense_cache["loss_mae_default"] = loss_mae.clone()
        if "loss_chamfer_default" in loss_func_dense_cache:
            loss_chamfer = loss_func_dense_cache["loss_chamfer_default"].clone()
        else:
            loss_chamfer = torch.tensor(-1.0, device=pred_occupancy_logits.device)
            loss_func_dense_cache["loss_chamfer_default"] = loss_chamfer.clone()

    res["loss_centroid"] = loss_mae if use_mae_loss else loss_chamfer

    return res


def get_chamfer_loss(pred_pts, target_pts):
    assert ChamferLoss is not None, "Chamfer loss not available"
    num_pts = sum([pts.shape[0] for pts in pred_pts])
    loss_chamfer = torch.tensor(0.0, device=target_pts[0].device)
    if num_pts > 10_000:
        # print(f"{num_pts=} is too large to compute Chamfer loss")
        loss_chamfer = torch.tensor(torch.nan, device=target_pts[0].device)
    else:
        loss_chamfer_callable = ChamferLoss
        for pts_gt, pts_pred in zip(target_pts, pred_pts):
            if len(pts_gt) == 0:
                print("Skipping empty ground truth point cloud for Chamfer loss")
                continue
            if len(pts_pred) == 0:
                print("Skipping empty predicted point cloud for Chamfer loss")
                continue
            loss_chamfer += loss_chamfer_callable(
                pts_gt.unsqueeze(0), pts_pred.unsqueeze(0)
            )
        loss_chamfer /= len(pred_pts)
    return loss_chamfer


def get_joint_gt_pred_vg_mask(
    pred_logits: torch.Tensor,
    gt: torch.Tensor,
    occupancy_thresh: float,
) -> torch.Tensor:
    """Returns a mask for cells occupied in both predicted and ground truth voxel grids."""
    centroid_mask_gt = get_vg_occupancy_mask(gt)
    centroid_mask_pred = get_vg_mask_from_logits(pred_logits, occupancy_thresh)
    centroid_mask = centroid_mask_gt & centroid_mask_pred

    return centroid_mask


def get_vg_occupancy_mask(gt):
    return gt[..., 0] > 0.0


def get_vg_mask_from_logits(pred_logits, occupancy_thresh):
    return nn.Sigmoid()(pred_logits.clone().detach()).squeeze(dim=1) > occupancy_thresh


def loss_func_sparse(
    data: dict,
    pos_weight: float = 1.5,
    centroid_weight: float = 0.2,
    use_mae_loss: bool = True,
    use_chamfer_loss: bool = False,
    batch_size: int = None,
    map_resolution: t.Optional[t.Any] = None,
    target_pts: t.Optional[t.Any] = None,
    **kwargs,
) -> dict:
    """Similar to loss_func_dense, but for sparse data. Does not support Chamfer loss."""

    layer_outputs = data["layer_outputs"]
    layer_targets = data["layer_targets"]

    # compute cross entropy loss for voxel occupancy
    num_layers = len(layer_outputs) - 1
    loss_occupancy = 0.0
    loss_centroid = 0.0

    if not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(pos_weight, device=layer_targets[0].device)

    # pos_weight = torch.tensor([1.5], device=layer_targets[0].device)
    for layer_output, layer_target in zip(layer_outputs[:-1], layer_targets[:-1]):
        curr_loss = F.binary_cross_entropy_with_logits(
            layer_output.F[:, 0].squeeze(),
            layer_target.type(layer_output.F.dtype),
            pos_weight=pos_weight,
        )
        loss_occupancy += curr_loss / num_layers

        if use_mae_loss or use_chamfer_loss:
            assert map_resolution is not None, "map_resolution must be provided"
            assert batch_size is not None, "batch_size must be provided"
            assert target_pts is not None, "target_pts must be provided"
            layer_output_pts = [
                sparse_tensor_to_point_cloud(layer_output, idx, map_resolution)
                for idx in range(batch_size)
            ]
            for output_pts, true_pts in zip(layer_output_pts, target_pts):
                if len(output_pts) == 0 or len(true_pts) == 0:
                    continue
                if use_mae_loss:
                    raise NotImplementedError("MAE loss not supported for point-based data")
                elif use_chamfer_loss:
                    loss_centroid += get_chamfer_loss([output_pts], [true_pts])
        else:
            loss_centroid = torch.tensor(torch.nan, device=layer_output.device)

    loss = loss_occupancy + centroid_weight * loss_centroid

    return {
        "loss": loss,
        "loss_occupancy": loss_occupancy,
        "loss_centroid": loss_centroid,
    }


@torch.no_grad()
def compute_occupancy_metrics(
    occupancy_estimate: torch.Tensor, target: torch.Tensor
) -> dict:
    """Computes precision, recall and f1 score for a batch of occupancy estimates as voxel grids."""
    occupied_true = target[..., 0] >= 0
    num_positives = torch.sum(occupied_true)

    if torch.sum(occupancy_estimate) > 0:
        tp = torch.sum(target[occupancy_estimate][:, 0] >= 0)
    else:
        tp = 0

    precision = tp / torch.sum(occupancy_estimate)
    recall = tp / num_positives
    f1 = 2 * precision * recall / (precision + recall)
    stats = {"precision": precision, "recall": recall, "f1": f1}
    return stats


@torch.no_grad()
def _compute_occupancy_metrics_sparse(
    output_pcl: torch.Tensor, target_pcl: torch.Tensor, kernel_map: torch.Tensor
) -> dict:
    """Computes precision, recall and f1 score for a batch of occupancy estimates. See sparse_trep module for details on the kernel_map argument."""
    assert (
        len(target_pcl.shape) == len(output_pcl.shape) == 2
    ), "batched inputs not supported"
    assert len(kernel_map.shape) == 2, "kernel_map is a 2xN tensor"

    tp = kernel_map.size(1)
    fp = len(output_pcl) - tp
    fn = len(target_pcl) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (fn + tp + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    stats = {"precision": precision, "recall": recall, "f1": f1}
    return stats


@torch.no_grad()
def compute_occupancy_metrics_sparse(
    # output_sparse, target_sparse, kernel_map_out, batch_idxs=None
    output_sparse: "ME.SparseTensor",
    target_sparse: "ME.SparseTensor",
    kernel_map_out: torch.Tensor,
    batch_idxs: t.Optional[t.List[int]] = None,
) -> dict:
    batch_idxs = batch_idxs or list(range(len(kernel_map_out)))
    stats_samples = defaultdict(list)
    for batch_idx in batch_idxs:
        stats_sample = _compute_occupancy_metrics_sparse(
            output_sparse.coordinates_at(batch_idx),
            target_sparse.coordinates_at(batch_idx),
            kernel_map_out[batch_idx],
        )
        for key, value in stats_sample.items():
            stats_samples[key].append(value)
    stats = {
        key: torch.tensor(value).mean().item() for key, value in stats_samples.items()
    }

    return stats
