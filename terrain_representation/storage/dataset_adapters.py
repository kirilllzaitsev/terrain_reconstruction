import torch
from terrain_representation.storage.dataset_torch import PointCloudDataset
from terrain_representation.utils.pointcloud_utils import normalize_pcl


class PointCloudDatasetAdapterPointAttn:
    def __init__(
        self, ds, num_pts=2048, gt_pts_scaler=1, use_pc_norm=False, use_randperm=True
    ):
        self.ds = ds
        self.num_pts = num_pts
        self.gt_pts_scaler = gt_pts_scaler
        self.use_pc_norm = use_pc_norm
        self.use_randperm = use_randperm

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        sample = self.ds[index]
        pcl = sample["measured"][0]
        gt = sample["mesh"][0]

        mean = gt.mean(0)
        max_dist = torch.max(torch.sqrt(torch.sum(gt**2, 1)))
        if self.use_pc_norm:
            pcl = normalize_pcl(pcl, mean, max_dist)
            gt = normalize_pcl(gt, mean, max_dist)

        labels = sample.get("labels", None)
        if labels is not None:
            labels = labels[0]

        if pcl.shape[0] > self.num_pts:
            if self.use_randperm:
                idx = torch.randperm(pcl.shape[0])[: self.num_pts]
            else:
                idx = torch.linspace(
                    0, pcl.shape[0] - 1, self.num_pts, dtype=torch.long
                )
            pcl = pcl[idx, :]
            if labels is not None:
                labels = labels[idx]
        else:
            upsample_res = upsample_torch(
                pcl, self.num_pts, use_randperm=self.use_randperm
            )
            pcl = upsample_res["pcl"]
            if labels is not None:
                labels = labels[upsample_res["idxs"]]

        if gt.shape[0] > self.num_pts * self.gt_pts_scaler:
            if self.use_randperm:
                idx = torch.randperm(gt.shape[0])[: self.num_pts * self.gt_pts_scaler]
            else:
                idx = torch.linspace(
                    0,
                    gt.shape[0] - 1,
                    self.num_pts * self.gt_pts_scaler,
                    dtype=torch.long,
                )
            gt = gt[idx, :]
        else:
            gt = upsample_torch(
                gt, self.num_pts * self.gt_pts_scaler, use_randperm=self.use_randperm
            )["pcl"]

        adapted_sample = {
            "measured": pcl,
            "gt": gt,
            "mesh": gt,
            "mean": mean,
            "max_dist": max_dist,
            "labels": labels,
        }

        for k in ["is_synthetic", "added_noise"]:
            v = sample[k][0]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            adapted_sample[k] = v

        if "area_map" in sample:
            adapted_sample["area_map"] = sample["area_map"][0]

        return adapted_sample


def upsample_torch(ptcloud, n_points, use_randperm=True):
    curr = ptcloud.shape[0]
    need = n_points - curr

    all_idxs = torch.arange(curr)
    while curr <= need:
        ptcloud = torch.cat([ptcloud, ptcloud], dim=0)
        all_idxs = torch.cat([all_idxs, all_idxs])
        need -= curr
        curr *= 2

    if use_randperm:
        idxs = torch.randperm(need)
    else:
        idxs = torch.linspace(0, curr - 1, need, dtype=torch.long)

    ptcloud = torch.cat([ptcloud, ptcloud[idxs]])
    all_idxs = torch.cat([all_idxs, all_idxs[idxs]])

    return {
        "pcl": ptcloud,
        "idxs": all_idxs,
    }


if __name__ == "__main__":
    ds = PointCloudDataset(
        base_folder="../../construction_site/terrain_ds_v13/collected_data_tilted/OS0_128ch10hz512res_processed",
        map_dim=torch.tensor([64, 64, 64]),
        map_resolution=torch.tensor([0.3125, 0.3125, 0.3125]),
        use_sparse=True,
        seq_start=0,
        overfit_config={
            "num_terrains": 1,
            "num_traj_per_terrain": 1,
            "traj_idxs": [0],
            "scan_idxs_in_traj": [0],
        },
    )
    dataset = PointCloudDatasetAdapterPointAttn(ds)
