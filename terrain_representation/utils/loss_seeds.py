from collections import defaultdict

import torch
import torch.nn.functional as F
try:
    from pointr.models.Transformer_utils import knn_point
except ImportError:
    knn_point = None
    print("knn_point not available")
from terrain_representation.losses.metrics import (
    compute_chamfer_dist,
    import_chamfer_dist,
)

ChamferDistanceL1 = import_chamfer_dist(use_l1=True)
similar_dist_thresh_default = 0.02
unk_label_default = -1


def compute_centroids(embeddings, labels):
    # embeddings: [batch_size, num_points, emb_dim]
    # labels: [batch_size, num_points]
    unique_labels = torch.unique(labels)
    centroids = []
    embeddings_for_label = []
    for label in unique_labels:
        mask = labels == label
        embedding_for_label = embeddings.transpose(1, 2)[mask]

        centroid = embedding_for_label.mean(dim=0)
        embeddings_for_label.append(embedding_for_label)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids, embeddings_for_label


def centroid_based_loss(embeddings, labels, margin=1.0):
    centroids, embeddings_for_label = compute_centroids(embeddings, labels)

    loss = torch.tensor(0.0, device=embeddings.device)
    if len(embeddings_for_label) == 1:
        return loss

    for i, embedding_for_label in enumerate(embeddings_for_label):
        # Positive pair loss (intra-class) (don't have positive in this case)

        # Negative pairs loss (inter-class)
        negative_centroids = centroids[[j for j in range(len(centroids)) if j != i]]
        for negative_centroid in negative_centroids:
            negative_distance = F.pairwise_distance(
                embedding_for_label.unsqueeze(0), negative_centroid
            )
            loss += F.relu(margin - negative_distance).sum()

    return loss / len(embeddings)


def get_seed_discrepancy_loss(
    measured,
    seed_xyz,
    seed_feat,
    labels=None,
    k=10,
    margin=4.0,
    unsup_seed_subsample_ratio=0.25,
    similar_dist_thresh=similar_dist_thresh_default,
    use_full_triplet_loss=False,
    do_flatten_batch_dim=True,
):
    if labels is None:
        seed_idxs = torch.linspace(
            0,
            seed_xyz.shape[2] - 1,
            int(seed_xyz.shape[2] * unsup_seed_subsample_ratio),
        ).long()
        seed_xyz = seed_xyz[:, :, seed_idxs]
        seed_feat = seed_feat[:, :, seed_idxs]
        return get_seed_discrepancy_loss_unsup(
            measured,
            seed_xyz,
            seed_feat,
            margin=margin,
            k=k,
            similar_dist_thresh=similar_dist_thresh,
            unk_label=unk_label_default,
            use_full_triplet_loss=use_full_triplet_loss,
            do_flatten_batch_dim=do_flatten_batch_dim,
        )
    return get_seed_discrepancy_loss_sup(
        measured, seed_xyz, seed_feat, labels, k=k, margin=margin
    )


def get_seed_discrepancy_loss_sup(measured, seed_xyz, seed_feat, labels, k, margin):
    coor_k = measured
    coor_q = seed_xyz
    idx = knn_point(k, coor_k.contiguous(), coor_q.transpose(-1, -2).contiguous())
    # emb_dim = seed_feat.shape[1]
    num_pts = coor_q.shape[2]
    mode = torch.gather(labels.unsqueeze(1).expand(-1, num_pts, -1), 2, idx).mode(dim=2)
    seed_labels = mode.values
    loss = centroid_based_loss(seed_feat, seed_labels, margin=margin)
    return {
        "loss": loss,
    }


def get_seed_discrepancy_loss_unsup(
    measured,
    seed_xyz,
    seed_feat,
    margin=4.0,
    k=50,
    similar_dist_thresh=0.02,
    unk_label=-1,
    use_full_triplet_loss=False,
    do_flatten_batch_dim=True,
):
    seed_labels = get_seed_labels_batch(
        measured,
        seed_xyz,
        k=k,
        similar_dist_thresh=similar_dist_thresh,
        unk_label=unk_label,
    )
    if use_full_triplet_loss:
        loss = calc_full_seed_triplet_loss_batch(
            seed_labels,
            seed_feat,
            n_cluster_members_default=5,
            do_flatten_batch_dim=do_flatten_batch_dim,
            unk_label=unk_label,
            margin=margin,
        )
    else:
        loss = centroid_based_loss(seed_feat, seed_labels, margin=margin)
    return {
        "loss": loss,
    }


def get_seed_labels_batch(
    measured,
    seed_xyz,
    k=50,
    similar_dist_thresh=similar_dist_thresh_default,
    unk_label=-1,
):

    bs = measured.shape[0]
    labels = []
    for idx in range(bs):
        labels_sample = get_seed_labels(
            measured[idx],
            seed_xyz[idx],
            k=k,
            similar_dist_thresh=similar_dist_thresh,
            unk_label=unk_label,
        )
        labels.append(labels_sample)
    return torch.stack(labels)


def get_seed_labels(
    measured,
    seed_xyz,
    k=50,
    similar_dist_thresh=similar_dist_thresh_default,
    unk_label=-1,
):

    assert ChamferDistanceL1 is not None, "ChamferDistanceL1 is not available"

    if seed_xyz.shape[0] == 3:
        seed_xyz = seed_xyz.transpose(0, 1)

    all_neighbors = []
    for center in seed_xyz:
        neighbors_raw = get_neighbors(measured, center, k)
        neighbors = prep_neighbors(neighbors_raw, center)
        all_neighbors.append(neighbors)

    dist_matrix = torch.zeros((len(all_neighbors), len(all_neighbors)))

    for idx1, pts1 in enumerate(all_neighbors):
        for idx2, pts2 in enumerate(all_neighbors[idx1:], idx1):
            dist_matrix[idx1, idx2] = ChamferDistanceL1(pts1, pts2)

    # find ids of similar neighbors
    similar_neighbors = defaultdict(list)
    similar_neighbors_inv = {}
    for idx1, pts1 in enumerate(all_neighbors):
        for idx2, pts2 in enumerate(all_neighbors[idx1:], idx1):
            if idx1 == idx2:
                continue
            if dist_matrix[idx1, idx2] < similar_dist_thresh:
                root_idx = similar_neighbors_inv.get(
                    idx2, similar_neighbors_inv.get(idx1, idx1)
                )
                if idx2 not in similar_neighbors[root_idx]:
                    similar_neighbors[root_idx].append(idx2)
                similar_neighbors_inv[idx2] = root_idx

    unk_label = -1
    labels = torch.ones(len(all_neighbors)) * unk_label
    for idx, neighbor_idxs in similar_neighbors.items():
        if len(neighbor_idxs) > 0:
            for other_idx in neighbor_idxs:
                labels[other_idx] = idx
            labels[idx] = idx
    return labels.to(measured.device)


def get_neighbors(measured, center, k):
    sq_dist = torch.sum((measured - center) ** 2, dim=-1)
    _, idx = torch.topk(sq_dist, k=k, largest=False)
    return measured[idx]


def prep_neighbors(neighbors, center):
    neighbors_local_frame = neighbors - center
    if len(neighbors_local_frame.shape) == 2:
        neighbors_local_frame = neighbors_local_frame.unsqueeze(0)
    return neighbors_local_frame.float()


def calc_full_seed_triplet_loss_batch(
    seed_labels,
    seed_feat,
    n_cluster_members_default=5,
    do_flatten_batch_dim=False,
    unk_label=-1,
    margin=1.0,
):
    loss = torch.tensor(0.0, device=seed_feat.device)
    if do_flatten_batch_dim:
        b, n = seed_labels.shape
        seed_labels = seed_labels.view(b * n)
        seed_feat = seed_feat.view(b * n, -1)
        loss += calc_full_seed_triplet_loss(
            seed_labels,
            seed_feat,
            take_n_cluster_members_default=n_cluster_members_default,
            unk_label=unk_label,
            margin=margin,
        )

    else:
        for b_idx in range(seed_labels.shape[0]):
            seed_labels_b = seed_labels[b_idx]
            seed_feat_b = seed_feat[b_idx]
            loss += calc_full_seed_triplet_loss(
                seed_labels_b,
                seed_feat_b,
                take_n_cluster_members_default=n_cluster_members_default,
                unk_label=unk_label,
                margin=margin,
            )
    # if loss == 0:
    #     return torch.tensor(torch.nan, device=seed_feat.device)

    return loss


def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = F.pairwise_distance(anchor.unsqueeze(0), positive.unsqueeze(0))
    neg_dist = F.pairwise_distance(anchor.unsqueeze(0), negative.unsqueeze(0))
    return F.relu(pos_dist - neg_dist + margin).squeeze()


def calc_full_seed_triplet_loss(
    seed_labels,
    seed_feat,
    take_n_cluster_members_default=5,
    unk_label=-1,
    min_cluster_size=2,
    margin=1.0,
):

    grouped_seed_feat = {}
    for idx, label in enumerate(seed_labels.detach().cpu().numpy().astype(int)):
        if label == unk_label:
            continue
        if label not in grouped_seed_feat:
            grouped_seed_feat[label] = []
        grouped_seed_feat[label].append(seed_feat[idx])
    grouped_seed_feat = {
        k: v for k, v in grouped_seed_feat.items() if len(v) >= min_cluster_size
    }

    if len(grouped_seed_feat) == 0 or len(grouped_seed_feat) == 1:
        return torch.tensor(0.0, device=seed_feat.device)

    for k, v in grouped_seed_feat.items():
        grouped_seed_feat[k] = torch.stack(v)

    n_cluster_members = min(
        take_n_cluster_members_default, *map(len, grouped_seed_feat.values())
    )

    subset_list = [
        cluster[torch.randperm(cluster.size(0))[:n_cluster_members]]
        for cluster in grouped_seed_feat.values()
    ]
    k = len(subset_list)
    triplets = []
    for i in range(k):
        for j in range(n_cluster_members):
            for m in range(n_cluster_members):
                if j != m:
                    anchor = subset_list[i][j]
                    positive = subset_list[i][m]

                    # Choose a random negative from another cluster
                    neg_cluster_idx = torch.randint(0, k, (1,)).item()
                    while (
                        neg_cluster_idx == i
                    ):  # Ensure the negative is from a different cluster
                        neg_cluster_idx = torch.randint(0, k, (1,)).item()

                    negative = subset_list[neg_cluster_idx][
                        torch.randint(0, n_cluster_members, (1,)).item()
                    ]
                    triplets.append((anchor, positive, negative))

    loss = 0.0
    for anchor, positive, negative in triplets:
        loss += triplet_loss(anchor, positive, negative, margin=margin)

    loss /= len(triplets)  # Average the loss
    return loss
