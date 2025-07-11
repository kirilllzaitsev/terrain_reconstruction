import argparse

import pandas as pd
from terrain_representation.utils.comet_utils import create_tracking_exp
from terrain_representation.utils.pipeline_utils import (
    compute_dist_metrics_and_elevated_idxs,
)
from terrain_representation.utils.pointcloud_utils import sparse_tensor_to_point_cloud
from terrain_representation.utils.vis_utils import display_model_preds

ds_name_to_full_name = {
    "hong": "Hong 06.05.2022",
    "bremg": "Bremgarten 04.05.2024",
    "hong2": "Hong 30.07.2024",
    "train": "Synthetic",
}

exp_name_to_meaning = {
    # ablation
    "scattered_balustrade_4955": "baseline_small",
    "intelligent_hostel_3212": "baseline_medium",
    "shallow_flux_2695": "baseline_big",
    "copper_veneer_5946": "noise",
    "imaginative_cash_294": "chamfer_loss",
    "capable_wolf_6104": "mae_loss",
    "intense_macaw_1204": "mae_loss+noise+skip_conn+dropout",
    "full_anaconda_3634": "mae_loss+centroids",
    "naval_porpoise_3295": "mae_loss+centroids+noise",
    "improved_frog_3564": "mae_loss+centroids+noise+skip_conn",
    "silly_level_8760": "mae_loss+centroids+noise+skip_conn+dropout",
    "continental_lamprey_2158": "mae_loss+centroids+noise+skip_conn+dropout+mixed",
    "voluntary_beans_3961": "mae_loss+centroids+noise+skip_conn+dropout+mixed+pos_weight=30",
    # benchmark
    "valuable_roundel_6276": "seedformer_run3+mixed+noise",
    "harsh_monastery_6093": "seedformer_medium+noise",
    "shared_flux_6107": "seedformer_run5+noise",
    "united_cornice_888": "seedformer_small+noise",
    "grand_column_8877": "seedformer_medium+mixed+noise",
    "right_silverfish_5247": "pointattn",
    # "passing_atoll_4057": "pointattn_linspace",
    "valid_prune_4730": "grnet",
    "dusty_blackbird_3736": "grnet+gridding_loss",
    "only_sap_9327": "snowflakenet+mixed+noise",
    # "modest_barracks_7171": "adapointr",
    "planned_parallax_8928": "foldingnet+mixed+noise",
    # "renewed_chickadee_7053": "pflow_gen",
    "spotty_cafe_1819": "pflow_ae+mixed+noise",
    "thoughtless_convertible_8624": "voxels_vs_points",
    # autoreg
    "soft_atlas_2196": "autoreg+seq_len=2_mae+centroids+noise+skip_conn_run2+dropout",
    "preliminary_ketchup_606": "autoreg+prev_pred+seq_len=2_mae+centroids+noise+skip_conn+dropout",
    "balanced_aggregate_6917": "autoreg+prev_pred+seq_len=2_mae+centroids+noise+skip_conn+dropout",
    "disastrous_damper_2571": "autoreg+seq_len=4_mae+centroids+noise+skip_conn+dropout",
    "inner_canid_280": "autoreg+prev_pred+seq_len=4_mae+centroids+noise+skip_conn+dropout",
    # sparse
    "healthy_condor_2906": "sparse+chamfer_loss+noise+mixed",
    "medical_cattle_6018": "sparse+chamfer_loss+noise",
    # grid resolution
    "harsh_garage_4429": "map_dim=96+mae_loss+centroids+noise+skip_conn+dropout+mixed",
    "easy_hadron_7538": "map_dim=128+mae_loss+centroids+noise+skip_conn+dropout+mixed",
    "rear_piles_4892": "map_dim=96+mae_loss+centroids+noise+skip_conn+dropout",
    "soft_plain_5967": "map_dim=128+mae_loss+centroids+noise+skip_conn+dropout",
    "small_cabana_3976": "num_pts=2048+random_downsampling+mae_loss+centroids+noise+skip_conn+dropout+mixed",
    # seed loss
    "strict_marsh_5064": "seed_loss+sup+coef=0.001",
    "slimy_bevel_2824": "seed_loss+sup+coef=0.01",
    "casual_molding_2412": "seed_loss+sup+coef=0.1",
    "short_jaguar_8694": "seed_loss+sup+coef=1",
    "steady_rice_8009": "seed_loss+sup+coef=1",
    "relevant_cinema_5582": "seed_loss+sup+full+coef=1",
    "sophisticated_fuse_5423": "seed_loss+sup+full+flatten+coef=1",
    "breezy_pilaster_4517": "seed_loss+unsup+coef=0.1",
    # "literary_emu_3000": "seed_loss+unsup+coef=0.1",
    "bitter_barbel_6046": "seed_loss+unsup+coef=1",
    "visible_tick_8772": "seed_loss+unsup+full+coef=1",
    "agreeable_foundry_9791": "seed_loss+unsup+full+flatten+coef=1",
    # other
    "developed_fish_6677": "num_pts=2048+mae_loss+centroids+noise+skip_conn+dropout+mixed",
    "spotty_muskox_6177": "mixed+randomization+skip_conn",
    "": "",
}


def exp_to_formatted_name_fn(exp_name):
    if exp_name == "noise":
        return "Noise (N)"
    elif exp_name == "mae_loss":
        return "MAE Loss (MAE)"
    elif exp_name == "chamfer_loss":
        return "Chamfer Loss (CL)"
    elif exp_name == "mae_loss+centroids":
        return "MAE + Centroids (C)"
    elif exp_name == "seed_loss+sup+coef=0.01":
        return "Supervised (S) + Seed Loss Coefficient (C)=0.01"
    elif exp_name == "seed_loss+unsup+coef=0.1":
        return "Unsupervised (U) + C=0.1"
    elif exp_name == "seed_loss+sup+full+coef=1":
        return "S + Triplet Loss (T) + C=1"
    elif exp_name == "seed_loss+sup+full+flatten+coef=1":
        return "S + T + Flatten Batch (FL) + C=1"
    # elif "coef=0.001" in exp_name:
    #     return "Baseline SeedFormer Small"
    # elif exp_name.endswith("+dropout"):
    #     exp_name = exp_name.replace("+dropout", "+Dropout (D)")
    elif exp_name.endswith("+skip_conn"):
        exp_name = exp_name.replace("+skip_conn", "+Skip-connection (SC)")
    # elif exp_name.endswith("+mixed"):
    #     exp_name = exp_name.replace("+mixed", "+Mixed data (M)")
    exp_name = (
        exp_name.replace("mixed", "M")
        .replace("randomization", "R")
        .replace("_mae", "+mae")
        .replace("skip_conn", "SC")
        .replace("seed_loss+", "")
        .replace("unsup+", "U+")
        .replace("sup+", "S+")
        .replace("coef=", "C=")
        .replace("+flatten", "+FL")
        .replace("+full", "+F")
        .replace("voxels_vs_points", "Equivalent Voxel-based UNet")
        .replace("pflow_ae", "PointFlow (Auto-encoder)")
        .replace("num_pts=2048", "NUM=2048")
        .replace("gridding_loss", "GL")
        .replace("baseline", "Baseline")
        .replace("noise", "N")
        .replace("chamfer_loss", "CL")
        .replace("mae_loss", "MAE")
        .replace("mae", "MAE")
        .replace("centroids", "C")
        .replace("autoreg", "AR")
        .replace("seq_len=", "SL=")
        .replace("prev_pred", "PP")
        .replace("sparse", "S")
        .replace("map_dim", "MD")
        .replace("_run2", " ")
        .replace("dropout", "D")
        .replace("random_downsampling+", "")
        # .replace("extended_input", "concat at test")
    )
    exp_name = (
        exp_name.replace("seedformer", "SeedFormer")
        .replace("pointattn", "PointAttn")
        .replace("grnet", "GRNet")
        .replace("snowflakenet", "SnowflakeNet")
        .replace("adapointr", "AdaPointR")
        .replace("foldingnet", "FoldingNet")
    )
    exp_name = exp_name.replace("_", " ")
    exp_name = (
        exp_name.replace("small", "Small")
        .replace("medium", "Medium")
        .replace("big", "Big")
    )
    return exp_name


ablation_exp_names = [
    "scattered_balustrade_4955",
    "intelligent_hostel_3212",
    "shallow_flux_2695",
    "copper_veneer_5946",
    "imaginative_cash_294",
    "capable_wolf_6104",
    "full_anaconda_3634",
    "naval_porpoise_3295",
    "improved_frog_3564",
    "silly_level_8760",
    "continental_lamprey_2158",
    # "small_cabana_3976",
]
sparse_exp_names = [
    # "healthy_condor_2906",
    "medical_cattle_6018",
]
ar_exp_names = [
    "soft_atlas_2196",
    "balanced_aggregate_6917",
    # "preliminary_ketchup_606",
    # "disastrous_damper_2571",  # failed
    # "inner_canid_280",  # failed
]
benchmark_exp_names = [
    # "thoughtless_convertible_8624",
    # "valuable_roundel_6276",
    "united_cornice_888",
    "grand_column_8877",
    "harsh_monastery_6093",
    "right_silverfish_5247",
    # "valid_prune_4730",
    "dusty_blackbird_3736",
    "only_sap_9327",
    # "modest_barracks_7171",
    # "conscious_marmoset_6757",
    "planned_parallax_8928",
    "spotty_cafe_1819",
]
grid_resolution_exp_names = [
    "rear_piles_4892",
    "soft_plain_5967",
]
seed_loss_exp_names = [
    "strict_marsh_5064",
    "slimy_bevel_2824",
    "casual_molding_2412",
    "steady_rice_8009",
    "breezy_pilaster_4517",
    "bitter_barbel_6046",
    "visible_tick_8772",
    "agreeable_foundry_9791",
    "relevant_cinema_5582",
    "sophisticated_fuse_5423",
]
ds_alias_to_name = {
    "hong": "test_hong",
    "bremg": "test_bremgarten",
    "hong2": "test_hong2",
    "train": "train",
}
req_stats_keys = [
    "maes",
    "maes_height",
    "chamfer_dists",
    "recall",
    "precision",
]


def vis_voxel_based(
    map_dim,
    map_resolution,
    network,
    visualizer,
    threshold,
    step,
    occupancy_logits,
    data_dict,
    centroids,
    log_img_name,
):
    vis_enabled = visualizer is not None

    if vis_enabled:
        common_display_kwargs = dict(
            visualizer=visualizer,
            data_dict=data_dict,
            net_input=network.input.detach().clone(),
            map_dim=map_dim,
            map_resolution=map_resolution,
            threshold=threshold,
            step=step,
        )

        for take_idx in [0]:
            display_model_preds(
                preds={
                    "occupancy_logits": occupancy_logits,
                    "centroids": centroids,
                },
                take_idx=take_idx,
                log_img_name=log_img_name,
                **common_display_kwargs,
            )


def vis_sparse_voxel_based(
    map_resolution,
    visualizer,
    step,
    sparse_input,
    sparse_output,
    gt,
    log_img_name,
    mesh=None,
):
    vis_enabled = visualizer is not None

    if vis_enabled:
        for take_idx in [0]:
            measured = sparse_tensor_to_point_cloud(
                sparse_input,
                take_idx,
                map_resolution=map_resolution,
            )

            vis_data_dict = {
                "measurement": measured,
                "input": measured,
                "output": sparse_tensor_to_point_cloud(
                    sparse_output,
                    take_idx,
                    map_resolution=map_resolution,
                ),
                "gt": gt[take_idx],
                "log_img_name": log_img_name,
                "step": step,
            }
            if mesh is not None:
                vis_data_dict["mesh"] = mesh[take_idx]

            visualizer.display_point_clouds(vis_data_dict)


def fix_args(args, cli_args, construction_site_path):
    args.norm_layer_name = "layer_norm"
    # replace remote path with local path
    args.real_ds_hong_folder = cli_args.hong_ds_dir or None
    args.real_ds_bremgarten_folder = cli_args.bremg_ds_dir or None
    args.real_ds_hong2_folder = cli_args.hong2_ds_dir or None
    args.use_ddp = False
    args.apply_rotation_prob = 0.0
    args.apply_mask = 0.0
    args.apply_shift = 0.0

    args.traj_folder = args.traj_folder.replace(
        "/home/kirillz/neural_terrain_representation/data",
        construction_site_path,
    )
    if "collected_data_tilted_processed" in args.traj_folder:
        args.traj_folder = f"{construction_site_path}/terrain_ds_v13/collected_data_tilted/OS0_128ch10hz512res_processed_no_mask"
    upd_args_dict = {}
    for k, v in args.__dict__.items():
        if "standalone" in k:
            upd_args_dict[k] = v
    for k, v in upd_args_dict.items():
        setattr(args, k.replace("standalone", "mesh"), v)
    return args


def create_eval_exp(exp_disabled=True):

    ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    meta_args = argparse.Namespace(**{})

    exp = create_tracking_exp(
        meta_args,
        exp_disabled=exp_disabled,
    )
    tags_to_log = [f"eval_{ts}", "local"]
    exp.add_tags(tags_to_log)
    return exp


def compute_dist_metrics(args, pose, area_maps, pts_gts, pts_preds):
    for i in range(len(pts_gts)):
        if len(pts_preds[i]) == 0:
            print("Empty pts_pred. Cannot compute dist metrics")
            break
        if len(pts_gts[i]) == 0:
            print("Empty pts_gt. Cannot compute dist metrics")
            break

    elevation_info, dist_metrics = compute_dist_metrics_and_elevated_idxs(
        args=args,
        pts_gts=pts_gts,
        pts_preds=pts_preds,
        pose=pose,
        area_maps=area_maps,
    )

    return dist_metrics
