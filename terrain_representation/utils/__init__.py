import argparse
import sys

import numpy as np


def parse_trep_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    pipe_args = parser.add_argument_group("pipeline")
    pipe_args.add_argument(
        "--do_not_calc_dist_metrics",
        action="store_true",
        help="Do NOT calculate distance metrics (e.g., MAE, Chamfer distance).",
    )
    pipe_args.add_argument(
        "--do_optimize", action="store_true", help="Optimize hyperparameters."
    )
    pipe_args.add_argument(
        "--do_overfit", action="store_true", help="Overfit a portion of the data."
    )
    pipe_args.add_argument(
        "--do_benchmark", action="store_true", help="Run a performance benchmark."
    )
    pipe_args.add_argument(
        "--exp_disabled", action="store_true", help="Do not log to Comet."
    )
    pipe_args.add_argument(
        "--vis_disabled", action="store_true", help="Do not use visualizer."
    )
    pipe_args.add_argument(
        "--plots_disabled",
        action="store_true",
        help="Do not display plots (may still log them).",
    )
    pipe_args.add_argument(
        "--plots_interactive", action="store_true", help="Allow interactive plots."
    )
    pipe_args.add_argument(
        "--use_val_ds", action="store_true", help="Use validation dataset."
    )
    pipe_args.add_argument("--use_mae_loss", action="store_true", help="Use MAE loss.")
    pipe_args.add_argument(
        "--use_occupancy_loss_with_input",
        action="store_true",
        help="Apply additional occupancy loss between input and prediction.",
    )
    pipe_args.add_argument(
        "--start_mae_from_n_epoch",
        type=int,
        default=0,
        help="Start using MAE loss from this epoch.",
    )
    pipe_args.add_argument(
        "--start_dist_metrics_from_n_epoch",
        type=int,
        default=0,
        help="Start calculating distance metrics from this epoch.",
    )
    pipe_args.add_argument(
        "--exp_tags", nargs="*", default=[], help="Tags for the experiment to log."
    )
    pipe_args.add_argument(
        "--log_subdir", default="logs", help="Subdirectory for logs."
    )
    pipe_args.add_argument("--device", default="cuda:0", help="Device to use.")
    pipe_args.add_argument(
        "--run_name",
        help="Name of the tb log folder. Overridden with experiment name if an experiment is used.",
    )
    pipe_args.add_argument(
        "--plots_zoom", default=1.0, type=float, help="Zoom of the plots."
    )
    pipe_args.add_argument("--seed", type=int, default=500, help="Random seed.")
    pipe_args.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=500,
        help="Save model every n epochs.",
    )
    pipe_args.add_argument(
        "--log_every_n_epochs", type=int, default=5, help="Log metrics every n epochs."
    )
    pipe_args.add_argument(
        "--vis_freq_batch_per_epoch",
        type=int,
        default=1,
        help="Visualize every n batches.",
    )
    pipe_args.add_argument(
        "--vis_freq_epoch", type=int, default=1, help="Visualize every n epochs."
    )
    pipe_args.add_argument(
        "--test_freq_epoch", type=int, default=100, help="Test every n epochs."
    )
    pipe_args.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to a previously trained session relative to the data/models folder.",
    )
    pipe_args.add_argument(
        "--args_path",
        type=str,
        help="Path to YAML with args to load. Overrides all other args except for those explicitly ignored.",
    )

    pipe_args.add_argument(
        "--ignored_file_args",
        nargs="*",
        default=[],
        help="List of args to ignore when loading from file.",
    )

    pipe_args.add_argument(
        "--use_cylindrical_coords",
        action="store_true",
        help="Use cylindrical coordinates instead of Cartesian.",
    )
    pipe_args.add_argument(
        "--do_save_model",
        action="store_true",
        help="Force saving the model according to schedule.",
    )
    pipe_args.add_argument(
        "--do_noisify", action="store_true", help="Apply noise to input."
    )
    pipe_args.add_argument(
        "--do_vis_entire_batch",
        action="store_true",
        help="Visualize all trajectories in a batch.",
    )
    pipe_args.add_argument(
        "--use_dataloader", action="store_true", help="Use dataloader."
    )
    pipe_args.add_argument(
        "--use_async_vis", action="store_true", help="Use async visualizer."
    )
    pipe_args.add_argument(
        "--use_ddp", action="store_true", help="Use DistributedDataParallel."
    )
    pipe_args.add_argument(
        "--hp_group_num",
        choices=[
            "v8_64_opt",
            "v8_64_mesh_opt",
            "v5_64_opt",
            "v9_64_opt",
            "v10_64_opt",
            "v12_64_opt",
            "v13_64_opt",
        ],
        help="Group of optimized hyperparameters to use (legacy).",
    )
    pipe_args.add_argument(
        "--hp_group_ignored",
        choices=[
            "centroid_weight",
            "occupancy_threshold",
            "pos_weight",
            "start_mae_from_n_epoch",
            "lr",
            "lr_decay",
        ],
        nargs="*",
        default=[],
        help="List of hyperparameters to ignore when loading from the group.",
    )
    pipe_args.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    pipe_args.add_argument(
        "--stdout_log_file",
        help="name of the file containing the stdout logs",
    )
    pipe_args.add_argument(
        "--model_name",
        default="unet",
        choices=[
            "unet",
            "pointattn",
            "seedformer",
            "adapointr",
            "grnet",
            "snowflakenet",
            "foldingnet",
            "pflow",
        ],
        help="Name of the model to use.",
    )

    data_args = parser.add_argument_group("data")
    data_args.add_argument(
        "--traj_folder",
        required=True,
        help="Absolute path to the folder with terrain_n/traj_m data.",
    )
    data_args.add_argument(
        "--real_ds_hong_folder",
        required=False,
        help="Absolute path to the folder containing the Hong dataset.",
    )
    data_args.add_argument(
        "--real_ds_bremgarten_folder",
        required=False,
        help="Absolute path to the folder containing the Bremgarten dataset.",
    )
    data_args.add_argument(
        "--real_ds_hong2_folder",
        required=False,
        help="Absolute path to the folder containing the Hong2 dataset.",
    )
    data_args.add_argument("--mini_batch_size", default=16, type=int)
    data_args.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="Number of workers for the dataloader. Multiple workers may cause a slowdown.",
    )
    data_args.add_argument(
        "--do_add_noise_to_real_data",
        action="store_true",
        help="Add noise to real data.",
    )
    data_args.add_argument(
        "--use_sparse", action="store_true", help="Must be set for the sparse pipeline."
    )
    data_args.add_argument(
        "--sequence_length",
        default=1,
        type=int,
        help="Length of the scan sequences. One means a random 140-degree subscan from a full 360-scan is used.",
    )
    data_args.add_argument(
        "--step_skip", default=1, type=int, help="Spacing of the steps in a sequence."
    )
    data_args.add_argument(
        "--map_dimension",
        default=32,
        type=int,
        help="Number of voxels in each dimension.",
    )
    data_args.add_argument(
        "--map_resolution_xy",
        default=0.125,
        type=float,
        help="Resolution of the map in the xy plane.",
    )
    data_args.add_argument(
        "--map_resolution_z",
        default=0.125,
        type=float,
        help="Resolution of the map in the z axis.",
    )
    data_args.add_argument(
        "--flat_region_z_lower_limit",
        default=-0.5,
        type=float,
        help="Lower limit of the height for what is considered a flat region (legacy).",
    )
    data_args.add_argument(
        "--flat_region_z_upper_limit",
        default=0.5,
        type=float,
        help="Upper limit of the height for what is considered a flat region (legacy).",
    )
    data_args.add_argument(
        "--mesh_cell_resolution",
        default=0.1,
        type=float,
        help="Resolution of cells in the point cloud obtained directly from synthesized terrains (legacy).",
    )
    data_args.add_argument("--use_pc_norm", action="store_true")
    data_args.add_argument(
        "--disable_randperm",
        action="store_true",
        help="Do not use random permutations when adjusting the size of the point cloud for point-based methods.",
    )
    data_args.add_argument(
        "--num_pts",
        type=int,
        help="Number of points to downsample the point cloud (legacy; 2048 points is used for point-based models).",
    )
    data_args.add_argument(
        "--gt_pts_scaler",
        type=int,
        default=1,
        help="Scaler for the number of points in the GT for point-based methods.",
    )

    training_args = parser.add_argument_group("training")
    training_args.add_argument(
        "--occupancy_threshold",
        type=float,
        default=0.9,
        help="Threshold for voxel occupancy probability.",
    )
    training_args.add_argument(
        "--centroid_weight",
        type=float,
        default=0.2,
        help="Weight of the MAE loss",
    )
    training_args.add_argument(
        "--pos_weight",
        type=float,
        default=5.0,
        help="Weight for positive occupancy in cross entropy loss. Set >1 to prioritize recall.",
    )
    training_args.add_argument("--num_epochs", type=int, default=300)

    # configs for overfitting/debugging. see dataset_torch.py for more details
    training_args.add_argument("--overfit_num_terrains", type=int, default=1)
    training_args.add_argument("--overfit_num_traj_per_terrain", type=int, default=1)
    training_args.add_argument("--overfit_num_scans_in_traj", type=int, default=1)
    training_args.add_argument("--overfit_traj_idxs", type=int, nargs="+", default=None)
    training_args.add_argument("--overfit_scan_idxs", type=int, nargs="+", default=None)

    training_args.add_argument("--lr", default=1e-3, type=float)
    training_args.add_argument("--lr_decay", type=float, default=0.995)
    training_args.add_argument(
        "--use_real_ds_in_train",
        action="store_true",
        help="Mix real data with synthetic data in training.",
    )
    training_args.add_argument(
        "--use_chamfer_loss",
        action="store_true",
        help="Use Chamfer loss instead of MAE loss.",
    )
    training_args.add_argument(
        "--use_centroids_for_dist_loss",
        action="store_true",
        help="Sets xy coordinates to 0.5 (voxel center) in gt. A better name is use_xy_centered_gt",
    )
    training_args.add_argument(
        "--use_early_stopping", action="store_true", help="Use early stopping."
    )
    training_args.add_argument("--early_stop_n_epochs", type=int, default=20)
    training_args.add_argument("--early_stop_delta", type=float, default=0.002)
    training_args.add_argument(
        "--do_rescale_lr",
        action="store_true",
        help="Rescale lr based on the number of GPUs/batch size.",
    )
    training_args.add_argument(
        "--use_seed_loss", action="store_true", help="Use seed loss."
    )
    training_args.add_argument(
        "--use_seed_loss_unsup", action="store_true", help="Use unsupervised seed loss."
    )
    training_args.add_argument(
        "--sl_use_full_triplet_loss",
        action="store_true",
        help="Calculate triplet loss instead of contrastive loss for seeds.",
    )
    training_args.add_argument(
        "--sl_do_flatten_batch_dim",
        action="store_true",
        help="Flatten all samples in a batch for the seed loss.",
    )
    training_args.add_argument(
        "--unsup_seed_subsample_ratio",
        type=float,
        default=0.25,
        help="How many seeds to sample for unsupervised seed loss.",
    )
    training_args.add_argument(
        "--similar_dist_thresh",
        type=float,
        default=0.02,
        help="Threshold for seed neighborhood similarity.",
    )
    training_args.add_argument(
        "--triplet_loss_margin",
        type=float,
        default=4.0,
        help="Margin for triplet loss.",
    )
    training_args.add_argument(
        "--seed_loss_coef", type=float, default=1, help="Weight for the seed loss."
    )

    model_args = parser.add_argument_group("model")
    model_args.add_argument(
        "--use_skip_conn", action="store_true", help="Use skip connections for UNet."
    )
    model_args.add_argument("--use_dropout", action="store_true")
    model_args.add_argument(
        "--use_large_model", action="store_true", help="Scale up the model size 2x."
    )
    model_args.add_argument("--dropout_rate", default=0.3, type=float)
    model_args.add_argument(
        "--hidden_dims_scaler",
        default=1.0,
        type=float,
        help="Multiply the hidden dims of the dense UNet by this value.",
    )
    model_args.add_argument(
        "--use_extended_input", action="store_true", help="Use autoregressive input."
    )
    model_args.add_argument(
        "--use_prev_pred_as_input",
        action="store_true",
        help="Use previous prediction as autoregressive input.",
    )
    model_args.add_argument(
        "--norm_layer_name",
        default="layer_norm",
        choices=["layer_norm", "batch_norm", "instance_norm", "group_norm"],
    )
    model_args.add_argument(
        "--pflow_use_gen",
        action="store_true",
        help="Use generator variant of the PFlow model.",
    )

    optuna_args = parser.add_argument_group("opt")
    optuna_args.add_argument("--opt_n_epochs", type=int, default=10)
    optuna_args.add_argument("--opt_n_trials", type=int, default=7)
    optuna_args.add_argument("--opt_n_jobs", type=int, default=2)
    optuna_args.add_argument("--opt_study_prefix", default="opt")
    optuna_args.add_argument(
        "--opt_storage",
        default="postgresql://optuna:optuna@localhost:5432/optuna_studies",
    )
    optuna_args.add_argument(
        "--opt_target_metrics", default="f1", choices=["f1", "mae", "mixed"]
    )
    optuna_args.add_argument(
        "--opt_params_to_tune",
        nargs="*",
        default=[],
        choices=[
            "centroid_weight",
            "occupancy_threshold",
            "pos_weight",
            "start_mae_from_n_epoch",
            "lr",
            "lr_decay",
        ],
    )

    # see noisifier.py for more details
    noise_args = parser.add_argument_group("noise")
    noise_args.add_argument("--noise_prob", default=1.0, type=float)
    noise_args.add_argument("--noise_num_blob_points", default=10, type=int)
    noise_args.add_argument("--noise_noise_scale", default=0.01, type=float)
    noise_args.add_argument("--noise_num_blobs", default=10, type=int)
    noise_args.add_argument("--noise_blob_extent", default=0.15, type=float)
    noise_args.add_argument("--noise_subsampling_ratio_min", default=1.0, type=float)
    noise_args.add_argument("--noise_subsampling_ratio_max", default=1.0, type=float)
    noise_args.add_argument("--noise_apply_rotation_prob", default=0.0, type=float)
    noise_args.add_argument("--noise_apply_shift_prob", default=0.0, type=float)
    noise_args.add_argument("--noise_add_boxes_prob", default=0.0, type=float)
    noise_args.add_argument("--noise_add_blobs_prob", default=0.0, type=float)
    noise_args.add_argument("--noise_apply_mask_prob", default=0.0, type=float)

    args = parser.parse_args()
    postprocess_args(args)

    return args


def postprocess_args(args):

    args.use_randperm = not args.disable_randperm

    if args.args_path:
        import yaml

        with open(args.args_path, "r") as f:
            loaded_args = yaml.safe_load(f)

        default_ignored_file_args = [
            "device",
            "run_name",
            "log_subdir",
            "exp_tags",
            "use_ddp",
            "use_async_vis",
            "exp_disabled",
            "mini_batch_size",
            "test_freq_epoch",
            "vis_freq_epoch",
        ]
        ignored_file_args = set(args.ignored_file_args) | set(default_ignored_file_args)
        for k, v in loaded_args.items():
            if k in ignored_file_args:
                print(f"Ignoring overriding {k}")
                continue
            setattr(args, k, v)

    args.vis_enabled = not args.vis_disabled
    args.do_calc_dist_metrics = not args.do_not_calc_dist_metrics

    # args.use_sparse = True if "sparse" in sys.argv[0] else False
    args.use_dataloader = True if "_v2" in sys.argv[0] else args.use_dataloader

    if args.hp_group_num:
        print(f"Overriding hyperparameters with group: {args.hp_group_num}")
        param_groups = {
            "v8_64_opt": {
                "centroid_weight": 0.1,
                "occupancy_threshold": 0.65,
                "pos_weight": 7,
                "start_mae_from_n_epoch": 150,
            },
            "v9_64_opt": {
                "centroid_weight": 0.25,
                "occupancy_threshold": 0.6,
                "pos_weight": 4,
                "start_mae_from_n_epoch": 150,
                "lr": 0.0008,
                "lr_decay": 0.999,
            },
            "v10_64_opt": {
                "centroid_weight": 0.45,
                "occupancy_threshold": 0.6,
                "pos_weight": 5.5,
                "start_mae_from_n_epoch": 100,
                "lr": 0.0008,
                "lr_decay": 0.999,
            },
            "v12_64_opt": {
                "centroid_weight": 0.35,
                "occupancy_threshold": 0.6,
                "pos_weight": 7.5,
                "start_mae_from_n_epoch": 0,
                "lr": 0.001,
                "lr_decay": 0.998,
            },
            "v13_64_opt": {
                "centroid_weight": 0.05,
                "occupancy_threshold": 0.55,
                "pos_weight": 8,
                "start_mae_from_n_epoch": 50,
                "lr": 0.0005,
                "lr_decay": 0.998,
            },
            "v8_64_mesh_opt": {
                "centroid_weight": 0.1,
                "occupancy_threshold": 0.6,
                "pos_weight": 6,
                "start_mae_from_n_epoch": 170,
            },
            "v5_64_opt": {
                "centroid_weight": 0.2,
                "occupancy_threshold": 0.55,
                "pos_weight": 6.8,
                "start_mae_from_n_epoch": 0,
            },
        }
        for k, v in param_groups[args.hp_group_num].items():
            if k in args.hp_group_ignored:
                print(f"Ignoring overriding {k}")
                continue
            setattr(args, k, v)
            print(f"Overriding {k} with {v}")

    if (
        args.do_overfit
        and args.overfit_traj_idxs is not None
        and len(args.overfit_traj_idxs) > 0
    ):
        assert len(args.overfit_traj_idxs) > 0
        if len(args.overfit_traj_idxs) != args.mini_batch_size:
            raise RuntimeError("overfit_traj_idxs and mini_batch_size do not match. ")
