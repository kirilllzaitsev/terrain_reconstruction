# Completion of partial point cloud scans

The goal is to complete partial point clouds captured by a robot's LiDAR sensor to enable the robot to navigate in unstructured terrains.

This directory hosts two packages: `terrain_representation` and `terrain_synthesis`. The former contains the code to train and evaluate several neural networks for point cloud completion. The latter contains the code to synthesize terrains for training. Simulation code is in the `excavation_terrain_sim` directory and is meant to collected data from IsaacSim.

## Pre-requisites

### Environment setup

The code is developed and tested on Ubuntu 22.04 with Python 3.8.15. Simulations in IsaacSim are done via Docker based on the `nvcr.io/nvidia/isaac-sim:2023.1.0-hotfix.1` image.

*Note.* It is recommended to use Docker for interacting with IsaacSim and pick an image with the tag `2023.1.0-hotfix.1` or newer to reduce the likelihood of unexpected issues. The native setup of the simulator on Ubuntu 22.04 could not run the code in full due to errors in loading the robot model (if you use the robot and not only sensor in the simulation).

A possible CUDA setup:

```
CUDA_VERSION=11.7
CUDA_HOME=/usr/local/cuda-11.7
```

with the driver version `535.183.01`.

### Python packages

- `pip install -e .` installs the `terrain_representation` and `terrain_synthesis` packages with their dependencies
- `point_based_models` contains three packages needed for working with point-based models, originally taken from their respective Git repositories. Make sure to follow READMEs in the respective directories to install their dependencies as well
- Minkowski Engine should be installed separately via `pip install git+https://github.com/NVIDIA/MinkowskiEngine.git` or by cloning the repository and running `python setup.py install --blas=openblas` from its root directory
- extra requirements for running the code in ROS are included in the `requirements_extra_ros.txt` file

### Data and model artifacts

There are two archives (not publicly available) containing data and model weights:

- `construction_site.zip` with synthetic and real datasets for training and evaluation 
- `artifacts.zip` with the best-performing models

### Docker

To create images for training the dense 3D UNet and point-based models, use provided Dockerfiles:

- `Dockerfile` - for the dense 3D UNet
- `related_work.Dockerfile` - for the point-based models

## Terrain representation

The package `terrain_representation` contains the code to train a neural network to estimate the terrain around the robot from a noisy pointcloud. The neural network is trained using a dataset of pointclouds collected from IsaacSim and the corresponding standalone pointclouds that represent the terrain as is.

The package is structured as follows:

```
terrain_representation
├── train_dense_v2.py <--- a pipeline to train the dense 3D UNet
├── train_sparse_v2.py <--- a pipeline to train the sparse 3D UNet
├── train_point_based.py <--- a pipeline to train the point-based models
├── hyperparam_tuning.py <--- a pipeline to optimize the hyperparameters of the `train_dense_v2` pipeline
├── eval  <--- Code to get the quantitative and qualitative results from trained models
│   |── results <--- Folder to store the results
│   |── eval_utils.py <--- Metadata and utils for evaluation
│   └── data_real_test_full.py <--- evaluation pipeline
├── losses
│   ├── chamfer_distance <--- Contains the code to efficiently compute the chamfer distance. JIT-compiled
│   │   └── *.py
│   └── *.py
├── modules  <--- Contains the neural network modules
│   └── *.py
├── storage <--- Contains the code to manipulate the data
│   └── *.py
└── utils <--- Contains various utility modules
    └── *.py
```

### Models

The package contains the following models:

- `modules/dense_trep.py` - the dense 3D UNet
- `modules/sparse_trep.py` - the sparse 3D UNet (requires Minkowski Engine)

Point-based models have to installed separately from their respective folders under `./point_based_models`:

- `point_based_models/seedformer` - SeedFormer ([source](https://github.com/hrzhou2/seedformer))
- `point_based_models/PointFlow` - PointFlow ([source](https://github.com/stevenygd/PointFlow))
- `point_based_models/PoinTr` - SnowflakeNet, FoldingNet, PointAttn, AdaPoinTr, GRNet ([source](https://github.com/yuxumin/PoinTr))

### Training

#### Training pipelines

There are three training pipelines:

- `train_dense_v2.py` - trains the dense 3D UNet
- `train_sparse_v2.py` - trains the sparse 3D UNet (requires Minkowski Engine)
- `train_point_based.py` - trains the point-based models (SeedFormer, SnowflakeNet, AdaPoinTr, GRNet, PointFlow, PointAttn)

To run the training, start from the following command, tweaking it with the arguments specified in `utils/__init__.py`:

```bash
cd terrain_representation
python train_dense_v2.py --exp_disabled --log_every_n_epochs=5 --log_subdir=test --map_dimension=64 --map_resolution_xy=0.3125 --map_resolution_z=0.3125 --mini_batch_size=4 --num_epochs=300 --plots_disabled --plots_zoom=0.5 --pos_weight=5 --run_name=test --sequence_length=1 --step_skip=1 --use_dataloader --use_val_ds --use_mae_loss --vis_disabled --vis_freq_batch_per_epoch=1 --vis_freq_epoch=20 --traj_folder=./construction_site/terrain_ds_v13/collected_data_tilted/OS0_128ch10hz512res_processed_no_mask --use_early_stopping
```

To use `DistributedDataParallel` and a multi-gpu setup, use the `--use_ddp` flag.

Training with two GPUs takes about 15 hours for 150 epochs. If configured in `comet_utils.py`, the code will log training metrics and artifacts to CometML.

##### Grid resolution

Training voxel-based models, pick the map_dim and map_resolution parameters that could cover the entire sensor range:

```
- 64 cells: --map_dimension=64 --map_resolution_xy=0.3125 --map_resolution_z=0.3125
- 96 cells: --map_dimension=96 --map_resolution_xy=0.2083 --map_resolution_z=0.2083
- 128 cells: --map_dimension=128 --map_resolution_xy=0.1563 --map_resolution_z=0.1563
```

#### Training the models reported in the results section

Several training configurations for the best models are provided in the `configs` directory:

- `train_best_dense.yaml` - the best dense 3D UNet quantitatively. Predicting the centroids of the cells in the voxel grid leads to significant artifacts in subsequent `elevation_mapping`
- `train_best_dense_no_centroids.yaml` - the best dense 3D UNet without centroids. Used for `elevation_mapping` and visuals
- `train_best_sparse.yaml` - the best sparse 3D UNet
- `train_best_point_based.yaml` - the best point-based model (SeedFormer)
- `train_best_autoreg.yaml` - the best dense 3D UNet with an autoregressive input

Ensure that the paths to data in the configurations are correct:
- `traj_folder` should point to the directory with the postprocessed synthetic dataset
- specify `real_ds_bremgarten_folder`, `real_ds_hong2_folder`, `real_ds_hong_folder` to use respective real datasets

To train using these configurations, use the `--args_path` argument:

```bash
cd terrain_representation
python train_dense_v2.py --args_path=configs/train_best_dense.yaml
```

### Evaluation

`eval/data_real_test_full.py` contains the code to evaluate models on real and synthetic datasets. The script computes the quantitative results and saves them in the `results` directory. Qualitative results are saved in the directories that correspond to experiment names: `./artifacts/exp_name/preds` and can be visualized using `eval/vis_preds.ipynb`.

Before running the evaluation, you may need to set the `COMET_API_KEY` environment variable to be able to fetch models from CometML.

The archive `artifacts.zip` contains artifacts of the best-performing experiments. Unzip it to repository's root directory to use the models for evaluation.

To get results from the main ablation study, run the following command:

```bash
python eval/data_real_test_full.py --dl_slice_len 50 --exp_names_pack ablation
```

Specify the `--exp_names='exp_name1 exp_name2'` argument to evaluate specific experiments.

For experiments listed in `ablation_exp_names` from `eval/eval_utils.py`, the script will download the artifacts from CometML and save several tables with quantitative results in the `results` directory.

See `eval/data_real_test_full.py` arguments for more options.

### Hyperparameter tuning

`hyperparam_tuning.py` contains the code to optimize the hyperparameters of the model. The code uses the `optuna` library to optimize the hyperparameters. The DB backend in use is PostgreSQL. Steps to set it up:

- `sudo apt-get install postgresql`
- `pip install optuna psycopg2-binary`
- create a new user and a database in PostgreSQL and supply `--opt_storage` with the connection string in the format `postgresql://user:password@host:port/dbname`

Now you can run the training script with the `--do_optimize` flag to optimize the hyperparameters. See `optuna_args` param group in `utils.__init__.py` for available optimization flags.

## Deployment

### Integration with elevation mapping (for unstructured terrains)

You can use the model with the `elevation_mapping` ROS package either by integrating the inference code into ROS or completing the pointclouds offline and pointing the `elevation_mapping` to the directory with the completed scans.

To use the `elevation_mapping` package in docker, you might find useful the `ros-docker-gpu.Dockerfile` for a minimal container setup of ROS with GPU support. The `elevation_mapping` [directory](https://github.com/ANYbotics/elevation_mapping/blob/master/README.md) should be installed according to the official instructions on top of the created docker image.


