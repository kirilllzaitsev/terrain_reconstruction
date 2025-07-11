import typing as t

import torch
import torch.nn as nn
from terrain_representation.utils.utils import DeviceType


def get_norm_layer_params(output_features, shape, norm_layer_type):
    if norm_layer_type == nn.LayerNorm:
        enc_block_norm_layer_kwargs = {
            "normalized_shape": (output_features, shape, shape, shape)
        }
    elif norm_layer_type == nn.BatchNorm3d:
        enc_block_norm_layer_kwargs = {
            "num_features": output_features,
        }
    elif norm_layer_type == nn.InstanceNorm3d:
        enc_block_norm_layer_kwargs = {
            "num_features": output_features,
        }
    elif norm_layer_type == nn.GroupNorm:
        enc_block_norm_layer_kwargs = {
            "num_groups": 8,
            "num_channels": output_features,
        }
    else:
        raise ValueError(f"Norm layer {norm_layer_type} not supported")
    return enc_block_norm_layer_kwargs


class EncoderBlock(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        shape: int,
        norm_layer=nn.LayerNorm,
        dropout_rate=None,
    ):
        super().__init__()
        enc_block_norm_layer_kwargs = get_norm_layer_params(
            output_features, shape, norm_layer
        )
        self.enc_block = nn.Sequential(
            nn.Conv3d(
                input_features,
                output_features,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=1,
            ),
            nn.ELU(),
            nn.Identity() if dropout_rate is None else nn.Dropout3d(p=dropout_rate),
            norm_layer(**enc_block_norm_layer_kwargs),
        )

    def forward(self, input):
        return self.enc_block(input)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        shape: int,
        pad=1,
        encoder_features=None,
        norm_layer=nn.LayerNorm,
        dropout_rate=None,
    ):
        super().__init__()

        intermid_features = (
            output_features if encoder_features is None else encoder_features
        )
        dec_block_norm_layer_kwargs = get_norm_layer_params(
            intermid_features, shape, norm_layer
        )

        self.dec_block = nn.Sequential(
            nn.ConvTranspose3d(
                input_features,
                intermid_features,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=1,
                output_padding=pad,
            ),
            nn.ELU(),
            nn.Identity() if dropout_rate is None else nn.Dropout3d(p=dropout_rate),
            norm_layer(**dec_block_norm_layer_kwargs),
        )
        self.dec_block2 = None
        if encoder_features is not None:
            dec_block2_norm_layer_kwargs = get_norm_layer_params(
                output_features, shape, norm_layer
            )
            self.dec_block2 = nn.Sequential(
                nn.Conv3d(
                    intermid_features * 2,
                    output_features,
                    kernel_size=(3, 3, 3),
                    stride=1,
                    padding=1,
                ),
                nn.ELU(),
                norm_layer(**dec_block2_norm_layer_kwargs),
            )

    def forward(self, input, encoder_input=None):
        dec = self.dec_block(input)
        if self.dec_block2 is not None:
            dec = torch.cat((dec, encoder_input), dim=1)
            dec = self.dec_block2(dec)
        return dec


class TerrainNetLayerNorm(nn.Module):
    """Dense version of the terrain representation network. No intermediate pruning is performed and skip connections are optional (the premise is that the bottleneck is this way more representative for downstream RL tasks). The network is trained to predict the occupancy and the centroids of the voxels in the terrain."""

    def __init__(
        self,
        map_dim: torch.Tensor,
        map_resolution: torch.Tensor,
        device_handle: t.Optional[DeviceType] = None,
        use_skip_conn: bool = False,
        use_prev_pred_as_input: bool = False,
        norm_layer_name: str = "layer_norm",
        use_dropout: bool = False,
        dropout_rate: float = 0.3,
        hidden_dims_scaler: float = 1.0,
    ):
        super().__init__()

        self.map_dim = map_dim
        self.map_resolution = map_resolution

        if use_prev_pred_as_input:
            self.in_channels = 6
        else:
            self.in_channels = 3
        self.use_prev_pred_as_input = use_prev_pred_as_input
        self.feature_dims_enc = [
            int(x * hidden_dims_scaler) for x in [16, 32, 32, 64, 64]
        ]
        self.feature_dims_dec = self.feature_dims_enc[::-1]

        map_dim = self.map_dim[0].item()

        norm_layer = get_norm_layer(norm_layer_name)

        self.use_dropout = use_dropout
        self.enc_block_1 = EncoderBlock(
            self.in_channels,
            self.feature_dims_enc[0],
            int(map_dim / 2),
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )
        self.enc_block_2 = EncoderBlock(
            self.feature_dims_enc[0],
            self.feature_dims_enc[1],
            int(map_dim / 4),
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )
        self.enc_block_3 = EncoderBlock(
            self.feature_dims_enc[1],
            self.feature_dims_enc[2],
            int(map_dim / 8),
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )
        self.enc_block_4 = EncoderBlock(
            self.feature_dims_enc[2],
            self.feature_dims_enc[3],
            int(map_dim / 16),
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )
        self.enc_block_5 = EncoderBlock(
            self.feature_dims_enc[3],
            self.feature_dims_enc[4],
            int(map_dim / 32),
            norm_layer=(
                norm_layer if norm_layer_name != "instance_norm" else nn.LayerNorm
            ),
        )

        self.feature_dims_enc_backwards = (
            self.feature_dims_enc[:-1][::-1] + [self.in_channels]
            if use_skip_conn
            else [None] * len(self.feature_dims_enc)
        )
        self.use_skip_conn = use_skip_conn

        self.dec_block_8 = DecoderBlock(
            self.feature_dims_enc[-1],
            self.feature_dims_dec[0],
            int(map_dim / 16),
            encoder_features=self.feature_dims_enc_backwards[0],
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )

        dec_block_4_idx = 1
        self.dec_block_4 = DecoderBlock(
            self.feature_dims_dec[0],
            self.feature_dims_dec[1],
            int(map_dim / 8),
            encoder_features=self.feature_dims_enc_backwards[dec_block_4_idx],
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )
        self.dec_block_3 = DecoderBlock(
            self.feature_dims_dec[1],
            self.feature_dims_dec[2],
            int(map_dim / 4),
            encoder_features=self.feature_dims_enc_backwards[dec_block_4_idx + 1],
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )
        self.dec_block_2 = DecoderBlock(
            self.feature_dims_dec[2],
            self.feature_dims_dec[3],
            int(map_dim / 2),
            encoder_features=self.feature_dims_enc_backwards[dec_block_4_idx + 2],
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )
        self.dec_block_1 = DecoderBlock(
            self.feature_dims_dec[3],
            self.feature_dims_dec[4],
            int(map_dim),
            encoder_features=self.feature_dims_enc_backwards[dec_block_4_idx + 3],
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )

        self.pruning_block = nn.Sequential(
            nn.Conv3d(
                self.feature_dims_dec[4], 1, kernel_size=(3, 3, 3), stride=1, padding=1
            ),
        )

        self.centroid_block = nn.Sequential(
            nn.Conv3d(
                self.feature_dims_dec[4], 3, kernel_size=(3, 3, 3), stride=1, padding=1
            ),
            nn.Tanh(),
        )

        self.prev_pose = None

    def reset(self, batch_ids: list = ...) -> None:
        self.occupancy[batch_ids] = False

    def initialize(self, num_batches: int) -> None:
        """Sets the initial values for the warp kernel data, occupancy and previous pose."""

        device = next(self.parameters()).device

        self.occupancy = torch.zeros(
            (num_batches, *self.map_dim.int().tolist()), dtype=torch.bool, device=device
        )

    def forward(
        self, measured_voxel_grid: torch.Tensor, threshold: float
    ) -> dict:

        if self.use_prev_pred_as_input and measured_voxel_grid.shape[-1] != 6:
            self.prev_voxel_grid_transformed = torch.zeros_like(measured_voxel_grid)
            self.prev_voxel_grid_transformed.fill_(-1.0)
            assert (
                measured_voxel_grid.shape == self.prev_voxel_grid_transformed.shape
            ), f"{measured_voxel_grid.shape=} != {self.prev_voxel_grid_transformed.shape=}"

            self.input = torch.cat(
                (measured_voxel_grid, self.prev_voxel_grid_transformed), dim=-1
            )
        else:
            self.input = measured_voxel_grid.clone()

        self.input = self.input.unsqueeze(1).transpose(1, -1)[..., 0]

        enc_2 = self.enc_block_1(self.input)
        enc_4 = self.enc_block_2(enc_2)
        enc_8 = self.enc_block_3(enc_4)
        enc_16 = self.enc_block_4(enc_8)
        enc_32 = self.enc_block_5(enc_16)

        if self.use_skip_conn:
            dec_16 = self.dec_block_8(enc_32, enc_16)
            dec_8 = self.dec_block_4(dec_16, enc_8)
            dec_4 = self.dec_block_3(dec_8, enc_4)
            dec_2 = self.dec_block_2(dec_4, enc_2)
            dec_1 = self.dec_block_1(dec_2, self.input)
        else:
            dec_16 = self.dec_block_8(enc_32)
            dec_8 = self.dec_block_4(dec_16)
            dec_4 = self.dec_block_3(dec_8)
            dec_2 = self.dec_block_2(dec_4)
            dec_1 = self.dec_block_1(dec_2)

        occupancy_logits = self.pruning_block(dec_1)
        self.occupancy = (
            nn.Sigmoid()(occupancy_logits.clone().detach()).squeeze(dim=1) > threshold
        )

        centroids = 0.5 + 0.5 * self.centroid_block(dec_1)

        return {
            "occupancy_logits": occupancy_logits,
            "occupancy": self.occupancy,
            "centroids": centroids.unsqueeze(-1).transpose(-1, 1)[:, 0],
            "enc16": enc_16,
        }


class SensorClassifierNet(nn.Module):
    """Classifier network for the LiDAR sensor model (sensors have different sparsity profiles, hinting the completion net)."""

    def __init__(
        self,
        map_dim: torch.Tensor,
        map_resolution: torch.Tensor,
        n_classes: int,
        norm_layer_name: str = "layer_norm",
        use_dropout: bool = False,
        dropout_rate: float = 0.3,
        hidden_dims_scaler: float = 1.0,
        fc_dims: t.List[int] = [128],
    ):
        super().__init__()

        self.map_dim = map_dim
        self.map_resolution = map_resolution

        self.in_channels = 3  # 3D point cloud
        self.feature_dims_enc = [
            int(x * hidden_dims_scaler) for x in [16, 32, 32, 64, 64]
        ]
        self.feature_dims_dec = self.feature_dims_enc[::-1]

        map_dim = self.map_dim[0].item()

        norm_layer = get_norm_layer(norm_layer_name)

        self.use_dropout = use_dropout
        self.enc_block_1 = EncoderBlock(
            self.in_channels,
            self.feature_dims_enc[0],
            int(map_dim / 2),
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )
        self.enc_block_2 = EncoderBlock(
            self.feature_dims_enc[0],
            self.feature_dims_enc[1],
            int(map_dim / 4),
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )
        self.enc_block_3 = EncoderBlock(
            self.feature_dims_enc[1],
            self.feature_dims_enc[2],
            int(map_dim / 8),
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )
        self.enc_block_4 = EncoderBlock(
            self.feature_dims_enc[2],
            self.feature_dims_enc[3],
            int(map_dim / 16),
            norm_layer=norm_layer,
            dropout_rate=dropout_rate if use_dropout else None,
        )
        self.enc_block_5 = EncoderBlock(
            self.feature_dims_enc[3],
            self.feature_dims_enc[4],
            int(map_dim / 32),
            norm_layer=(
                norm_layer if norm_layer_name != "instance_norm" else nn.LayerNorm
            ),
        )
        fc_input_dim = self.feature_dims_enc[-1] * (int(map_dim / 32)) ** 3
        vg_flat_to_fc_layer = nn.Linear(fc_input_dim, fc_dims[0])
        fc_layers = [vg_flat_to_fc_layer]
        fc_prev_dim = fc_input_dim
        for fc_dim in fc_dims[:-1]:
            fc_layers.append(nn.Linear(fc_prev_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(
                nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity()
            )
            fc_prev_dim = fc_dim
        fc_layers.append(nn.Linear(fc_dims[-1], n_classes))
        self.head = nn.Sequential(*fc_layers)

    def forward(self, measured_voxel_grid: torch.Tensor) -> dict:

        self.input = measured_voxel_grid.unsqueeze(1).transpose(1, -1)[..., 0]

        enc_2 = self.enc_block_1(self.input)
        enc_4 = self.enc_block_2(enc_2)
        enc_8 = self.enc_block_3(enc_4)
        enc_16 = self.enc_block_4(enc_8)
        enc_32 = self.enc_block_5(enc_16)

        enc_flatten = enc_32.view(enc_32.size(0), -1)
        logits = self.head(enc_flatten)

        return {
            "logits": logits,
        }

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs) -> dict:
        return self.forward(*args, **kwargs)


def get_norm_layer(norm_layer_name):
    if norm_layer_name == "layer_norm":
        norm_layer = nn.LayerNorm
    elif norm_layer_name == "batch_norm":
        norm_layer = nn.BatchNorm3d
    elif norm_layer_name == "instance_norm":
        norm_layer = nn.InstanceNorm3d
    elif norm_layer_name == "group_norm":
        norm_layer = nn.GroupNorm
    else:
        raise ValueError(f"Norm layer {norm_layer_name} not supported")
    return norm_layer
