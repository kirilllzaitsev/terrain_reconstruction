import copy
import typing as t

import MinkowskiEngine as ME
import torch
import torch.nn as nn
from terrain_representation.utils.pointcloud_utils import (
    point_clouds_to_sparse_tensor,
    sparse_tensor_to_point_cloud,
    transform_point_cloud,
)
from torch import Tensor


class EncoderBlock(nn.Module):
    def __init__(self, input_features: int, output_features: int, reduce_dim=False):
        super().__init__()
        kernel_size = [3, 3, 3, 2]
        if reduce_dim:
            stride = [2, 2, 2, 2]
        else:
            stride = [2, 2, 2, 1]

        self.enc_block = nn.Sequential(
            ME.MinkowskiConvolution(
                input_features,
                output_features,
                kernel_size=kernel_size,
                stride=stride,
                dimension=4,
            ),
            ME.MinkowskiInstanceNorm(output_features),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                output_features,
                output_features,
                kernel_size=kernel_size,
                dimension=4,
            ),
            ME.MinkowskiInstanceNorm(output_features),
            ME.MinkowskiELU(),
        )

    def forward(self, input):
        return self.enc_block(input)


class DecoderBlock(nn.Module):
    def __init__(self, input_features: int, output_features: int, alpha: float = 0.0):
        super().__init__()

        self.dec_block_1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                input_features,
                output_features,
                kernel_size=[3, 3, 3, 1],
                stride=[2, 2, 2, 1],
                dimension=4,
            ),
            ME.MinkowskiInstanceNorm(output_features),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                output_features,
                output_features,
                kernel_size=[3, 3, 3, 1],
                dimension=4,
            ),
            ME.MinkowskiInstanceNorm(output_features),
            ME.MinkowskiELU(),
        )

        self.dec_block_2 = nn.Sequential(
            ME.MinkowskiConvolution(
                output_features,
                output_features,
                kernel_size=2,
                stride=[1, 1, 1, 2],
                dimension=4,
            ),
            ME.MinkowskiInstanceNorm(output_features),
            ME.MinkowskiELU(),
        )

        self.dec_pruning = ME.MinkowskiConvolution(
            output_features, 1, kernel_size=1, stride=1, bias=True, dimension=4
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

        self.alpha = alpha

        self.pruning_target = None
        self.pruning_output = None

    def get_target(self, out, target_key):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key,
                out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key, strided_target_key, kernel_size=1
            )
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target

    def forward(self, input, encoder_input, target_key=None, keep_targets=True):
        x = self.dec_block_1(input)
        x = self.dec_block_2(x + encoder_input)
        pruning = self.dec_pruning(x)

        if target_key is not None:
            target = self.get_target(x, target_key)
            self.pruning_target = target
            self.pruning_output = pruning
        keep = (pruning.F > self.alpha).squeeze()
        if self.training and keep_targets:
            keep += target
        x = self.pruning(x, keep)
        return x


class TerrainCompletionNetSparse(nn.Module):
    """Model that closely matches the architecture suggested in 'Neural Scene Representation for Locomotion on Structured Terrain' paper."""

    def __init__(self, map_dim: Tensor, map_resolution: Tensor, alpha: float = 0.0):
        super().__init__()

        self.map_dim = map_dim
        self.map_resolution = map_resolution

        self.alpha = alpha

        self.feature_dims_enc = [8, 16, 32, 64, 128, 256]
        self.feature_dims_dec = [256, 128, 64, 32, 16, 8]

        self.input_block = nn.Sequential(
            ME.MinkowskiConvolution(
                3,
                self.feature_dims_enc[0],
                kernel_size=[3, 3, 3, 2],
                stride=1,
                dimension=4,
            ),
            ME.MinkowskiInstanceNorm(self.feature_dims_enc[0]),
            ME.MinkowskiELU(),
        )

        # Encoder
        self.enc_block_1 = EncoderBlock(
            self.feature_dims_enc[0], self.feature_dims_enc[1]
        )
        self.enc_block_2 = EncoderBlock(
            self.feature_dims_enc[1], self.feature_dims_enc[2]
        )
        self.enc_block_3 = EncoderBlock(
            self.feature_dims_enc[2], self.feature_dims_enc[3]
        )
        self.enc_block_4 = EncoderBlock(
            self.feature_dims_enc[3], self.feature_dims_enc[4], True
        )
        self.enc_block_42 = EncoderBlock(
            self.feature_dims_enc[4], self.feature_dims_enc[5], True
        )

        # Decoder
        self.dec_block_42 = DecoderBlock(
            self.feature_dims_dec[0], self.feature_dims_dec[1], 0
        )
        start_idx_dec = 1
        self.dec_block_4 = DecoderBlock(
            self.feature_dims_dec[start_idx_dec],
            self.feature_dims_dec[start_idx_dec + 1],
            0,
        )
        self.dec_block_3 = DecoderBlock(
            self.feature_dims_dec[start_idx_dec + 1],
            self.feature_dims_dec[start_idx_dec + 2],
            0,
        )
        self.dec_block_2 = DecoderBlock(
            self.feature_dims_dec[start_idx_dec + 2],
            self.feature_dims_dec[start_idx_dec + 3],
            0,
        )
        self.dec_block_1 = DecoderBlock(
            self.feature_dims_dec[start_idx_dec + 3],
            self.feature_dims_dec[start_idx_dec + 4],
            self.alpha,
        )

        self.centroid_block = nn.Sequential(
            ME.MinkowskiConvolution(
                self.feature_dims_dec[start_idx_dec + 4],
                3,
                kernel_size=[3, 3, 3, 1],
                stride=1,
                bias=True,
                dimension=4,
            ),
            ME.MinkowskiSigmoid(),
        )

        self.input = None
        self.output = None

    def to(self, device):
        self.map_resolution = self.map_resolution.to(device)
        return super().to(device)

    def clear_memory(self):
        self.output = None

    def get_target_deltas(self, out, target_key, target_feats, kernel_size=1):
        with torch.no_grad():
            target_deltas = torch.zeros((len(out), 3), device=out.device) + 0.5
            cm = out.coordinate_manager
            strided_target_key = cm.stride(target_key, out.tensor_stride[0])
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=1,
            )
            for k, curr_in in kernel_map.items():
                target_deltas[curr_in[0].long()] = target_feats[curr_in[1].long()]
        return target_deltas

    def transform(
        self,
        sparse_tensor: ME.SparseTensor,
        prev_pose: Tensor,
        pose: Tensor,
        coordinate_manager: ME.CoordinateManager,
    ) -> ME.SparseTensor:
        device = sparse_tensor.C.device
        transformed_points = []
        for idx in range(len(sparse_tensor.decomposed_coordinates)):
            points = sparse_tensor_to_point_cloud(
                sparse_tensor, idx, self.map_resolution
            ).detach()
            # do the transform in double precision for better result
            delta_pose = torch.matmul(
                torch.linalg.inv(pose[idx].double()), prev_pose[idx].double()
            ).to(device)
            transformed = transform_point_cloud(points.double(), delta_pose).float()
            mask = (
                torch.abs(transformed) < (self.map_resolution * self.map_dim / 2)
            ).all(dim=1)
            transformed = transformed[mask]
            transformed_points.append(transformed)
        return point_clouds_to_sparse_tensor(
            self.map_resolution, transformed_points, 1, coordinate_manager
        )
    
    def reset(self):
        pass

    def forward(self, measured_point_clouds, target_point_clouds=None):
        """
        Returns:
            output: ME.SparseTensor
            kernel_map: batched list of torch.Tensor, a 2d tensor where in_out_tensor[0] is the input row indices that correspond to in_out_tensor[1], which is the row indices for output
            target_sparse: ME.SparseTensor (only if target_point_clouds is not None)
        """

        if isinstance(measured_point_clouds, ME.SparseTensor):
            self.input = measured_point_clouds
        else:
            self.input = point_clouds_to_sparse_tensor(
                self.map_resolution, measured_point_clouds, 0
            )
        cm = self.input.coordinate_manager

        if target_point_clouds is not None:
            batched_target = point_clouds_to_sparse_tensor(
                self.map_resolution, target_point_clouds, 0
            )
            target_key, _ = cm.insert_and_map(batched_target.C, string_id="target")
        else:
            target_key = None

        self.output = self._forward(self.input, target_key)

        result = {}
        if target_point_clouds is not None:
            target_deltas = self.get_target_deltas(
                self.output, target_key, batched_target.F
            )

            strided_target_key = cm.stride(target_key, self.output.tensor_stride[0])

            kernel_map = cm.kernel_map(
                strided_target_key,
                self.output.coordinate_map_key,
                kernel_size=1,
            )

            kernel_map_out = []
            batch_indices = batched_target.C[:, 0]
            batch_size = len(torch.unique(batch_indices))

            if len(kernel_map) > 0:
                kernel_map = kernel_map[0]

                # Iterate over each unique batch index
                for batch_index in range(batch_size):
                    # Find indices in the kernel_map where input points belong to the current batch
                    input_indices_in_batch = (
                        batch_indices[kernel_map[0, :]] == batch_index
                    )
                    filtered_kernel_map = kernel_map[:, input_indices_in_batch]

                    # Store the filtered kernel_map for the current batch
                    kernel_map_out.append(filtered_kernel_map)
            else:
                kernel_map_out = [
                    torch.zeros((2, 0), device=self.output.device)
                ] * batch_size

            layer_outputs, layer_targets = self.get_outputs_and_targets()

            layer_targets.append(target_deltas)
            layer_outputs.append(self.output.F)

            result["layer_outputs"] = layer_outputs
            result["layer_targets"] = layer_targets
            result["kernel_map_out"] = kernel_map_out
            result["target_sparse"] = batched_target

        result["output"] = self.output
        return result

    def _forward(self, input, target_key=None):
        enc_1 = self.input_block(input)
        enc_2 = self.enc_block_1(enc_1)
        enc_4 = self.enc_block_2(enc_2)
        enc_8 = self.enc_block_3(enc_4)
        enc_16 = self.enc_block_4(enc_8)
        enc_162 = self.enc_block_42(enc_16)

        dec_82 = self.dec_block_42(enc_162, enc_16, target_key, True)
        dec_8 = self.dec_block_4(dec_82, enc_8, target_key, True)
        dec_4 = self.dec_block_3(dec_8, enc_4, target_key, True)
        dec_2 = self.dec_block_2(dec_4, enc_2, target_key, True)
        dec_1 = self.dec_block_1(dec_2, enc_1, target_key, False)

        output = self.centroid_block(dec_1)

        return output

    def get_outputs_and_targets(self):
        layer_outputs, layer_targets = [], []

        layer_outputs.append(self.dec_block_4.pruning_output)
        layer_outputs.append(self.dec_block_3.pruning_output)
        layer_outputs.append(self.dec_block_2.pruning_output)
        layer_outputs.append(self.dec_block_1.pruning_output)

        layer_targets.append(self.dec_block_4.pruning_target)
        layer_targets.append(self.dec_block_3.pruning_target)
        layer_targets.append(self.dec_block_2.pruning_target)
        layer_targets.append(self.dec_block_1.pruning_target)

        return layer_outputs, layer_targets
