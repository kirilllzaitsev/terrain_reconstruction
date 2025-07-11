import torch
import torch.nn as nn


class TerrainNetRefiner(nn.Module):
    """Refines the initial terrain reconstruction by another network."""

    def __init__(self, map_dim: torch.Tensor, map_resolution: torch.Tensor):
        super().__init__()

        self.map_dim = map_dim
        self.map_resolution = map_resolution

        map_dim = self.map_dim[0].item()

        self.up_sample = nn.Sequential(
            nn.ConvTranspose3d(
                3,
                16,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=1,
                output_padding=1,
            ),
            nn.ELU(),
            nn.LayerNorm((16, 32, 32, 32)),
        )

        self.enc1 = nn.Sequential(
            nn.Conv3d(19, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.ELU(),
            nn.LayerNorm((32, 16, 16, 16)),
        )

        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.ELU(),
            nn.LayerNorm((32, 8, 8, 8)),
        )

        self.dec_2 = nn.Sequential(
            nn.ConvTranspose3d(
                32,
                32,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=1,
                output_padding=1,
            ),
            nn.ELU(),
            nn.LayerNorm((32, 16, 16, 16)),
        )

        self.dec_3 = nn.Sequential(
            nn.ConvTranspose3d(
                32,
                16,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=1,
                output_padding=1,
            ),
            nn.ELU(),
            nn.LayerNorm((16, 32, 32, 32)),
        )

        self.pruning_block = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=(3, 3, 3), stride=1, padding=1),
        )

        self.centroid_block = nn.Sequential(
            nn.Conv3d(16, 3, kernel_size=(3, 3, 3), stride=1, padding=1), nn.Tanh()
        )

    def forward(self, measured_voxel_grid, decoded_voxel_grid, threshold):
        decoded_voxel_grid[decoded_voxel_grid[..., 0] < 0.0] = 0.0
        measured_voxel_grid[measured_voxel_grid[..., 0] < 0.0] = 0.0

        up_sampled = self.up_sample(
            decoded_voxel_grid.unsqueeze(1).transpose(1, -1)[..., 0]
        )

        self.input = torch.cat(
            (measured_voxel_grid.unsqueeze(1).transpose(1, -1)[..., 0], up_sampled),
            dim=1,
        )

        enc_1 = self.enc1(self.input)
        enc_2 = self.enc2(enc_1)
        dec_2 = self.dec_2(enc_2)
        dec_3 = self.dec_3(dec_2)

        occupancy_logits = self.pruning_block(dec_3)
        self.occupancy = (
            nn.Sigmoid()(occupancy_logits.clone().detach())[:, 0] > threshold
        )
        centroids = 0.5 + 0.5 * self.centroid_block(dec_3)
        return (
            occupancy_logits,
            self.occupancy,
            centroids.unsqueeze(-1).transpose(-1, 1)[:, 0],
        )
