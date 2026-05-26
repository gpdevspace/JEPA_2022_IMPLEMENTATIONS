import torch
import torch.nn as nn

from shared.modules import MLP


class EBM(nn.Module):
    """
    Energy-based model F_w(x, y) -> scalar incompatibility energy.
    Low energy = compatible pair; high energy = incompatible.
    """

    def __init__(self, x_dim: int = 1, y_dim: int = 2, embed_dim: int = 128):
        super().__init__()
        self.x_encoder = MLP([x_dim, 64, embed_dim])
        self.y_encoder = MLP([y_dim, 64, embed_dim])
        self.energy_head = MLP([2 * embed_dim, 128, 64, 1])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_enc = self.x_encoder(x)
        y_enc = self.y_encoder(y)
        return self.energy_head(torch.cat([x_enc, y_enc], dim=-1)).squeeze(-1)
