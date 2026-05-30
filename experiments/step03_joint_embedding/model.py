import torch
import torch.nn as nn
import torch.nn.functional as F


class JointEmbeddingModel(nn.Module):
    """Shared encoder for joint embedding training."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def energy_embeddings(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return torch.sum((z1 - z2) ** 2, dim=1)

    def energy(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self(x1)
        z2 = self(x2)
        return self.energy_embeddings(z1, z2)


def collapse_distance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Minimize squared distances for all cross-sample pairs, causing collapse."""
    diff = z1.unsqueeze(1) - z2.unsqueeze(0)
    distances = torch.sum(diff ** 2, dim=-1)
    return torch.mean(distances)


def normalize_embeddings(z: torch.Tensor, eps: float = 1e-08) -> torch.Tensor:
    norm = torch.norm(z, dim=1, keepdim=True).clamp(min=eps)
    return z / norm


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z1 = normalize_embeddings(z1)
    z2 = normalize_embeddings(z2)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a + loss_b)
