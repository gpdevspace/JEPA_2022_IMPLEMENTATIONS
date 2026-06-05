"""JEPA (Joint Embedding Predictive Architecture) model with pluggable loss functions."""

from networkx.generators import spectral_graph_forge
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple


class Encoder(nn.Module):
    """CNN encoder for CIFAR-10 images."""

    def __init__(self, encoder_dim: int = 1024):
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
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1024),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LatentEncoder(nn.Module):
    def __init__(self, encoder_dim: int = 1024,latent_dim: int = 16):
        super().__init__()
        self.mu_head = nn.Linear(encoder_dim, latent_dim)
        self.logvar_head = nn.Linear(encoder_dim, latent_dim)
    
    def forward(self, s_x: torch.Tensor) -> torch.Tensor:
        mean = self.mu_head(s_x)
        log_var = self.logvar_head(s_x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z_x = mean + eps * std
        return z_x


class Predictor(nn.Module):
    """MLP predictor that maps s_x to predicted s_y."""

    def __init__(self, embedding_dim: int = 1024, hidden_dim: int = 2048, latent_dim: int = 16):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim + latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, s_x: torch.Tensor, z_x: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([s_x, z_x], dim=-1))


class JEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture.

    Components:
    - encoder_x: Encodes input image x to representation s_x
    - encoder_y: Encodes target image y to representation s_y (shared weights with encoder_x)
    - predictor: Predicts s_y from s_x

    Energy: F(x, y) = ||s_y - predictor(s_x)||^2
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        predictor_hidden: int = 2048,
        shared_encoder: bool = True,
        latent_dim: int = 16,
    ):
        super().__init__()
        self.encoder_x = Encoder(encoder_dim)
        
        if shared_encoder:
            self.encoder_y = self.encoder_x  # Shared weights
        else:
            self.encoder_y = Encoder(encoder_dim)
        
        self.predictor = Predictor(encoder_dim, predictor_hidden)
        self.shared_encoder = shared_encoder
        self.latent_encoder = LatentEncoder(encoder_dim,latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, 32, 32)
            y: Target images (B, 3, 32, 32)

        Returns:
            s_x: Embedding of x (B, embedding_dim)
            s_y: Embedding of y (B, embedding_dim)
            s_y_pred: Predicted embedding of y from x (B, embedding_dim)
            z_x: Latent embedding of y (B, latent_dim)
        """
        s_x = self.encoder_x(x)
        s_y = self.encoder_y(y)
        z_x = self.latent_encoder(s_x)
        s_y_pred = self.predictor(s_x, z_x)
        return s_x, s_y, z_x, s_y_pred

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for a single image."""
        return self.encoder_x(x)


def mse_loss(
    s_y: torch.Tensor,
    s_y_pred: torch.Tensor,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """
    Simple MSE loss (will cause collapse).

    This is the naive approach that leads to representation collapse:
    the model learns to assign a constant vector to all inputs.
    """
    loss = F.mse_loss(s_y_pred, s_y)
    metrics = {
        "loss": float(loss.item()),
        "invariance": float(loss.item()),
        "variance_loss": 0.0,
        "covariance_loss": 0.0,
    }
    return loss, metrics


def vicreg_loss(
    s_y: torch.Tensor,
    s_y_pred: torch.Tensor,
    z_x: torch.Tensor,
    variance_target: float = 1.0,
    variance_lambda: float = 25.0,
    covariance_lambda: float = 1.0,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """
    VICReg loss to prevent collapse.

    VICReg (Variance-Invariance-Covariance Regularization):
    - Invariance: Make s_y_pred close to s_y
    - Variance: Keep each embedding dimension alive (prevent collapse to constant)
    - Covariance: Decorrelate dimensions (prevent collapse to low-rank)

    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
    for Self-Supervised Learning", ICCV 2022.
    """
    # Invariance term
    invariance_loss = F.mse_loss(s_y_pred, s_y)

    # Variance term: keep std of each dimension above threshold
    def variance_term(z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return torch.mean(torch.relu(variance_target - std))

    variance_loss = variance_term(s_y_pred) + variance_term(s_y)

    # Covariance term: decorrelate dimensions
    def covariance_term(z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / (z.shape[0] - 1)
        off_diag = cov - torch.diag_embed(torch.diagonal(cov))
        return torch.sum(off_diag ** 2) / z.shape[1]

    covariance_loss = covariance_term(s_y_pred) + covariance_term(s_y)
    latent_penalty = z_x.pow(2).mean()
    beta = 0.01

    # Combine losses
    loss = invariance_loss + variance_lambda * variance_loss + covariance_lambda * covariance_loss + beta * latent_penalty

    metrics = {
        "loss": float(loss.item()),
        "invariance": float(invariance_loss.item()),
        "variance_loss": float(variance_loss.item()),
        "covariance_loss": float(covariance_loss.item()),
        "latent_penalty": float(latent_penalty.item()),
    }
    return loss, metrics


def compute_embedding_statistics(
    embeddings: torch.Tensor,
) -> dict[str, float]:
    """
    Compute statistics to detect collapse.

    Returns:
        - mean_std: Average standard deviation across dimensions (low = collapse)
        - min_std: Minimum std across dimensions (near zero = dead dimension)
        - effective_rank: Effective rank of covariance matrix (low = collapse to low-rank)
        - max_abs_mean: Maximum absolute mean (high = bias)
    """
    # Per-dimension statistics
    std_per_dim = embeddings.std(dim=0, unbiased=False)
    mean_per_dim = embeddings.mean(dim=0)

    mean_std = float(std_per_dim.mean().item())
    min_std = float(std_per_dim.min().item())
    max_abs_mean = float(mean_per_dim.abs().max().item())

    # Effective rank from covariance
    embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (embeddings_centered.T @ embeddings_centered) / (embeddings.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    
    total = eigvals.sum()
    if total > 1e-8:
        p = eigvals / total
        effective_rank = float(torch.exp(-torch.sum(p * torch.log(p + 1e-12))).item())
    else:
        effective_rank = 0.0

    return {
        "mean_std": mean_std,
        "min_std": min_std,
        "effective_rank": effective_rank,
        "max_abs_mean": max_abs_mean,
    }


def get_loss_function(
    loss_type: Literal["mse", "vicreg"],
) -> callable:
    """Return loss function by name."""
    if loss_type == "mse":
        return mse_loss
    elif loss_type == "vicreg":
        return vicreg_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'mse' or 'vicreg'.")
