import torch
import torch.nn as nn
import torch.nn.functional as F


class JointEmbeddingModel(nn.Module):
    """Image encoder with a Barlow Twins projection head."""

    def __init__(
        self,
        encoder_dim: int = 512,
        projector_hidden: int = 1024,
        projector_out: int = 128,
    ):
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
            nn.Linear(128, encoder_dim),
        )

        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, projector_hidden),
            nn.BatchNorm1d(projector_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden, projector_hidden),
            nn.BatchNorm1d(projector_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden, projector_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.projector(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def normalize_features(z: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, unbiased=False, keepdim=True)
    return z / (std + eps)


def cross_correlation_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1 = normalize_features(z1)
    z2 = normalize_features(z2)
    return torch.matmul(z1.T, z2) / z1.shape[0]


def compute_covariance_eigenvalues(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    cov = z.T @ z / (z.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    return eigvals.clamp(min=0.0)


def effective_rank_from_eigenvalues(eigvals: torch.Tensor) -> torch.Tensor:
    total = eigvals.sum()
    if total <= 0:
        return torch.tensor(0.0, device=eigvals.device)
    p = eigvals / total
    return torch.exp(-torch.sum(p * torch.log(p + 1e-12)))


def barlow_twins_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lamb: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    c = cross_correlation_matrix(z1, z2)
    diag = torch.diagonal(c)
    off_diag = c - torch.diag_embed(diag)
    diag_loss = torch.mean((diag - 1) ** 2)
    off_diag_loss = torch.mean(off_diag ** 2)
    loss = diag_loss + lamb * off_diag_loss
    metrics = {
        "mean_diag": float(diag.mean().item()),
        "mean_abs_offdiag": float(off_diag.abs().mean().item()),
        "diag_loss": float(diag_loss.item()),
        "off_diag_loss": float(off_diag_loss.item()),
    }
    return loss, metrics


def variance_covariance_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    variance_target: float = 1.0,
    variance_lambda: float = 1.0,
    covariance_lambda: float = 0.005,
) -> torch.Tensor:
    invariance = F.mse_loss(z1, z2)

    def variance_term(z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-04)
        return torch.mean(torch.relu(variance_target - std))

    def covariance_term(z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / (z.shape[0] - 1)
        off_diag = cov - torch.diag_embed(torch.diagonal(cov))
        return torch.mean(off_diag ** 2)

    loss = invariance
    loss = loss + variance_lambda * (variance_term(z1) + variance_term(z2))
    loss = loss + covariance_lambda * (covariance_term(z1) + covariance_term(z2))
    return loss
