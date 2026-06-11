"""Step 5: JEPA model architectures (JEPA + Generative baseline)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalEncoder(nn.Module):
    """2D CNN applied to multi-frame input (simpler, more stable than 3D conv).

    Input canvas is 64x64 (up from 28x28). We use 3 conv+pool stages so the
    spatial map shrinks 64 → 32 → 16 → 8, giving a flat feature size of
    64 * 8 * 8 = 4096 before the FC head.
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 4 stacked frames as input channels, 64x64 canvas
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2)  # 64 → 32 → 16 → 8 (3 applications)

        # After 3 pooling steps on 64x64: 8x8 spatial map
        # 64 channels * 8 * 8 = 4096 features
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 4, 64, 64) — 4 stacked frames as channels

        Returns:
            s_x: (batch, embedding_dim)
        """
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 32, 32)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 16, 16)
        x = self.pool(F.relu(self.conv3(x)))  # (batch, 64, 8, 8)

        x = x.flatten(1)          # (batch, 4096)
        x = F.relu(self.fc1(x))   # (batch, 256)
        x = self.fc2(x)           # (batch, embedding_dim)

        return x


class SpatialEncoder(nn.Module):
    """2D CNN encoder for single-frame input (future frame y).

    Mirrors SpatioTemporalEncoder's depth but takes 1 input channel.
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Single-channel 64x64 input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2)  # 64 → 32 → 16 → 8

        # 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (batch, 1, 64, 64) — single future frame

        Returns:
            s_y: (batch, embedding_dim)
        """
        y = self.pool(F.relu(self.conv1(y)))  # (batch, 32, 32, 32)
        y = self.pool(F.relu(self.conv2(y)))  # (batch, 64, 16, 16)
        y = self.pool(F.relu(self.conv3(y)))  # (batch, 64, 8, 8)

        y = y.flatten(1)          # (batch, 4096)
        y = F.relu(self.fc1(y))   # (batch, 256)
        y = self.fc2(y)           # (batch, embedding_dim)

        return y


class Projector(nn.Module):
    """
    SSL projector head.
    Encoder embedding -> projector space.
    """

    def __init__(
        self,
        embedding_dim=256,
        projector_dim=512,
        output_dim=512,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.GELU(),

            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.GELU(),

            nn.Linear(projector_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):

    def __init__(
        self,
        input_dim=512,
        hidden_dim=1024,
        output_dim=512,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class JEPAModel(nn.Module):
    """Joint-Embedding Predictive Architecture."""
    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 1024, projector_dim: int = 512, image_size: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        #Online X-Enc
        self.encoder_x = SpatioTemporalEncoder(embedding_dim)

        #Online Y-Enc
        self.encoder_y = SpatialEncoder(embedding_dim)
        
        #EMA Target-Enc
        self.target_encoder = SpatialEncoder(embedding_dim)

        self.target_encoder.load_state_dict(
            self.encoder_y.state_dict()
        )
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.projector = Projector(
            embedding_dim=embedding_dim,
            projector_dim=projector_dim,
            output_dim=projector_dim*2,
        )
        self.predictor = Predictor(input_dim=projector_dim*2, hidden_dim=hidden_dim, output_dim=projector_dim*2)

        self.image_size = image_size

    @staticmethod
    def vicreg_loss(z1, z2, sim_w=25.0, var_w=25.0, cov_w=1.0):
        sim = F.mse_loss(z1, z2)
        eps = 1e-4
        std1 = torch.sqrt(z1.var(dim=0) + eps)
        std2 = torch.sqrt(z2.var(dim=0) + eps)
        var = (F.relu(1 - std1).mean() + F.relu(1 - std2).mean())
        z1c = z1 - z1.mean(0)
        z2c = z2 - z2.mean(0)
        n = z1.shape[0]
        cov1 = (z1c.T @ z1c) / max(n - 1, 1)
        cov2 = (z2c.T @ z2c) / max(n - 1, 1)
        cov = ((cov1 - torch.diag(torch.diag(cov1))) ** 2).sum() / z1.shape[1] + \
              ((cov2 - torch.diag(torch.diag(cov2))) ** 2).sum() / z2.shape[1]
        return sim_w * sim + var_w * var + cov_w * cov

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # online branch
        s_x = self.encoder_x(x)
        z_x = self.projector(s_x)
        z_pred = self.predictor(z_x)

        # target branch
        with torch.no_grad():
            s_y = self.target_encoder(y)
            z_target = self.projector(s_y)

        loss = self.vicreg_loss(
            z_pred,
            z_target,
        )

        return {
            "s_x": s_x,
            "s_y": s_y,
            "z_pred": z_pred,
            "z_target": z_target,
            "loss": loss,
        }

    @staticmethod
    @torch.no_grad()
    def update_ema(student, teacher, momentum=0.999):
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)

    def compute_metrics(
        self,
        s_x: torch.Tensor,
        s_y: torch.Tensor,
    ) -> dict[str, float]:
        """Compute diagnostic metrics over a full validation set."""
        # Effective rank of the joint embedding matrix
        s_stacked = torch.cat([s_x, s_y], dim=0)          # (2*N, embedding_dim)
        s_centered = s_stacked - s_stacked.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(s_centered, full_matrices=False)
        s_norm = S / S.sum()
        eff_rank = torch.exp(-torch.sum(s_norm * torch.log(s_norm + 1e-12))).item()

        return {
            "effective_rank": eff_rank,
        }


class GenerativeModel(nn.Module):
    """Pixel-space generative baseline: encode y → decode y.

    Updated for 64x64 canvas: 3 conv+pool stages bring spatial dims
    64 → 32 → 16 → 8, giving a bottleneck of 64 * 8 * 8 = 4096.
    The decoder mirrors this with 3 ConvTranspose2d upsampling steps.
    """

    def __init__(self, embedding_dim: int = 128, image_size: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.image_size = image_size

        # ── Encoder: (1, 64, 64) → (embedding_dim,) ────────────────────────
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2)  # applied 3× → 64→32→16→8

        # 64 * 8 * 8 = 4096
        self.fc_encode = nn.Linear(64 * 8 * 8, embedding_dim)

        # ── Decoder: (embedding_dim,) → (1, 64, 64) ────────────────────────
        self.fc_decode = nn.Linear(embedding_dim, 64 * 8 * 8)

        # 3 upsampling steps: 8 → 16 → 32 → 64
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y: (batch, 1, 64, 64)

        Returns:
            y_recon: (batch, 1, 64, 64)
            loss:    scalar MSE
        """
        # Encode
        h = self.pool(F.relu(self.conv1(y)))  # (batch, 32, 32, 32)
        h = self.pool(F.relu(self.conv2(h)))  # (batch, 64, 16, 16)
        h = self.pool(F.relu(self.conv3(h)))  # (batch, 64, 8, 8)
        h = h.flatten(1)                       # (batch, 4096)
        code = self.fc_encode(h)               # (batch, embedding_dim)

        # Decode
        h = self.fc_decode(code)                   # (batch, 4096)
        h = h.view(-1, 64, 8, 8)                    # (batch, 64, 8, 8)
        h = F.relu(self.deconv1(h))                # (batch, 64, 16, 16)
        h = F.relu(self.deconv2(h))                # (batch, 32, 32, 32)
        y_recon = torch.sigmoid(self.deconv3(h))   # (batch, 1, 64, 64)

        loss = F.mse_loss(y_recon, y)
        return y_recon, loss

    def compute_metrics(
        self,
        y: torch.Tensor,
        y_recon: torch.Tensor,
    ) -> dict[str, float]:
        return {"recon_error": F.mse_loss(y_recon, y).item()}