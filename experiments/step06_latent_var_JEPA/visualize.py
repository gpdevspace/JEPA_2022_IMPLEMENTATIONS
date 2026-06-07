"""Step 5: JEPA visualization and diagnostics."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step06_latent_var_JEPA.model import JEPAModel, GenerativeModel
from shared.data import MovingMNISTDataset
from shared.device import get_device

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Must match the canvas size used during training
IMAGE_SIZE = 64


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_jepa_checkpoint(
    model: JEPAModel,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded JEPA checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}")


def load_generative_checkpoint(
    model: GenerativeModel,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded Generative checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — Loss curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_curves() -> None:
    """Training/validation loss curves for both models on the same figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    jepa_hist_path = OUTPUT_DIR / "loss_history_jepa.json"
    if jepa_hist_path.exists():
        with open(jepa_hist_path) as f:
            jepa_hist = json.load(f)
        axes[0].plot(jepa_hist["train_loss"], label="Train", linewidth=2)
        axes[0].plot(jepa_hist["val_loss"],   label="Val",   linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("MSE Loss", fontsize=12)
        axes[0].set_title("JEPA: Embedding-Space Prediction Loss", fontsize=13, fontweight="bold")
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

    gen_hist_path = OUTPUT_DIR / "loss_history_generative.json"
    if gen_hist_path.exists():
        with open(gen_hist_path) as f:
            gen_hist = json.load(f)
        axes[1].plot(gen_hist["train_loss"], label="Train", linewidth=2)
        axes[1].plot(gen_hist["val_loss"],   label="Val",   linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("MSE Loss", fontsize=12)
        axes[1].set_title("Generative: Pixel-Space Reconstruction Loss", fontsize=13, fontweight="bold")
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "loss_curves.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — JEPA effective rank over training
# ──────────────────────────────────────────────────────────────────────────────

def plot_effective_rank_jepa() -> None:
    """Effective rank of the JEPA embedding matrix per epoch.

    A rank that collapses toward 1 means representation collapse —
    the encoder maps everything to the same point. We want it to stay high.
    """
    jepa_hist_path = OUTPUT_DIR / "loss_history_jepa.json"
    if not jepa_hist_path.exists():
        print("Warning: JEPA history not found, skipping effective rank plot")
        return

    with open(jepa_hist_path) as f:
        jepa_hist = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(jepa_hist["val_eff_rank"], marker="o", linewidth=2, markersize=7, color="royalblue")
    ax.axhline(y=128, color="red",    linestyle="--", linewidth=2, label="Max rank (dim=128)")
    ax.axhline(y=1,   color="orange", linestyle="--", linewidth=2, label="Collapse (rank=1)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Effective Rank", fontsize=12)
    ax.set_title("JEPA: Embedding Effective Rank (collapse detector)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    _save(fig, "effective_rank_jepa.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3 — Temporal input strip  [NEW]
# ──────────────────────────────────────────────────────────────────────────────

def plot_temporal_strips(
    val_loader: DataLoader,
    n_sequences: int = 6,
) -> None:
    """Show 4 input frames + ground-truth future frame for several sequences.

    This lets you visually confirm that digits are actually moving between
    frames (the whole point of fixing the dataset bug).

    Layout: one row per sequence, 5 columns (t-3, t-2, t-1, t, t+1).
    """
    x_batch, y_batch = next(iter(val_loader))
    # squeeze spurious channel dim added by __getitem__
    x_batch = x_batch.squeeze(1)   # (B, 4, H, W)
    y_batch = y_batch.squeeze(1)   # (B, 1, H, W)

    n = min(n_sequences, x_batch.shape[0])
    fig, axes = plt.subplots(n, 5, figsize=(12, 2.4 * n))

    col_titles = ["t − 3", "t − 2", "t − 1", "t", "t + 1 (target)"]

    for row in range(n):
        for col in range(4):
            axes[row, col].imshow(x_batch[row, col].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=10, fontweight="bold")

        axes[row, 4].imshow(y_batch[row, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 4].axis("off")
        if row == 0:
            axes[row, 4].set_title(col_titles[4], fontsize=10, fontweight="bold", color="darkgreen")

        axes[row, 0].set_ylabel(f"seq {row+1}", fontsize=9, rotation=0, labelpad=30, va="center")

    plt.suptitle("Moving MNIST: Temporal Input Strips (digits should move between frames)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "temporal_strips.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4 — t-SNE of JEPA embedding space
# ──────────────────────────────────────────────────────────────────────────────

def extract_embeddings_jepa(
    model: JEPAModel,
    val_loader: DataLoader,
    device: torch.device,
    n_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect s_x, s_y, ŝ_y from the validation set."""
    model.eval()
    all_s_x, all_s_y, all_s_y_pred = [], [], []
    collected = 0

    with torch.no_grad():
        for x, y in val_loader:
            if collected >= n_samples:
                break
            x = x.squeeze(1).to(device)
            y = y.squeeze(1).to(device)

            s_x, s_y, s_y_pred, _ = model(x, y)
            all_s_x.append(s_x.cpu().numpy())
            all_s_y.append(s_y.cpu().numpy())
            all_s_y_pred.append(s_y_pred.cpu().numpy())
            collected += x.shape[0]

    s_x      = np.vstack(all_s_x)[:n_samples]
    s_y      = np.vstack(all_s_y)[:n_samples]
    s_y_pred = np.vstack(all_s_y_pred)[:n_samples]
    return s_x, s_y, s_y_pred


def plot_embedding_space_jepa(
    s_x: np.ndarray,
    s_y: np.ndarray,
    s_y_pred: np.ndarray,
) -> None:
    """t-SNE scatter of s_x, s_y, ŝ_y.

    A well-trained JEPA should have ŝ_y (red) sitting on top of s_y (green).
    Arrows connect each (s_y, ŝ_y) pair so you can see the residual error.
    """
    print("Computing t-SNE (may take ~1 min)…")
    all_emb   = np.vstack([s_x, s_y, s_y_pred])
    tsne      = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb_2d    = tsne.fit_transform(all_emb)

    n         = s_x.shape[0]
    s_x_2d    = emb_2d[:n]
    s_y_2d    = emb_2d[n:2*n]
    s_yp_2d   = emb_2d[2*n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(s_x_2d[:, 0],  s_x_2d[:, 1],  alpha=0.45, s=25, label="s_x (past)",            color="steelblue")
    ax.scatter(s_y_2d[:, 0],  s_y_2d[:, 1],  alpha=0.45, s=25, label="s_y (true future)",      color="seagreen")
    ax.scatter(s_yp_2d[:, 0], s_yp_2d[:, 1], alpha=0.45, s=25, label="ŝ_y (predicted future)", color="tomato")

    # Arrows: s_y → ŝ_y for first 60 samples
    for i in range(min(60, n)):
        ax.annotate("",
                    xy=s_yp_2d[i], xytext=s_y_2d[i],
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.25, lw=0.8))

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("JEPA: Embedding Space — ŝ_y should cluster on top of s_y",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    _save(fig, "embedding_space_tsne.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5 — JEPA embedding heatmaps (s_y vs ŝ_y per sample)
# ──────────────────────────────────────────────────────────────────────────────

def plot_jepa_embedding_heatmaps(
    jepa_model: JEPAModel,
    val_loader: DataLoader,
    device: torch.device,
    n_display: int = 8,
) -> None:
    """3-row panel per sample: input frame | s_y heatmap | ŝ_y heatmap.

    The embedding vectors are reshaped to a 2-D grid purely for visual
    comparison; closeness between row 2 and row 3 indicates good prediction.
    """
    jepa_model.eval()

    with torch.no_grad():
        x, y = next(iter(val_loader))
        x = x.squeeze(1).to(device)
        y = y.squeeze(1).to(device)

        s_x, s_y, s_y_pred, _ = jepa_model(x, y)

        x        = x.cpu()
        s_y      = s_y.cpu().numpy()
        s_y_pred = s_y_pred.cpu().numpy()

    embedding_dim = s_y.shape[1]
    # Pick a grid shape that tiles the embedding dimension neatly
    grid_cols = 16
    grid_rows = embedding_dim // grid_cols   # e.g. 128 // 16 = 8

    pred_errors = np.sqrt(np.mean((s_y - s_y_pred) ** 2, axis=1))
    n_display   = min(n_display, x.shape[0])

    fig, axes = plt.subplots(3, n_display, figsize=(2.2 * n_display, 7))

    for i in range(n_display):
        # Row 0: last input frame
        axes[0, i].imshow(x[i, 3].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"input t\n#{i+1}", fontsize=8)
        axes[0, i].axis("off")

        # Common colour scale so rows 1 and 2 are directly comparable
        vmin = min(s_y[i].min(), s_y_pred[i].min())
        vmax = max(s_y[i].max(), s_y_pred[i].max())

        # Row 1: s_y (true)
        im1 = axes[1, i].imshow(
            s_y[i].reshape(grid_rows, grid_cols),
            cmap="RdBu_r", vmin=vmin, vmax=vmax,
        )
        axes[1, i].set_title("s_y\n(true)", fontsize=8)
        axes[1, i].axis("off")

        # Row 2: ŝ_y (predicted)
        im2 = axes[2, i].imshow(
            s_y_pred[i].reshape(grid_rows, grid_cols),
            cmap="RdBu_r", vmin=vmin, vmax=vmax,
        )
        axes[2, i].set_title(f"ŝ_y\nerr={pred_errors[i]:.3f}", fontsize=8)
        axes[2, i].axis("off")

    # One shared colorbar on the right
    fig.colorbar(im2, ax=axes[1:, -1], fraction=0.05, pad=0.04, label="Activation")

    plt.suptitle("JEPA: Embedding Heatmaps — s_y (true) vs ŝ_y (predicted)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "jepa_embedding_heatmaps.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 6 — JEPA pixel-space decode panel  [NEW]
# ──────────────────────────────────────────────────────────────────────────────

def plot_jepa_pixel_decode(
    jepa_model: JEPAModel,
    val_loader: DataLoader,
    device: torch.device,
    n_display: int = 8,
) -> None:
    """Decode s_y and ŝ_y back to pixel space for visual comparison.

    Even though JEPA trains in embedding space, the auxiliary decoder lets us
    peek at what the embeddings "look like" as images.

    Row layout:
        0 — Ground truth y (target frame)
        1 — Decode(s_y):      how well the encoder captures y
        2 — Decode(ŝ_y):      how well the predictor approximates s_y
        3 — |Decode(s_y) − Decode(ŝ_y)|  error map
    """
    jepa_model.eval()

    with torch.no_grad():
        x, y = next(iter(val_loader))
        x = x.squeeze(1).to(device)
        y = y.squeeze(1).to(device)

        s_x, s_y, s_y_pred, _ = jepa_model(x, y)

        dec_sy    = jepa_model.decode_embedding(s_y)       # (B, 1, H, W)
        dec_sy_p  = jepa_model.decode_embedding(s_y_pred)  # (B, 1, H, W)

        y         = y.cpu().numpy()
        dec_sy    = dec_sy.cpu().numpy()
        dec_sy_p  = dec_sy_p.cpu().numpy()

    n_display = min(n_display, y.shape[0])
    fig, axes = plt.subplots(4, n_display, figsize=(2.2 * n_display, 9))

    row_labels = [
        "Ground truth y",
        "Decode(s_y)  — encoder quality",
        "Decode(ŝ_y)  — predictor quality",
        "|Decode(s_y) − Decode(ŝ_y)|  error",
    ]

    for i in range(n_display):
        gt    = y[i, 0]
        recon = dec_sy[i, 0]
        pred  = dec_sy_p[i, 0]
        err   = np.abs(recon - pred)

        psnr = _psnr(gt, pred)

        axes[0, i].imshow(gt,    cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(recon, cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(pred,  cmap="gray", vmin=0, vmax=1)
        axes[3, i].imshow(err,   cmap="hot",  vmin=0, vmax=0.5)

        for row in range(4):
            axes[row, i].axis("off")

        axes[2, i].set_title(f"PSNR {psnr:.1f} dB", fontsize=8)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=0,
                                labelpad=140, va="center")

    plt.suptitle("JEPA: Pixel-Space Decode Panel (auxiliary decoder)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "jepa_pixel_decode.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 7 — Per-dimension prediction error  [NEW]
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_dimension_error(
    s_y: np.ndarray,
    s_y_pred: np.ndarray,
) -> None:
    """Bar chart of mean squared error per embedding dimension.

    Reveals which dimensions of the embedding space the predictor struggles
    with most. Consistently high-error dimensions may represent aspects of
    motion that are hard to predict (e.g. digit identity changes, overlaps).
    """
    # per-dim MSE: (embedding_dim,)
    per_dim_mse = np.mean((s_y - s_y_pred) ** 2, axis=0)
    dims        = np.arange(len(per_dim_mse))

    # Sort descending so the hardest dimensions are on the left
    order        = np.argsort(per_dim_mse)[::-1]
    sorted_mse   = per_dim_mse[order]
    sorted_dims  = dims[order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: sorted bar chart
    axes[0].bar(np.arange(len(sorted_mse)), sorted_mse, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Dimension (sorted by error)", fontsize=12)
    axes[0].set_ylabel("Mean Squared Error", fontsize=12)
    axes[0].set_title("Per-Dimension Prediction Error (sorted)", fontsize=13, fontweight="bold")
    axes[0].axhline(y=per_dim_mse.mean(), color="red", linestyle="--",
                    linewidth=2, label=f"Mean MSE = {per_dim_mse.mean():.4f}")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right: unsorted, to show original index structure
    axes[1].bar(dims, per_dim_mse, color="seagreen", alpha=0.8)
    axes[1].set_xlabel("Dimension index", fontsize=12)
    axes[1].set_ylabel("Mean Squared Error", fontsize=12)
    axes[1].set_title("Per-Dimension Prediction Error (by index)", fontsize=13, fontweight="bold")
    axes[1].axhline(y=per_dim_mse.mean(), color="red", linestyle="--",
                    linewidth=2, label=f"Mean MSE = {per_dim_mse.mean():.4f}")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save(fig, "per_dimension_error.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 8 — Cosine similarity distribution  [NEW]
# ──────────────────────────────────────────────────────────────────────────────

def plot_cosine_similarity_distribution(
    s_y: np.ndarray,
    s_y_pred: np.ndarray,
) -> None:
    """Histogram of cos(s_y, ŝ_y) across the validation set.

    A well-trained JEPA predictor should have nearly all samples near 1.0.
    A wide or left-shifted distribution means the predictor is off in direction,
    not just magnitude — a sign of poor training or representation collapse.
    """
    s_y_t    = torch.from_numpy(s_y).float()
    s_yp_t   = torch.from_numpy(s_y_pred).float()
    cos_sims = F.cosine_similarity(s_y_t, s_yp_t, dim=1).numpy()  # (N,)

    mean_cos = cos_sims.mean()
    med_cos  = np.median(cos_sims)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(cos_sims, bins=50, color="royalblue", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(x=mean_cos, color="red",    linestyle="--", linewidth=2,
               label=f"Mean = {mean_cos:.3f}")
    ax.axvline(x=med_cos,  color="orange", linestyle="--", linewidth=2,
               label=f"Median = {med_cos:.3f}")
    ax.axvline(x=1.0,      color="green",  linestyle=":",  linewidth=2,
               label="Perfect = 1.0")
    ax.set_xlabel("Cosine Similarity (s_y · ŝ_y)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("JEPA: Cosine Similarity Distribution — closer to 1.0 is better",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.05, 1.05)

    _save(fig, "cosine_similarity_distribution.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 9 — Generative model reconstructions
# ──────────────────────────────────────────────────────────────────────────────

def plot_generative_reconstructions(
    model: GenerativeModel,
    val_loader: DataLoader,
    device: torch.device,
    n_display: int = 8,
) -> None:
    """Ground truth vs pixel reconstruction for the generative baseline.

    Row 0 — target frame y
    Row 1 — reconstruction ŷ
    Row 2 — |y − ŷ|  pixel error map
    """
    model.eval()

    with torch.no_grad():
        x, y = next(iter(val_loader))
        y      = y.squeeze(1).to(device)
        y_recon, _ = model(y)
        y      = y.cpu().numpy()
        y_recon= y_recon.cpu().numpy()

    n_display = min(n_display, y.shape[0])
    fig, axes = plt.subplots(3, n_display, figsize=(2.2 * n_display, 7))

    for i in range(n_display):
        gt  = y[i, 0]
        rec = y_recon[i, 0]
        err = np.abs(gt - rec)
        psnr = _psnr(gt, rec)

        axes[0, i].imshow(gt,  cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(rec, cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(err, cmap="hot",  vmin=0, vmax=0.5)

        for row in range(3):
            axes[row, i].axis("off")

        axes[1, i].set_title(f"PSNR {psnr:.1f} dB", fontsize=8)

    row_labels = ["Ground truth y", "Reconstruction ŷ", "|y − ŷ|  error"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=0,
                                labelpad=130, va="center")

    plt.suptitle("Generative Model: Pixel-Space Reconstruction",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "generative_reconstructions.png")


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to OUTPUT_DIR and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def _psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (higher = better reconstruction)."""
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(data_range / np.sqrt(mse))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    # Validation dataset — use IMAGE_SIZE=64 to match training
    val_dataset = MovingMNISTDataset(
        n_sequences=200,
        seq_length=5,
        image_size=IMAGE_SIZE,
        num_digits=2,
        seed=43,           # different from training seeds
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # ── Static plots (no model needed) ───────────────────────────────────────
    print("\n=== [1/8] Loss Curves ===")
    plot_loss_curves()

    print("\n=== [2/8] Effective Rank ===")
    plot_effective_rank_jepa()

    print("\n=== [3/8] Temporal Input Strips ===")
    plot_temporal_strips(val_loader)

    # ── JEPA model plots ──────────────────────────────────────────────────────
    print("\n=== Loading JEPA checkpoint ===")
    jepa_model = JEPAModel(embedding_dim=128, hidden_dim=512, image_size=IMAGE_SIZE)
    load_jepa_checkpoint(jepa_model, OUTPUT_DIR / "checkpoint_jepa.pt", device)
    jepa_model = jepa_model.to(device)

    print("\n=== [4/8] t-SNE Embedding Space ===")
    s_x, s_y, s_y_pred = extract_embeddings_jepa(jepa_model, val_loader, device, n_samples=500)
    plot_embedding_space_jepa(s_x, s_y, s_y_pred)

    print("\n=== [5/8] Embedding Heatmaps ===")
    plot_jepa_embedding_heatmaps(jepa_model, val_loader, device)

    print("\n=== [6/8] Pixel-Space Decode Panel ===")
    plot_jepa_pixel_decode(jepa_model, val_loader, device)

    print("\n=== [7/8] Per-Dimension Prediction Error ===")
    plot_per_dimension_error(s_y, s_y_pred)

    print("\n=== [8/8] Cosine Similarity Distribution ===")
    plot_cosine_similarity_distribution(s_y, s_y_pred)

    # ── Generative model plots ────────────────────────────────────────────────
    print("\n=== [+] Generative Model Reconstructions ===")
    gen_model = GenerativeModel(embedding_dim=128, image_size=IMAGE_SIZE)
    load_generative_checkpoint(gen_model, OUTPUT_DIR / "checkpoint_generative.pt", device)
    gen_model = gen_model.to(device)
    plot_generative_reconstructions(gen_model, val_loader, device)

    print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    main()
