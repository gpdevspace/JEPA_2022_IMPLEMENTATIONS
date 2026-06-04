"""Visualize JEPA training with sophisticated plots showing collapse and recovery."""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step05_jepa.model import JEPA
from shared.data import CIFAR10PairDataset
from shared.device import get_device

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def plot_training_curves(history: list[dict], out_path: Path) -> None:
    """Plot training loss and VICReg components over epochs."""
    epochs = [h["epoch"] for h in history]
    loss_type = history[0]["loss"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"JEPA Training Curves ({loss_type.upper()} Loss)", fontsize=16, fontweight="bold")
    
    # Loss
    ax = axes[0, 0]
    mean_loss = [h["mean_loss"] for h in history]
    std_loss = [h["std_loss"] for h in history]
    ax.plot(epochs, mean_loss, color="#1f77b4", linewidth=2, label="Mean Loss")
    ax.fill_between(epochs, 
                    np.array(mean_loss) - np.array(std_loss),
                    np.array(mean_loss) + np.array(std_loss),
                    alpha=0.3, color="#1f77b4")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # VICReg components (if applicable)
    ax = axes[0, 1]
    if loss_type == "vicreg":
        invariance = [h["invariance"] for h in history]
        variance_loss = [h["variance_loss"] for h in history]
        covariance_loss = [h["covariance_loss"] for h in history]
        ax.plot(epochs, invariance, color="#ff7f0e", linewidth=2, label="Invariance")
        ax.plot(epochs, variance_loss, color="#2ca02c", linewidth=2, label="Variance")
        ax.plot(epochs, covariance_loss, color="#d62728", linewidth=2, label="Covariance")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss Component", fontsize=11)
        ax.set_title("VICReg Loss Components", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "MSE loss has no components\n(single term)", 
                ha="center", va="center", fontsize=12, style="italic")
        ax.set_title("Loss Components", fontsize=12, fontweight="bold")
    
    # Effective rank (collapse indicator)
    ax = axes[1, 0]
    s_y_pred_rank = [h["s_y_pred_effective_rank"] for h in history]
    encoder_dim = history[0].get("encoder_dim", 512)
    ax.plot(epochs, s_y_pred_rank, color="#9467bd", linewidth=2, label="Effective Rank")
    ax.axhline(y=encoder_dim, color="gray", linestyle="--", alpha=0.5, label="Max Rank")
    ax.axhline(y=encoder_dim * 0.1, color="red", linestyle="--", alpha=0.5, label="Collapse Threshold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title("Embedding Effective Rank (Collapse Detector)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean std (collapse indicator)
    ax = axes[1, 1]
    s_y_pred_std = [h["s_y_pred_mean_std"] for h in history]
    s_y_pred_min_std = [h["s_y_pred_min_std"] for h in history]
    ax.plot(epochs, s_y_pred_std, color="#8c564b", linewidth=2, label="Mean Std")
    ax.plot(epochs, s_y_pred_min_std, color="#e377c2", linewidth=2, label="Min Std")
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="Dead Dim Threshold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Standard Deviation", fontsize=11)
    ax.set_title("Embedding Standard Deviation (Collapse Detector)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_comparison(
    s_x: np.ndarray,
    s_y: np.ndarray,
    s_y_pred: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    """Compare s_x, s_y, and s_y_pred using PCA and t-SNE."""
    # Use PCA for faster visualization
    pca = PCA(n_components=2)
    s_x_pca = pca.fit_transform(s_x)
    s_y_pca = pca.transform(s_y)
    s_y_pred_pca = pca.transform(s_y_pred)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Embedding Space Comparison (PCA)", fontsize=16, fontweight="bold")
    
    for ax, data, title in zip(
        axes,
        [s_x_pca, s_y_pca, s_y_pred_pca],
        ["s_x (Input)", "s_y (Target)", "s_y_pred (Predicted)"],
    ):
        scatter = ax.scatter(
            data[:, 0], data[:, 1],
            c=labels, cmap="tab10", s=8, alpha=0.6,
        )
        ax.set_xlabel("PC 1", fontsize=11)
        ax.set_ylabel("PC 2", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Class", fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_error_distribution(
    s_y: np.ndarray,
    s_y_pred: np.ndarray,
    out_path: Path,
) -> None:
    """Plot distribution of prediction errors in embedding space."""
    errors = np.linalg.norm(s_y - s_y_pred, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Prediction Error Distribution in Embedding Space", fontsize=16, fontweight="bold")
    
    # Histogram
    ax = axes[0]
    ax.hist(errors, bins=50, color="#1f77b4", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(errors), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(errors):.4f}")
    ax.axvline(np.median(errors), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(errors):.4f}")
    ax.set_xlabel("L2 Error (||s_y - s_y_pred||)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Error Histogram", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot with statistics
    ax = axes[1]
    bp = ax.boxplot(errors, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("#1f77b4")
    bp["boxes"][0].set_alpha(0.7)
    ax.set_ylabel("L2 Error", fontsize=11)
    ax.set_title("Error Statistics", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add text statistics
    stats_text = f"Mean: {np.mean(errors):.4f}\nStd: {np.std(errors):.4f}\nMin: {np.min(errors):.4f}\nMax: {np.max(errors):.4f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_wise_statistics(
    s_y: np.ndarray,
    s_y_pred: np.ndarray,
    out_path: Path,
) -> None:
    """Plot per-dimension statistics to detect collapse."""
    # Compute per-dimension statistics
    s_y_std = s_y.std(axis=0)
    s_y_pred_std = s_y_pred.std(axis=0)
    s_y_mean = s_y.mean(axis=0)
    s_y_pred_mean = s_y_pred.mean(axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-Dimension Embedding Statistics", fontsize=16, fontweight="bold")
    
    # Standard deviation comparison
    ax = axes[0, 0]
    ax.plot(s_y_std, color="#1f77b4", alpha=0.7, linewidth=1, label="s_y")
    ax.plot(s_y_pred_std, color="#ff7f0e", alpha=0.7, linewidth=1, label="s_y_pred")
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="Dead Threshold")
    ax.set_xlabel("Dimension", fontsize=11)
    ax.set_ylabel("Standard Deviation", fontsize=11)
    ax.set_title("Per-Dimension Standard Deviation", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean comparison
    ax = axes[0, 1]
    ax.plot(s_y_mean, color="#1f77b4", alpha=0.7, linewidth=1, label="s_y")
    ax.plot(s_y_pred_mean, color="#ff7f0e", alpha=0.7, linewidth=1, label="s_y_pred")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Dimension", fontsize=11)
    ax.set_ylabel("Mean", fontsize=11)
    ax.set_title("Per-Dimension Mean (Bias Check)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Std histogram
    ax = axes[1, 0]
    ax.hist(s_y_std, bins=30, color="#1f77b4", alpha=0.5, label="s_y", edgecolor="black")
    ax.hist(s_y_pred_std, bins=30, color="#ff7f0e", alpha=0.5, label="s_y_pred", edgecolor="black")
    ax.axvline(x=0.01, color="red", linestyle="--", alpha=0.5, label="Dead Threshold")
    ax.set_xlabel("Standard Deviation", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Std Distribution Across Dimensions", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean histogram
    ax = axes[1, 1]
    ax.hist(s_y_mean, bins=30, color="#1f77b4", alpha=0.5, label="s_y", edgecolor="black")
    ax.hist(s_y_pred_mean, bins=30, color="#ff7f0e", alpha=0.5, label="s_y_pred", edgecolor="black")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Mean Distribution Across Dimensions", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_collapse_evolution(
    snapshot_paths: list[Path],
    out_path: Path,
) -> None:
    """Plot how collapse metrics evolve over training."""
    target_epochs = {0, 10, 20, 30, 39}
    
    snapshots = []
    epochs = []
    
    for path in snapshot_paths:
        epoch = int(path.stem.split("_")[-1])
        if epoch in target_epochs:
            data = np.load(path)
            snapshots.append(data)
            epochs.append(epoch)
    
    if not snapshots:
        print("No embedding snapshots found")
        return
    
    # Sort by epoch
    epochs, snapshots = zip(*sorted(zip(epochs, snapshots)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Collapse Evolution Over Training", fontsize=16, fontweight="bold")
    
    # Track metrics over epochs
    mean_stds = []
    min_stds = []
    effective_ranks = []
    
    for epoch, data in zip(epochs, snapshots):
        s_y_pred = data["s_y_pred"]
        std_per_dim = s_y_pred.std(axis=0)
        mean_stds.append(std_per_dim.mean())
        min_stds.append(std_per_dim.min())
        
        # Effective rank
        centered = s_y_pred - s_y_pred.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / (s_y_pred.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)
        total = eigvals.sum()
        if total > 1e-8:
            p = eigvals / total
            rank = np.exp(-np.sum(p * np.log(p + 1e-12)))
        else:
            rank = 0
        effective_ranks.append(rank)
    
    # Plot metrics evolution
    ax = axes[0, 0]
    ax.plot(epochs, mean_stds, marker="o", linewidth=2, markersize=8, color="#1f77b4")
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="Dead Threshold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Std", fontsize=11)
    ax.set_title("Mean Std Over Training", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(epochs, min_stds, marker="o", linewidth=2, markersize=8, color="#ff7f0e")
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="Dead Threshold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Min Std", fontsize=11)
    ax.set_title("Min Std Over Training", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    encoder_dim = snapshots[0]["s_y_pred"].shape[1]
    ax.plot(epochs, effective_ranks, marker="o", linewidth=2, markersize=8, color="#2ca02c")
    ax.axhline(y=encoder_dim, color="gray", linestyle="--", alpha=0.5, label="Max Rank")
    ax.axhline(y=encoder_dim * 0.1, color="red", linestyle="--", alpha=0.5, label="Collapse Threshold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title("Effective Rank Over Training", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PCA evolution (shared basis)
    all_embeddings = np.concatenate([s["s_y_pred"] for s in snapshots], axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)
    
    for idx, (epoch, data) in enumerate(zip(epochs, snapshots)):
        ax_idx = idx // 2 + 1
        ax_col = idx % 2
        if ax_idx >= 2:
            continue
        
        s_y_pred_pca = pca.transform(data["s_y_pred"])
        ax = axes[ax_idx, ax_col]
        scatter = ax.scatter(
            s_y_pred_pca[:, 0], s_y_pred_pca[:, 1],
            c=data["labels"], cmap="tab10", s=8, alpha=0.6,
        )
        ax.set_xlabel("PC 1", fontsize=11)
        ax.set_ylabel("PC 2", fontsize=11)
        ax.set_title(f"Epoch {epoch} (Shared PCA)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_comparison(
    s_x: np.ndarray,
    s_y: np.ndarray,
    s_y_pred: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    """Compare embeddings using t-SNE for better visualization."""
    # Subsample for t-SNE (computationally expensive)
    n_samples = min(1000, len(s_x))
    indices = np.random.choice(len(s_x), n_samples, replace=False)
    
    s_x_sub = s_x[indices]
    s_y_sub = s_y[indices]
    s_y_pred_sub = s_y_pred[indices]
    labels_sub = labels[indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Embedding Space Comparison (t-SNE)", fontsize=16, fontweight="bold")
    
    for ax, data, title in zip(
        axes,
        [s_x_sub, s_y_sub, s_y_pred_sub],
        ["s_x (Input)", "s_y (Target)", "s_y_pred (Predicted)"],
    ):
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, 
                    init="pca", random_state=42)
        projection = tsne.fit_transform(data)
        scatter = ax.scatter(
            projection[:, 0], projection[:, 1],
            c=labels_sub, cmap="tab10", s=8, alpha=0.6,
        )
        ax.set_xlabel("t-SNE 1", fontsize=11)
        ax.set_ylabel("t-SNE 2", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Class", fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_space_jepa(
    s_x: np.ndarray,
    s_y: np.ndarray,
    s_y_pred: np.ndarray,
    out_path: Path,
) -> None:
    """Plot t-SNE of JEPA embedding space with alignment arrows."""
    print("  Generating embedding space visualization (t-SNE with arrows)...")
    n_samples = min(1000, len(s_x))
    indices = np.random.choice(len(s_x), n_samples, replace=False)
    
    s_y_sub = s_y[indices]
    s_y_pred_sub = s_y_pred[indices]

    # Combine all embeddings for consistent t-SNE
    all_embeddings = np.vstack([s_y_sub, s_y_pred_sub])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, init="pca")
    embeddings_2d = tsne.fit_transform(all_embeddings)

    n = s_y_sub.shape[0]
    s_y_2d = embeddings_2d[:n]
    s_y_pred_2d = embeddings_2d[n :]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(s_y_2d[:, 0], s_y_2d[:, 1], alpha=0.5, s=30, label="s_y (actual target)", color="green")
    ax.scatter(
        s_y_pred_2d[:, 0],
        s_y_pred_2d[:, 1],
        alpha=0.5,
        s=30,
        label="ŝ_y (predicted target)",
        color="red",
    )

    # Draw arrows from s_y to s_y_pred for first 50 samples
    for i in range(min(50, n)):
        ax.annotate(
            "",
            xy=s_y_pred_2d[i],
            xytext=s_y_2d[i],
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3, lw=1),
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("JEPA: Embedding Space (Predicted ŝ_y should cluster near actual s_y)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Step 5 JEPA training")
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=Path, default=Path("./data"))
    args = parser.parse_args()

    ckpt_path = OUTPUT_DIR / "checkpoint.pt"
    hist_path = OUTPUT_DIR / "loss_history.json"
    
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint at {ckpt_path}. Run train.py first.")
    if not hist_path.exists():
        raise SystemExit(f"No history at {hist_path}. Run train.py first.")

    device = get_device()
    print(f"Using device: {device}")
    set_seed(args.seed)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model = JEPA(
        encoder_dim=ckpt["encoder_dim"],
        predictor_hidden=ckpt["predictor_hidden"],
        shared_encoder=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load history
    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    loss_type = history[0]["loss"]
    print(f"Visualizing training with {loss_type} loss")

    # Load final embeddings
    final_epoch = history[-1]["epoch"]
    final_embeddings_path = OUTPUT_DIR / f"embeddings_epoch_{final_epoch}.npz"
    
    if final_embeddings_path.exists():
        data = np.load(final_embeddings_path)
        s_x = data["s_x"]
        s_y = data["s_y"]
        s_y_pred = data["s_y_pred"]
        labels = data["labels"]
        print(f"Loaded final embeddings from epoch {final_epoch}")
    else:
        # Generate embeddings on the fly
        print("Generating embeddings on the fly...")
        dataset = CIFAR10PairDataset(
            root=args.root,
            train=True,
            subset_size=args.subset_size,
            seed=args.seed,
            eval_mode=True,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

        s_x_list, s_y_list, s_y_pred_list, labels_list = [], [], [], []
        with torch.no_grad():
            for x, y, lbl in loader:
                x, y = x.to(device), y.to(device)
                s_x_emb, s_y_emb, s_y_pred_emb = model(x, y)
                s_x_list.append(s_x_emb.cpu().numpy())
                s_y_list.append(s_y_emb.cpu().numpy())
                s_y_pred_list.append(s_y_pred_emb.cpu().numpy())
                labels_list.append(lbl.numpy())

        s_x = np.concatenate(s_x_list, axis=0)
        s_y = np.concatenate(s_y_list, axis=0)
        s_y_pred = np.concatenate(s_y_pred_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

    # Generate visualizations
    print("Generating visualizations...")
    
    plot_training_curves(history, OUTPUT_DIR / "training_curves.png")
    print("  Saved training_curves.png")
    
    plot_embedding_comparison(s_x, s_y, s_y_pred, labels, OUTPUT_DIR / "embedding_comparison_pca.png")
    print("  Saved embedding_comparison_pca.png")
    
    plot_prediction_error_distribution(s_y, s_y_pred, OUTPUT_DIR / "prediction_error_distribution.png")
    print("  Saved prediction_error_distribution.png")
    
    plot_dimension_wise_statistics(s_y, s_y_pred, OUTPUT_DIR / "dimension_wise_statistics.png")
    print("  Saved dimension_wise_statistics.png")
    
    # Collapse evolution (if snapshots exist)
    snapshot_paths = sorted(
        OUTPUT_DIR.glob("embeddings_epoch_*.npz"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    if len(snapshot_paths) > 1:
        plot_collapse_evolution(snapshot_paths, OUTPUT_DIR / "collapse_evolution.png")
        print("  Saved collapse_evolution.png")
    
    # t-SNE (computationally expensive, do last)
    print("  Generating t-SNE (this may take a while)...")
    plot_tsne_comparison(s_x, s_y, s_y_pred, labels, OUTPUT_DIR / "embedding_comparison_tsne.png")
    print("  Saved embedding_comparison_tsne.png")
    
    plot_embedding_space_jepa(s_x, s_y, s_y_pred, OUTPUT_DIR / "embedding_space_jepa.png")

    print(f"\nAll visualizations saved to {OUTPUT_DIR}")
    
    # Print summary
    final_stats = history[-1]
    print(f"\nFinal summary ({loss_type} loss):")
    print(f"  Effective rank: {final_stats['s_y_pred_effective_rank']:.1f} / {ckpt['encoder_dim']}")
    print(f"  Mean std: {final_stats['s_y_pred_mean_std']:.4f}")
    print(f"  Min std: {final_stats['s_y_pred_min_std']:.4f}")
    
    if loss_type == "mse":
        if final_stats["s_y_pred_effective_rank"] < ckpt["encoder_dim"] * 0.1:
            print("  ⚠️  COLLAPSE CONFIRMED: Low effective rank")
        if final_stats["s_y_pred_min_std"] < 0.01:
            print("  ⚠️  COLLAPSE CONFIRMED: Dead dimensions detected")
    elif loss_type == "vicreg":
        if final_stats["s_y_pred_effective_rank"] > ckpt["encoder_dim"] * 0.5:
            print("  ✓ VICReg successfully prevented collapse")


if __name__ == "__main__":
    main()
