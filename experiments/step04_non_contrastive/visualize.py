"""Visualize Step 4 non-contrastive joint embedding training."""

import argparse
import json
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

from experiments.step04_non_contrastive.model import (
    JointEmbeddingModel,
    cross_correlation_matrix,
)
from shared.data import CIFAR10PairDataset
from shared.device import get_device

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def set_seed(seed: int) -> None:
    import numpy as np
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, init="pca", random_state=42)
    projection = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.8,
    )
    ax.set_title("CIFAR-10 joint embedding t-SNE")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    legend = ax.legend(
        *scatter.legend_elements(num=10),
        title="class",
        loc="best",
        fontsize="small",
    )
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(correlation: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("Barlow Twins cross-correlation matrix")
    ax.set_xlabel("projection dimension")
    ax.set_ylabel("projection dimension")
    fig.colorbar(im, ax=ax, label="correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_covariance_eigen_spectrum(embeddings: np.ndarray, out_path: Path) -> None:
    cov = np.cov(embeddings, rowvar=False, bias=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(eigenvalues) + 1), np.sort(eigenvalues)[::-1], marker=".")
    ax.set_yscale("log")
    ax.set_title("Covariance eigenvalue spectrum")
    ax.set_xlabel("component index")
    ax.set_ylabel("eigenvalue")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_effective_rank(history: list[dict], out_path: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    ranks = [entry.get("effective_rank", 0.0) for entry in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, ranks, marker="o")
    ax.set_title("Effective rank vs epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("effective rank")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_and_barlow_stats(history: list[dict], out_path: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    mean_loss = [entry["mean_loss"] for entry in history]
    mean_diag = [entry.get("mean_diag", 0.0) for entry in history]
    mean_abs_offdiag = [entry.get("mean_abs_offdiag", 0.0) for entry in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, mean_loss, color="tab:blue", label="mean loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, mean_diag, color="tab:green", label="mean diag")
    ax2.plot(epochs, mean_abs_offdiag, color="tab:red", label="mean abs off-diag")
    ax2.set_ylabel("correlation stats", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", fontsize="small")
    ax1.set_title("Training loss and Barlow Twins metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca_density_contours(
    embeddings: np.ndarray,
    out_path: Path,
) -> None:
    coords = PCA(n_components=2).fit_transform(embeddings)

    x = coords[:, 0]
    y = coords[:, 1]

    kde = gaussian_kde(np.vstack([x, y]))

    xx, yy = np.mgrid[
        x.min():x.max():200j,
        y.min():y.max():200j,
    ]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))

    contour = ax.contourf(
        xx,
        yy,
        density,
        levels=30,
        cmap="viridis",
    )

    ax.scatter(
        x,
        y,
        s=5,
        alpha=0.25,
        color="white",
    )

    fig.colorbar(contour, ax=ax, label="density")

    ax.set_title("Embedding manifold density")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_umap(embeddings: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab10",
        s=6,
        alpha=0.8,
    )
    ax.set_title("UMAP projection of joint embeddings")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    legend = ax.legend(
        *scatter.legend_elements(num=10),
        title="class",
        loc="best",
        fontsize="small",
    )
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    

def plot_pca_manifold_growth(
    snapshot_paths: list[Path],
    out_path: Path,
) -> None:
    TARGET_EPOCHS = {0, 20, 39}

    snapshots = []
    labels = []
    epochs = []

    for snapshot_path in snapshot_paths:
        epoch = int(snapshot_path.stem.split("_")[-1])

        if epoch not in TARGET_EPOCHS:
            continue

        data = np.load(snapshot_path)

        snapshots.append(data["embeddings"])
        labels.append(data["labels"])
        epochs.append(epoch)

    if not snapshots:
        return

    selected = sorted(zip(epochs, snapshots, labels), key=lambda x: x[0])
    epochs, snapshots, labels = zip(*selected)

    stacked = np.concatenate(snapshots, axis=0)

    pca = PCA(n_components=2)
    pca.fit(stacked)

    all_coords = [pca.transform(e) for e in snapshots]

    x_min = min(c[:, 0].min() for c in all_coords)
    x_max = max(c[:, 0].max() for c in all_coords)
    y_min = min(c[:, 1].min() for c in all_coords)
    y_max = max(c[:, 1].max() for c in all_coords)

    pad_x = 0.05 * (x_max - x_min)
    pad_y = 0.05 * (y_max - y_min)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        sharex=True,
        sharey=True,
    )

    for ax, epoch, coords, lbl in zip(
        axes,
        epochs,
        all_coords,
        labels,
    ):
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=lbl,
            cmap="tab10",
            s=4,
            alpha=0.35,
        )

        ax.set_title(f"Epoch {epoch}")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    fig.suptitle(
        "Embedding manifold evolution in a shared PCA basis",
        fontsize=18,
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Step 4 non-contrastive joint embedding")
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=Path, default=Path("./data"))
    args = parser.parse_args()

    ckpt_path = OUTPUT_DIR / "checkpoint.pt"
    hist_path = OUTPUT_DIR / "loss_history.json"
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint at {ckpt_path}. Run train.py first.")

    device = get_device()
    print(f"Using device: {device}")
    set_seed(args.seed)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = JointEmbeddingModel().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = CIFAR10PairDataset(
        root=args.root,
        train=True,
        subset_size=args.subset_size,
        seed=args.seed,
        eval_mode=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    embeddings = []
    labels = []
    z1_embeddings = []
    z2_embeddings = []
    with torch.no_grad():
        for x1, x2, y in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            z1 = model(x1).cpu()
            z2 = model(x2).cpu()
            embeddings.append(((z1 + z2) / 2).numpy())
            labels.append(y.numpy())
            z1_embeddings.append(z1)
            z2_embeddings.append(z2)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    z1_embeddings = torch.cat(z1_embeddings, dim=0)
    z2_embeddings = torch.cat(z2_embeddings, dim=0)
    correlation = cross_correlation_matrix(z1_embeddings, z2_embeddings).cpu().numpy()

    plot_tsne(embeddings, labels, OUTPUT_DIR / "embedding_tsne.png")
    plot_umap(embeddings, labels, OUTPUT_DIR / "embedding_umap.png")
    plot_correlation_heatmap(correlation, OUTPUT_DIR / "cross_correlation_heatmap.png")
    plot_covariance_eigen_spectrum(embeddings, OUTPUT_DIR / "covariance_eigen_spectrum.png")
    plot_pca_density_contours(embeddings, OUTPUT_DIR / "embedding_pca_density_contours.png")

    snapshot_paths = sorted(OUTPUT_DIR.glob("embeddings_epoch_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    if snapshot_paths:
        plot_pca_manifold_growth(snapshot_paths, OUTPUT_DIR / "embedding_pca_growth_grid.png")

    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    plot_loss_and_barlow_stats(history, OUTPUT_DIR / "training_loss_and_metrics.png")
    plot_effective_rank(history, OUTPUT_DIR / "effective_rank.png")

    print(f"Saved visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
