"""Visualize Step 3 Joint Embedding Architecture embeddings."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step03_joint_embedding.model import JointEmbeddingModel
from shared.data import CIFAR10PairDataset
from shared.device import get_device
from shared.viz import plot_energy_surface, plot_training_curve

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


def plot_dimension_variance(embeddings: np.ndarray, out_path: Path) -> None:
    variances = embeddings.var(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(variances)), variances)
    ax.set_title("Embedding dimension variance")
    ax.set_xlabel("dimension")
    ax.set_ylabel("variance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_embedding_correlation(embeddings: np.ndarray, out_path: Path) -> None:
    corr = np.corrcoef(embeddings, rowvar=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title("Embedding feature correlation")
    ax.set_xlabel("embedding dim")
    ax.set_ylabel("embedding dim")
    fig.colorbar(im, ax=ax, label="correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_energy_surface(
    embeddings: np.ndarray,
    z_ref: np.ndarray,
    method: str,
    out_path: Path,
    title: str,
) -> None:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min - margin_x, x_max + margin_x, 100),
        np.linspace(y_min - margin_y, y_max + margin_y, 100),
    )
    grid_flat = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    grid_full = pca.inverse_transform(grid_flat)
    if method == "distance":
        energies = np.sum((grid_full - z_ref[None, :]) ** 2, axis=1)
    else:
        z_ref_norm = z_ref / np.linalg.norm(z_ref)
        grid_norm = grid_full / np.linalg.norm(grid_full, axis=1, keepdims=True)
        energies = -np.sum(z_ref_norm[None, :] * grid_norm, axis=1)
    energies = energies.reshape(grid_x.shape)
    grid = np.stack([grid_x, grid_y], axis=-1)
    plot_energy_surface(grid, energies, out_path, title)


def plot_training_curve_comparison(histories: dict[str, dict[str, list[float]]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, history in histories.items():
        ax.plot(history["epoch"], history["mean_loss"], label=f"{method} mean")
    ax.set_title("Training loss comparison")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_variance_comparison(embeddings_dict: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, embeddings in embeddings_dict.items():
        variances = np.sort(np.var(embeddings, axis=0))[::-1]
        ax.plot(variances, label=method)
    ax.set_title("Sorted embedding variance comparison")
    ax.set_xlabel("dimension rank")
    ax.set_ylabel("variance")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Step 3 Joint Embedding Architecture")
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=Path, default=Path("./data"))
    parser.add_argument("--method", choices=["distance", "info_nce", "both"], default="both")
    args = parser.parse_args()

    methods = [args.method] if args.method != "both" else ["distance", "info_nce"]
    metadata = {}
    embeddings_by_method = {}
    histories = {}

    device = get_device()
    print(f"Using device: {device}")
    set_seed(args.seed)

    for method in methods:
        method_dir = OUTPUT_DIR / method
        method_dir.mkdir(exist_ok=True, parents=True)

        ckpt_path = OUTPUT_DIR / f"checkpoint_{method}.pt"
        hist_path = OUTPUT_DIR / f"loss_history_{method}.json"
        if not ckpt_path.exists():
            raise SystemExit(f"No checkpoint at {ckpt_path}. Run train.py for method {method} first.")

        ckpt = torch.load(ckpt_path, map_location=device)
        model = JointEmbeddingModel().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dataset = CIFAR10PairDataset(root=args.root, train=True, subset_size=args.subset_size, seed=args.seed, eval_mode=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        embeddings = []
        labels = []
        with torch.no_grad():
            for x1, _, y in loader:
                x1 = x1.to(device)
                z = model(x1).cpu().numpy()
                embeddings.append(z)
                labels.append(y.numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        embeddings_by_method[method] = embeddings

        plot_tsne(embeddings, labels, method_dir / f"embedding_tsne_{method}.png")
        plot_dimension_variance(embeddings, method_dir / f"embedding_variance_{method}.png")
        plot_embedding_correlation(embeddings, method_dir / f"embedding_correlation_{method}.png")
        compute_energy_surface(
            embeddings,
            embeddings[0],
            method,
            method_dir / f"energy_surface_{method}.png",
            f"Energy surface for {method} joint embedding",
        )

        with open(hist_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        histories[method] = history
        plot_training_curve(
            [h["epoch"] for h in history],
            [h["mean_loss"] for h in history],
            [h["std_loss"] for h in history],
            method_dir / f"training_curve_{method}.png",
            f"Joint embedding {method} training curve",
        )

    if len(methods) > 1:
        plot_training_curve_comparison(
            {m: {"epoch": [h["epoch"] for h in histories[m]], "mean_loss": [h["mean_loss"] for h in histories[m]]} for m in methods},
            OUTPUT_DIR / "training_curve_comparison.png",
        )
        plot_variance_comparison(embeddings_by_method, OUTPUT_DIR / "embedding_variance_comparison.png")

    print("Saved visualizations to outputs/")


if __name__ == "__main__":
    main()
