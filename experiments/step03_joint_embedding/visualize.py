"""Visualize Step 3 Joint Embedding Architecture embeddings."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step03_joint_embedding.model import JointEmbeddingModel
from shared.data import CIFAR10PairDataset
from shared.device import get_device
from shared.viz import plot_training_curve

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Step 3 Joint Embedding Architecture")
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

    plot_tsne(embeddings, labels, OUTPUT_DIR / "embedding_tsne.png")
    plot_dimension_variance(embeddings, OUTPUT_DIR / "embedding_variance.png")

    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    plot_training_curve(
        [h["epoch"] for h in history],
        [h["mean_loss"] for h in history],
        [h["std_loss"] for h in history],
        OUTPUT_DIR / "training_curve.png",
        "Joint embedding InfoNCE training curve",
    )
    print("Saved visualizations to outputs/")


if __name__ == "__main__":
    main()
