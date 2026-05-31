"""Step 4: Non-contrastive joint embedding training on CIFAR-10."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step04_non_contrastive.model import (
    JointEmbeddingModel,
    barlow_twins_loss,
    compute_covariance_eigenvalues,
    effective_rank_from_eigenvalues,
    variance_covariance_loss,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Step 4 non-contrastive joint embedding")
    parser.add_argument("--root", type=Path, default=Path("./data"))
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["barlow", "variance_covariance"], default="barlow")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-embeddings-every", type=int, default=10)
    args = parser.parse_args()

    device = get_device()
    if device.type == "cpu":
        print("Warning: MPS not available; training on CPU.")
    print(f"Using device: {device}")

    set_seed(args.seed)
    dataset = CIFAR10PairDataset(
        root=args.root,
        train=True,
        subset_size=args.subset_size,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
        persistent_workers=(args.workers > 0),
    )

    model = JointEmbeddingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        epoch_embeddings = []
        epoch_effective_ranks = []
        epoch_metrics = {
            "mean_diag": [],
            "mean_abs_offdiag": [],
            "diag_loss": [],
            "off_diag_loss": [],
        }

        epoch_labels = []
        for x1, x2, y in tqdm(loader, desc=f"Epoch {epoch}"):
            x1, x2 = x1.to(device), x2.to(device)
            z1 = model(x1)
            z2 = model(x2)

            if args.method == "barlow":
                loss, metrics = barlow_twins_loss(z1, z2)
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
            else:
                loss = variance_covariance_loss(z1, z2)
                metrics = {
                    "mean_diag": 0.0,
                    "mean_abs_offdiag": 0.0,
                    "diag_loss": 0.0,
                    "off_diag_loss": 0.0,
                }

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            repr_batch = ((z1 + z2) / 2).detach().cpu()
            epoch_embeddings.append(repr_batch)
            epoch_labels.append(y.numpy())

        epoch_embeddings = torch.cat(epoch_embeddings, dim=0)
        epoch_labels = np.concatenate(epoch_labels, axis=0)
        eigvals = compute_covariance_eigenvalues(epoch_embeddings)
        effective_rank = float(effective_rank_from_eigenvalues(eigvals).item())

        mean_loss = float(torch.tensor(losses).mean())
        std_loss = float(torch.tensor(losses).std(unbiased=False))
        mean_diag = float(torch.tensor(epoch_metrics["mean_diag"]).mean()) if epoch_metrics["mean_diag"] else 0.0
        mean_abs_offdiag = float(torch.tensor(epoch_metrics["mean_abs_offdiag"]).mean()) if epoch_metrics["mean_abs_offdiag"] else 0.0
        diag_loss = float(torch.tensor(epoch_metrics["diag_loss"]).mean()) if epoch_metrics["diag_loss"] else 0.0
        off_diag_loss = float(torch.tensor(epoch_metrics["off_diag_loss"]).mean()) if epoch_metrics["off_diag_loss"] else 0.0

        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "effective_rank": effective_rank,
                "mean_diag": mean_diag,
                "mean_abs_offdiag": mean_abs_offdiag,
                "diag_loss": diag_loss,
                "off_diag_loss": off_diag_loss,
                "cov_eigenvalues": eigvals.tolist(),
            }
        )

        print(
            f"Epoch {epoch}: mean_loss={mean_loss:.6f}, std_loss={std_loss:.6f}, "
            f"mean_diag={mean_diag:.4f}, mean_abs_offdiag={mean_abs_offdiag:.4f}, "
            f"effective_rank={effective_rank:.2f}"
        )

        if (epoch % args.save_embeddings_every == 0) or (epoch == args.epochs - 1):
            np.savez_compressed(
                OUTPUT_DIR / f"embeddings_epoch_{epoch}.npz",
                embeddings=epoch_embeddings.numpy(),
                labels=epoch_labels,
            )
            print(f"Saved epoch embeddings to outputs/embeddings_epoch_{epoch}.npz")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "subset_size": args.subset_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "method": args.method,
    }
    torch.save(checkpoint, OUTPUT_DIR / "checkpoint.pt")
    with open(OUTPUT_DIR / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoint and history to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
