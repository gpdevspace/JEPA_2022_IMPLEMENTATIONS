"""Train JEPA on CIFAR-10 with pluggable loss functions to demonstrate collapse and recovery."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step06_latent_var_jepa.model import (
    JEPA,
    compute_embedding_statistics,
    get_loss_function,
)
from shared.data import CIFAR10PairDataset
from shared.device import get_device

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Step 6 Latent Variable JEPA with pluggable loss")
    parser.add_argument("--root", type=Path, default=Path("./data"))
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loss",
        choices=["mse", "vicreg"],
        default="vicreg",
        help="Loss function: mse (will collapse) or vicreg (prevents collapse)",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-embeddings-every", type=int, default=10)
    parser.add_argument("--encoder-dim", type=int, default=1024)
    parser.add_argument("--predictor-hidden", type=int, default=2048)
    parser.add_argument("--latent-dim", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    if device.type == "cpu":
        print("Warning: MPS not available; training on CPU.")
    print(f"Using device: {device}")
    print(f"Loss function: {args.loss}")

    set_seed(args.seed)

    # Dataset
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

    # Model
    model = JEPA(
        encoder_dim=args.encoder_dim,
        predictor_hidden=args.predictor_hidden,
        shared_encoder=True,
        latent_dim=16,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss function
    loss_fn = get_loss_function(args.loss)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    history = []

    print(f"\nTraining configuration:")
    print(f"  Dataset: CIFAR-10 (subset={args.subset_size})")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Loss: {args.loss}")
    print(f"  Encoder dim: {args.encoder_dim}")
    print(f"  Predictor hidden: {args.predictor_hidden}")

    for epoch in range(args.epochs):
        model.train()
        losses = []
        loss_metrics = {
            "invariance": [],
            "variance_loss": [],
            "covariance_loss": [],
            "latent_penalty": [],
        }

        # Collect embeddings for collapse detection
        all_s_x = []
        all_s_y = []
        all_s_y_pred = []
        all_z_x = []
        all_labels = []

        for x, y, labels in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs-1}"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            s_x, s_y, z_x, s_y_pred = model(x, y)

            # Compute loss
            loss, metrics = loss_fn(s_y, s_y_pred, z_x)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            losses.append(loss.item())
            for key in ["invariance", "variance_loss", "covariance_loss", "latent_penalty"]:
                loss_metrics[key].append(metrics[key])

            # Collect embeddings (detached to save memory)
            with torch.no_grad():
                all_s_x.append(s_x.detach().cpu())
                all_s_y.append(s_y.detach().cpu())
                all_s_y_pred.append(s_y_pred.detach().cpu())
                all_z_x.append(z_x.detach().cpu())
                all_labels.append(labels.numpy())

        # Concatenate embeddings
        all_s_x = torch.cat(all_s_x, dim=0)
        all_s_y = torch.cat(all_s_y, dim=0)
        all_s_y_pred = torch.cat(all_s_y_pred, dim=0)
        all_z_x = torch.cat(all_z_x, dim=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute collapse statistics
        s_x_stats = compute_embedding_statistics(all_s_x)
        s_y_stats = compute_embedding_statistics(all_s_y)
        s_y_pred_stats = compute_embedding_statistics(all_s_y_pred)
        z_x_stats = compute_embedding_statistics(all_z_x)

        # Aggregate metrics
        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses))
        mean_invariance = float(np.mean(loss_metrics["invariance"]))
        mean_variance_loss = float(np.mean(loss_metrics["variance_loss"]))
        mean_covariance_loss = float(np.mean(loss_metrics["covariance_loss"]))
        mean_latent_penalty = float(np.mean(loss_metrics["latent_penalty"]))

        # Log epoch
        epoch_data = {
            "epoch": epoch,
            "loss": args.loss,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "invariance": mean_invariance,
            "variance_loss": mean_variance_loss,
            "covariance_loss": mean_covariance_loss,
            "latent_penalty": mean_latent_penalty,
            "s_x_mean_std": s_x_stats["mean_std"],
            "s_x_min_std": s_x_stats["min_std"],
            "s_x_effective_rank": s_x_stats["effective_rank"],
            "s_x_max_abs_mean": s_x_stats["max_abs_mean"],
            "s_y_mean_std": s_y_stats["mean_std"],
            "s_y_min_std": s_y_stats["min_std"],
            "s_y_effective_rank": s_y_stats["effective_rank"],
            "s_y_max_abs_mean": s_y_stats["max_abs_mean"],
            "s_y_pred_mean_std": s_y_pred_stats["mean_std"],
            "s_y_pred_min_std": s_y_pred_stats["min_std"],
            "s_y_pred_effective_rank": s_y_pred_stats["effective_rank"],
            "s_y_pred_max_abs_mean": s_y_pred_stats["max_abs_mean"],
            "z_x_mean_std": z_x_stats["mean_std"],
            "z_x_min_std": z_x_stats["min_std"],
            "z_x_effective_rank": z_x_stats["effective_rank"],
            "z_x_max_abs_mean": z_x_stats["max_abs_mean"],
        }
        history.append(epoch_data)

        # Print progress
        print(
            f"Epoch {epoch:3d}: loss={mean_loss:.6f} ± {std_loss:.6f} | "
            f"s_y_pred: std={s_y_pred_stats['mean_std']:.4f}, "
            f"rank={s_y_pred_stats['effective_rank']:.1f}/{args.encoder_dim} | "
            f"z_x: std={z_x_stats['mean_std']:.4f}, "
            f"rank={z_x_stats['effective_rank']:.1f}/{args.latent_dim} | "
            f"invar={mean_invariance:.6f} | "
            f"latent_penalty={mean_latent_penalty:.6f}"
        )

        # Save embeddings periodically
        if (epoch % args.save_embeddings_every == 0) or (epoch == args.epochs - 1):
            np.savez_compressed(
                OUTPUT_DIR / f"embeddings_epoch_{epoch}.npz",
                s_x=all_s_x.numpy(),
                s_y=all_s_y.numpy(),
                s_y_pred=all_s_y_pred.numpy(),
                z_x=all_z_x.numpy(),
                labels=all_labels,
            )
            print(f"  Saved embeddings to outputs/embeddings_epoch_{epoch}.npz")

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "subset_size": args.subset_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "loss": args.loss,
        "encoder_dim": args.encoder_dim,
        "predictor_hidden": args.predictor_hidden,
        "latent_dim": args.latent_dim,
    }
    torch.save(checkpoint, OUTPUT_DIR / "checkpoint.pt")

    # Save training history
    with open(OUTPUT_DIR / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Saved checkpoint and history to {OUTPUT_DIR}")

    # Print collapse summary
    final_stats = history[-1]
    print(f"\nFinal collapse metrics ({args.loss}):")
    print(f"  s_y_pred mean_std: {final_stats['s_y_pred_mean_std']:.4f} (higher is better)")
    print(f"  s_y_pred min_std: {final_stats['s_y_pred_min_std']:.4f} (near zero = dead dims)")
    print(f"  s_y_pred effective_rank: {final_stats['s_y_pred_effective_rank']:.1f}/{args.encoder_dim} (low = collapse)")
    print(f"  s_y_pred max_abs_mean: {final_stats['s_y_pred_max_abs_mean']:.4f} (high = bias)")
    print(f"  z_x mean_std: {final_stats['z_x_mean_std']:.4f} (higher is better)")
    print(f"  z_x min_std: {final_stats['z_x_min_std']:.4f} (near zero = dead dims)")
    print(f"  z_x effective_rank: {final_stats['z_x_effective_rank']:.1f}/{args.latent_dim} (low = collapse)")
    print(f"  z_x max_abs_mean: {final_stats['z_x_max_abs_mean']:.4f} (high = bias)")

    if args.loss == "mse":
        if final_stats["s_y_pred_effective_rank"] < args.encoder_dim * 0.1:
            print("\n  ⚠️  COLLAPSE DETECTED: Effective rank is very low.")
        if final_stats["s_y_pred_min_std"] < 0.01:
            print("  ⚠️  COLLAPSE DETECTED: Minimum std is near zero (dead dimensions).")
    elif args.loss == "vicreg":
        if final_stats["s_y_pred_effective_rank"] > args.encoder_dim * 0.5:
            print("\n  ✓ VICReg successfully prevented collapse.")


if __name__ == "__main__":
    main()
