"""Step 3: Joint Embedding Architecture training on CIFAR-10."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step03_joint_embedding.model import (
    JointEmbeddingModel,
    collapse_distance_loss,
    info_nce_loss,
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
    parser = argparse.ArgumentParser(description="Train Step 3 Joint Embedding Architecture")
    parser.add_argument("--root", type=Path, default=Path("./data"))
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--method", choices=["distance", "info_nce"], default="info_nce")
    parser.add_argument("--workers", type=int, default=2)
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
        for x1, x2, _ in tqdm(loader, desc=f"Epoch {epoch}"):
            x1, x2 = x1.to(device), x2.to(device)
            z1 = model(x1)
            z2 = model(x2)
            if args.method == "distance":
                # Use both positive and negative batch relationships.
                # Collapse happens because the loss minimizes all pairwise distances.
                loss = collapse_distance_loss(z1, z2)
            else:
                loss = info_nce_loss(z1, z2, temperature=args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        mean_loss = float(torch.tensor(losses).mean())
        std_loss = float(torch.tensor(losses).std(unbiased=False))
        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "std_loss": std_loss,
            }
        )
        print(f"Epoch {epoch}: mean_loss={mean_loss:.6f}, std_loss={std_loss:.6f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "subset_size": args.subset_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "temperature": args.temperature,
        "method": args.method,
    }
    checkpoint_path = OUTPUT_DIR / f"checkpoint_{args.method}.pt"
    history_path = OUTPUT_DIR / f"loss_history_{args.method}.json"
    torch.save(checkpoint, checkpoint_path)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoint to {checkpoint_path} and history to {history_path}")
    print(f"Saved checkpoint and history to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
