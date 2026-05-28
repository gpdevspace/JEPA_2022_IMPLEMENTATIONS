"""Step 2: Contrastive EBM training (hinge, InfoNCE, logistic)"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step02_contrastive_ebm.model import EBM
from shared.data import make_dataset
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


def contrastive_loss(
    model, x, y, negatives, loss_type="hinge", margin=1.0
):
    pos_energy = model(x, y)
    neg_energy = model(x, negatives)
    if loss_type == "hinge":
        return torch.relu(pos_energy - neg_energy + margin).mean()
    elif loss_type == "nce":
        # InfoNCE: F(x,y) + log sum exp(-F(x,ŷ))
        logits = -neg_energy
        return (pos_energy + torch.logsumexp(logits, dim=0)).mean()
    elif loss_type == "logistic":
        return torch.log1p(torch.exp(pos_energy - neg_energy)).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def sample_negatives(y_all, yb, strategy="random"):
    if strategy == "random":
        idx = torch.randint(0, y_all.shape[0], (yb.shape[0],), device=yb.device)
        return y_all[idx]
    if strategy == "in_batch":
        perm = torch.randperm(yb.shape[0], device=yb.device)
        neg = yb[perm]
        if (perm == torch.arange(yb.shape[0], device=yb.device)).any():
            perm = (perm + 1) % yb.shape[0]
            neg = yb[perm]
        return neg
    raise ValueError(f"Unknown negative strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Step 2 Contrastive EBM")
    parser.add_argument("--dataset", choices=["swiss_roll", "circles"], default="swiss_roll")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-type", choices=["hinge", "nce", "logistic"], default="hinge")
    parser.add_argument("--negative-strategy", choices=["random", "in_batch"], default="random")
    args = parser.parse_args()

    device = get_device()
    if device.type == "cpu":
        print("Warning: MPS not available; training on CPU.")
    print(f"Using device: {device}")

    set_seed(args.seed)
    x_all, y_all = make_dataset(name=args.dataset, seed=args.seed)
    loader = DataLoader(
        TensorDataset(x_all, y_all),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = EBM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    OUTPUT_DIR.mkdir(exist_ok=True)
    history = []

    for epoch in tqdm(range(args.epochs)):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            negatives = sample_negatives(y_all.to(device), yb, strategy=args.negative_strategy)
            loss = contrastive_loss(model, xb, yb, negatives, loss_type=args.loss_type)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        losses_tensor = torch.tensor(losses, dtype=torch.float32)
        mean_loss = float(losses_tensor.mean())
        std_loss = float(losses_tensor.std(unbiased=False))
        history.append({"epoch": epoch, "mean_loss": mean_loss, "std_loss": std_loss})
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: mean_loss={mean_loss:.6f}, std_loss={std_loss:.6f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "loss_type": args.loss_type,
    }
    torch.save(checkpoint, OUTPUT_DIR / "checkpoint.pt")
    with open(OUTPUT_DIR / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoint and history to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
