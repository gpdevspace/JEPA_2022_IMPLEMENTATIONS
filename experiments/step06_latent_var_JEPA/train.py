"""Step 5: JEPA training on Moving MNIST."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step06_latent_var_JEPA.model import JEPAModel, GenerativeModel
from shared.data import MovingMNISTDataset
from shared.device import get_device

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Canvas size used by MovingMNISTDataset — must match model constructors
IMAGE_SIZE = 64


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def train_jepa(
    model: JEPAModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> dict[str, list[float]]:
    """Train JEPA model with auxiliary reconstruction loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss":   [],
        "val_eff_rank": [],
    }

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_losses: list[float] = []

        for x, y in tqdm(train_loader, desc=f"JEPA Train Epoch {epoch+1}/{epochs}"):
            # __getitem__ returns x: (1, 4, H, W) and y: (1, 1, H, W)
            # The leading "1" is a spurious channel dim — squeeze it out.
            x = x.squeeze(1).to(device)   # (batch, 4, 64, 64)
            y = y.squeeze(1).to(device)   # (batch, 1, 64, 64)

            optimizer.zero_grad()
            s_x, s_y, s_y_pred, loss, y_recon, recon_loss = model(x, y, return_recon=True)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_losses: list[float] = []
        all_s_x, all_s_y, all_s_y_pred = [], [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.squeeze(1).to(device)   # (batch, 4, 64, 64)
                y = y.squeeze(1).to(device)   # (batch, 1, 64, 64)

                s_x, s_y, s_y_pred, loss, _, _ = model(x, y, return_recon=True)
                val_losses.append(loss.item())

                all_s_x.append(s_x.cpu())
                all_s_y.append(s_y.cpu())
                all_s_y_pred.append(s_y_pred.cpu())

        # ── Metrics ──────────────────────────────────────────────────────────
        s_x_all       = torch.cat(all_s_x,      dim=0)
        s_y_all       = torch.cat(all_s_y,      dim=0)
        s_y_pred_all  = torch.cat(all_s_y_pred, dim=0)
        metrics = model.compute_metrics(s_x_all, s_y_all, s_y_pred_all)

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_eff_rank"].append(metrics["effective_rank"])

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"Eff Rank: {metrics['effective_rank']:.2f}"
        )

    return history


def train_generative(
    model: GenerativeModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> dict[str, list[float]]:
    """Train generative (pixel-space) model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_losses: list[float] = []

        for x, y in tqdm(train_loader, desc=f"Gen Train Epoch {epoch+1}/{epochs}"):
            # Generative model only uses y; still squeeze the spurious dim.
            y = y.squeeze(1).to(device)   # (batch, 1, 64, 64)

            optimizer.zero_grad()
            y_recon, loss = model(y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_losses: list[float] = []

        with torch.no_grad():
            for x, y in val_loader:
                y = y.squeeze(1).to(device)   # (batch, 1, 64, 64)
                y_recon, loss = model(y)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Step 5 JEPA on Moving MNIST")
    parser.add_argument(
        "--model",
        choices=["jepa", "generative"],
        default="jepa",
        help="Which model to train",
    )
    parser.add_argument("--quick",         action="store_true",  help="Quick mode: small dataset, few epochs")
    parser.add_argument("--epochs",        type=int,   default=30,    help="Number of epochs")
    parser.add_argument("--batch-size",    type=int,   default=128,   help="Batch size")
    parser.add_argument("--lr",            type=float, default=1e-3,  help="Learning rate")
    parser.add_argument("--seed",          type=int,   default=42,    help="Random seed")
    parser.add_argument("--embedding-dim", type=int,   default=128,   help="Embedding dimension")
    parser.add_argument("--hidden-dim",    type=int,   default=512,   help="Hidden dimension for predictor")
    parser.add_argument("--workers",       type=int,   default=2,     help="Data loader workers")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    set_seed(args.seed)

    # ── Dataset sizes ─────────────────────────────────────────────────────────
    if args.quick:
        n_train, n_val, epochs = 100, 20, 3
        print("Quick mode: 100 train, 20 val, 3 epochs (~2 min)")
    else:
        n_train, n_val, epochs = 1000, 200, args.epochs
        print(f"Full mode: {n_train} train, {n_val} val, {epochs} epochs")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Generating Moving MNIST datasets...")
    train_dataset = MovingMNISTDataset(
        n_sequences=n_train,
        seq_length=5,           # frames: t-3, t-2, t-1, t, t+1
        image_size=IMAGE_SIZE,  # 64×64 canvas (digits are 28×28)
        num_digits=2,
        seed=args.seed,
        device=str(device),
    )
    val_dataset = MovingMNISTDataset(
        n_sequences=n_val,
        seq_length=5,
        image_size=IMAGE_SIZE,
        num_digits=2,
        seed=args.seed + 1,
        device=str(device),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(str(device) != "cpu"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(str(device) != "cpu"),
    )

    # ── Model + training ──────────────────────────────────────────────────────
    if args.model == "jepa":
        print("Training JEPA model...")
        model = JEPAModel(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            image_size=IMAGE_SIZE,
        )
        history = train_jepa(model, train_loader, val_loader, epochs=epochs, device=device, lr=args.lr)
        checkpoint_path = OUTPUT_DIR / "checkpoint_jepa.pt"
        history_path    = OUTPUT_DIR / "loss_history_jepa.json"
    else:
        print("Training Generative model...")
        model = GenerativeModel(
            embedding_dim=args.embedding_dim,
            image_size=IMAGE_SIZE,
        )
        history = train_generative(model, train_loader, val_loader, epochs=epochs, device=device, lr=args.lr)
        checkpoint_path = OUTPUT_DIR / "checkpoint_generative.pt"
        history_path    = OUTPUT_DIR / "loss_history_generative.json"

    # ── Save artefacts ────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved loss history to {history_path}")


if __name__ == "__main__":
    main()
