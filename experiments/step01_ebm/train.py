"""Step 1: naive EBM training (positive pairs only) — demonstrates energy collapse."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step01_ebm.model import EBM
from shared.data import make_dataset
from shared.device import get_device

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Step 1 EBM (naive collapse demo)")
    parser.add_argument("--dataset", choices=["swiss_roll", "circles"], default="swiss_roll")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
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

    history: list[dict[str, float]] = []
    for epoch in tqdm(range(1, args.epochs + 1), desc="epochs"):
        batch_energies: list[float] = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            energy = model(x_batch, y_batch)
            loss = energy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_energies.extend(energy.detach().cpu().tolist())

        mean_e = float(sum(batch_energies) / len(batch_energies))
        std_e = float(torch.tensor(batch_energies).std().item())
        history.append({"epoch": epoch, "mean_energy": mean_e, "std_energy": std_e})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "dataset": args.dataset,
        "epochs": args.epochs,
        "seed": args.seed,
        "final_mean_energy": history[-1]["mean_energy"],
    }
    torch.save(checkpoint, OUTPUT_DIR / "checkpoint.pt")
    with open(OUTPUT_DIR / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Final mean energy: {history[-1]['mean_energy']:.6e}")
    print(
        "Expected: energy collapses (unbounded decrease) — positive-only loss has no "
        "term to raise energy on wrong y. See visualize.py and Step 2 in the plan."
    )
    print(f"Saved checkpoint and history to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
