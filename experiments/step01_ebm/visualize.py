"""Visualize Step 1 EBM energy landscape and training collapse."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step01_ebm.model import EBM
from shared.data import make_dataset
from shared.device import get_device
from shared.viz import plot_energy_heatmap, plot_energy_surface, plot_training_curve

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
GRID_RES = 80
Y_RANGE = 3.0


def energy_grid(
    model: EBM,
    x_fixed: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    lin = np.linspace(-Y_RANGE, Y_RANGE, GRID_RES, dtype=np.float32)
    yy, xx = np.meshgrid(lin, lin)
    grid_y = np.stack([xx, yy], axis=-1)
    flat_y = torch.from_numpy(grid_y.reshape(-1, 2)).to(device)
    x_rep = x_fixed.expand(flat_y.shape[0], -1)
    with torch.no_grad():
        energies = model(x_rep, flat_y).cpu().numpy()
    return grid_y, energies.reshape(GRID_RES, GRID_RES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Step 1 EBM")
    parser.add_argument("--x-index", type=int, default=0, help="Training sample index for fixed x")
    args = parser.parse_args()

    ckpt_path = OUTPUT_DIR / "checkpoint.pt"
    hist_path = OUTPUT_DIR / "loss_history.json"
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint at {ckpt_path}. Run train.py first.")

    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = EBM().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset_name = ckpt.get("dataset", "swiss_roll")
    seed = ckpt.get("seed", 42)
    x_all, y_all = make_dataset(name=dataset_name, seed=seed)
    idx = min(args.x_index, len(x_all) - 1)
    x_fixed = x_all[idx : idx + 1].to(device)

    grid_y, energies = energy_grid(model, x_fixed, device)
    manifold = y_all.numpy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    title_suffix = f"fixed x (sample {idx}, {dataset_name})"

    plot_energy_heatmap(
        grid_y,
        energies,
        manifold,
        OUTPUT_DIR / "energy_heatmap.png",
        f"Energy landscape — {title_suffix}",
    )
    plot_energy_heatmap(
        grid_y,
        energies - energies.mean(),
        manifold,
        OUTPUT_DIR / "energy_deviation_heatmap.png",
        f"Deviation from mean energy — {title_suffix}",
    )
    plot_energy_surface(
        grid_y,
        energies,
        OUTPUT_DIR / "energy_surface.png",
        f"Energy surface — {title_suffix}",
    )

    if hist_path.exists():
        with open(hist_path, encoding="utf-8") as f:
            history = json.load(f)
        plot_training_curve(
            [h["epoch"] for h in history],
            [h["mean_energy"] for h in history],
            [h["std_energy"] for h in history],
            OUTPUT_DIR / "training_curve.png",
        )

    e_min, e_max = float(energies.min()), float(energies.max())
    e_std = float(energies.std())
    e_mean = float(energies.mean())
    cv = e_std / max(abs(e_mean), 1e-8)
    print(f"Grid energy range: [{e_min:.4e}, {e_max:.4e}], std={e_std:.4e}, CV={cv:.4f}")
    print(
        "Collapse: naive training drives F(x,y) down without a floor — "
        "energies diverge to large negative values (Step 2 contrastive fixes this)."
    )
    if cv < 0.15:
        print("Landscape is nearly flat relative to |mean| (flat minimum basin).")

    if device.type == "mps":
        torch.mps.empty_cache()

    print(f"Wrote figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
