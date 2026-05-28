"""Visualize Step 2 Contrastive EBM energy landscape and training."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step02_contrastive_ebm.model import EBM
from shared.data import make_dataset
from shared.device import get_device
from shared.viz import plot_energy_heatmap, plot_energy_surface, plot_training_curve

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
GRID_RES = 80
Y_RANGE = 3.0

def energy_grid(model: EBM, x_fixed: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    y_grid = np.linspace(-Y_RANGE, Y_RANGE, GRID_RES)
    yy1, yy2 = np.meshgrid(y_grid, y_grid)
    y_points = np.stack([yy1.ravel(), yy2.ravel()], axis=-1)
    y_tensor = torch.tensor(y_points, dtype=torch.float32, device=device)
    x_rep = x_fixed.repeat(y_tensor.shape[0], 1)
    with torch.no_grad():
        energies = model(x_rep, y_tensor).cpu().numpy()
    return y_points, energies

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Step 2 Contrastive EBM")
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


    # Reshape for heatmap/surface
    y_grid = y_points.reshape(GRID_RES, GRID_RES, 2)
    energies_grid = energies.reshape(GRID_RES, GRID_RES)

    # Get manifold points for overlay (true y for all x)
    manifold_y = y_all.cpu().numpy() if hasattr(y_all, 'cpu') else y_all

    plot_energy_heatmap(
        y_grid,
        energies_grid,
        manifold_y,
        OUTPUT_DIR / "energy_heatmap.png",
        f"Step 2: Energy Heatmap (x_idx={idx})"
    )
    plot_energy_surface(
        y_grid,
        energies_grid,
        OUTPUT_DIR / "energy_surface.png",
        f"Step 2: Energy Surface (x_idx={idx})"
    )

    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    plot_training_curve(
        [h["epoch"] for h in history],
        [h["mean_loss"] for h in history],
        None,
        OUTPUT_DIR / "training_curve.png",
    )
    print("Saved visualizations to outputs/")

if __name__ == "__main__":
    main()
