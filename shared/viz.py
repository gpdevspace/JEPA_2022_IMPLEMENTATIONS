from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_energy_heatmap(
    grid_y: np.ndarray,
    energies: np.ndarray,
    manifold_y: np.ndarray | None,
    out_path: Path,
    title: str,
) -> None:
    """grid_y: (H, W, 2), energies: (H, W)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        energies,
        origin="lower",
        extent=[grid_y[:, :, 0].min(), grid_y[:, :, 0].max(),
                grid_y[:, :, 1].min(), grid_y[:, :, 1].max()],
        aspect="auto",
        cmap="viridis",
    )
    if manifold_y is not None:
        ax.scatter(
            manifold_y[:, 0],
            manifold_y[:, 1],
            s=4,
            c="white",
            alpha=0.35,
            linewidths=0,
        )
    ax.set_xlabel("y₁")
    ax.set_ylabel("y₂")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="F(x, y)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_energy_surface(
    grid_y: np.ndarray,
    energies: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        grid_y[:, :, 0],
        grid_y[:, :, 1],
        energies,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
    )
    ax.set_xlabel("y₁")
    ax.set_ylabel("y₂")
    ax.set_zlabel("F(x, y)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curve(
    epochs: list[int],
    mean_energy: list[float],
    std_energy: list[float],
    out_path: Path,
    title: str = "Training curve",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, mean_energy, label="mean F(x, y)")
    mean = np.array(mean_energy)
    std = np.array(std_energy)
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.25, label="±1 std (batch)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("energy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
