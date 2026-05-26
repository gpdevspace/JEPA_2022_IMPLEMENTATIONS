import numpy as np
import torch


def swiss_roll_pairs(
    n_samples: int = 10_000,
    noise: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Positive pairs (x, y): x is normalized arc-length in [0, 1], y is R^2 on the roll.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    y1 = t * np.cos(t)
    y2 = t * np.sin(t)
    y = np.stack([y1, y2], axis=1).astype(np.float32)
    y += rng.normal(0.0, noise, size=y.shape).astype(np.float32)

    # Normalize arc-length parameter to [0, 1]
    t_sorted = np.sort(t)
    ranks = np.searchsorted(t_sorted, t, side="left")
    x = (ranks / max(n_samples - 1, 1)).astype(np.float32).reshape(-1, 1)

    return torch.from_numpy(x), torch.from_numpy(y)


def concentric_circles_pairs(
    n_samples: int = 10_000,
    noise: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive pairs on two concentric circles; x indexes angle in [0, 1]."""
    rng = np.random.default_rng(seed)
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    def circle(n: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        y = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
        y += rng.normal(0.0, noise, size=y.shape)
        x = (theta / (2.0 * np.pi)).astype(np.float32).reshape(-1, 1)
        return x, y.astype(np.float32)

    x_in, y_in = circle(n_inner, 1.0)
    x_out, y_out = circle(n_outer, 2.0)
    x = np.vstack([x_in, x_out])
    y = np.vstack([y_in, y_out])
    perm = rng.permutation(n_samples)
    return torch.from_numpy(x[perm]), torch.from_numpy(y[perm])


def make_dataset(
    name: str = "swiss_roll",
    n_samples: int = 10_000,
    noise: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "swiss_roll":
        return swiss_roll_pairs(n_samples=n_samples, noise=noise, seed=seed)
    if name == "circles":
        return concentric_circles_pairs(n_samples=n_samples, noise=noise, seed=seed)
    raise ValueError(f"Unknown dataset: {name!r}. Use 'swiss_roll' or 'circles'.")
