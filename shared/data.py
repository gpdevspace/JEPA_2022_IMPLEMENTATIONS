from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset


class CIFAR10PairDataset(Dataset):
    """CIFAR-10 dataset wrapper producing two augmented views of each image."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        subset_size: int | None = None,
        seed: int = 42,
        eval_mode: bool = False,
    ):
        self.transform = self._build_transform(eval_mode)
        base_dataset = torchvision.datasets.CIFAR10(
            root=str(root),
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

        if subset_size is not None and subset_size < len(base_dataset):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(base_dataset), size=subset_size, replace=False)
            self.dataset = torch.utils.data.Subset(base_dataset, indices)
        else:
            self.dataset = base_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image, label = self.dataset[idx]
        if eval_mode := hasattr(self, "eval_mode") and self.eval_mode:
            return image, image, label
        return self._augment(image), self._augment(image), label

    def _build_transform(self, eval_mode: bool) -> transforms.Compose:
        self.eval_mode = eval_mode
        if eval_mode:
            return transforms.Compose([
                transforms.ToTensor(),
            ])
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
        ])

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            pil = transforms.ToPILImage()(image)
            return self.transform(pil)
        return self.transform(image)


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
