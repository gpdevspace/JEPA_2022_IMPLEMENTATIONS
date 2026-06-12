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


class MovingMNISTDataset(Dataset):
    """Synthetic Moving MNIST: 2 digits bouncing on a 64×64 canvas with random velocities."""

    def __init__(
        self,
        n_sequences: int = 1000,
        seq_length: int = 10,
        image_size: int = 64,       # Canvas must be larger than digit size (28)
        digit_size: int = 28,       # MNIST digit size (fixed)
        num_digits: int = 2,
        seed: int = 42,
        device: str = "cpu",
        return_state: bool = False,
    ):
        self.n_sequences = n_sequences
        self.seq_length = seq_length
        self.image_size = image_size
        self.digit_size = digit_size
        self.num_digits = num_digits
        self.seed = seed
        self.device = device
        self.return_state = return_state

        assert image_size > digit_size, \
            f"image_size ({image_size}) must be larger than digit_size ({digit_size})"

        mnist_data = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.mnist_digits = {}
        for i in range(10):
            digit_images = [img for img, lbl in mnist_data if lbl == i]
            self.mnist_digits[i] = torch.cat(
                [img.unsqueeze(0) for img in digit_images[:100]], dim=0
            )

        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Stacked 4 frames [t-3, t-2, t-1, t] shape (4, H, W)
            y: Single future frame [t+1]               shape (1, H, W)
        """
        seq, states = self._generate_sequence(idx)  # seq: (seq_length, 1, H, W)

        x = torch.cat([seq[0], seq[1], seq[2], seq[3]], dim=0)  # (4, H, W)
        y = seq[4]                                                # (1, H, W)

        x = x.unsqueeze(0)  # (1, 4, H, W)
        y = y.unsqueeze(0)  # (1, 1, H, W)

        if self.return_state:
            # We want the state at the final input frame, which is t=3
            state = states[3]
            return x, y, state

        return x, y

    def _generate_sequence(self, idx: int) -> tuple[torch.Tensor, list[dict]]:
        """Generate one sequence of bouncing digits on a larger canvas."""
        rng = np.random.default_rng(self.seed + idx)

        max_pos = self.image_size - self.digit_size   # e.g. 64 - 28 = 36

        # Random digit IDs
        digit_ids = rng.integers(0, 10, size=self.num_digits)

        # Initial positions within valid range, and velocities
        positions  = [rng.uniform(0, max_pos, size=2) for _ in range(self.num_digits)]
        velocities = [rng.uniform(-4, 4, size=2) for _ in range(self.num_digits)]

        frames = []
        states = []
        for t in range(self.seq_length):
            canvas = torch.zeros(1, self.image_size, self.image_size)

            for i, digit_id in enumerate(digit_ids):
                # --- Bounce physics: update position, reflect off walls ---
                positions[i]  = positions[i] + velocities[i]

                for axis in range(2):
                    if positions[i][axis] < 0:
                        positions[i][axis]  = -positions[i][axis]        # reflect
                        velocities[i][axis] = -velocities[i][axis]
                    elif positions[i][axis] > max_pos:
                        positions[i][axis]  = 2 * max_pos - positions[i][axis]
                        velocities[i][axis] = -velocities[i][axis]

                # Pick a random sample of this digit class
                digit_idx = rng.integers(0, len(self.mnist_digits[digit_id]))
                digit_img = self.mnist_digits[digit_id][digit_idx]  # (1, 28, 28)

                # Composite onto canvas
                y0 = int(positions[i][1])
                x0 = int(positions[i][0])
                canvas[:, y0:y0 + self.digit_size, x0:x0 + self.digit_size] = torch.maximum(
                    canvas[:, y0:y0 + self.digit_size, x0:x0 + self.digit_size],
                    digit_img,
                )

            brightness = rng.uniform(0.8, 1.0)
            canvas = torch.clamp(canvas * brightness, 0, 1)
            frames.append(canvas)
            states.append({
                "positions": np.copy(positions),
                "velocities": np.copy(velocities),
                "digit_ids": np.copy(digit_ids)
            })

        return torch.stack(frames, dim=0), states  # (seq_length, 1, H, W), list of states


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
