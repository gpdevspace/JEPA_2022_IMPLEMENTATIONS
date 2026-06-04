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
    """Synthetic Moving MNIST: 2 digits bouncing on 28×28 canvas with random velocities."""

    def __init__(
        self,
        n_sequences: int = 1000,
        seq_length: int = 10,
        image_size: int = 28,
        num_digits: int = 2,
        seed: int = 42,
        device: str = "cpu",
    ):
        """
        Args:
            n_sequences: Number of video sequences to generate
            seq_length: Frames per sequence
            image_size: Canvas size (28x28 for MNIST)
            num_digits: Number of moving digits per sequence
            seed: Random seed
            device: 'cpu' or 'mps' (for on-the-fly generation)
        """
        self.n_sequences = n_sequences
        self.seq_length = seq_length
        self.image_size = image_size
        self.num_digits = num_digits
        self.seed = seed
        self.device = device

        # Load MNIST digits once
        mnist_data = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        # Extract unique digits: shape (10, n_per_digit, 1, 28, 28)
        self.mnist_digits = {}
        for i in range(10):
            digit_images = [img for img, lbl in mnist_data if lbl == i]
            self.mnist_digits[i] = torch.cat([img.unsqueeze(0) for img in digit_images[:100]], dim=0)

        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Stacked 4 frames [t-3, t-2, t-1, t] (4, 28, 28) normalized
            y: Single future frame [t+1] (1, 28, 28) normalized
        """
        # Generate a single sequence
        seq = self._generate_sequence(idx)  # (seq_length, 1, 28, 28)

        # Extract frames: input is t-3:t+1, output is t+1
        # Indices: 0=t-3, 1=t-2, 2=t-1, 3=t, 4=t+1
        x = torch.cat([seq[0], seq[1], seq[2], seq[3]], dim=0)  # (4, 28, 28)
        y = seq[4]  # (1, 28, 28)

        # Reshape x and y to have the same number of channels
        x = x.unsqueeze(0)  # (1, 4, 28, 28)
        y = y.unsqueeze(0)  # (1, 1, 28, 28)

        return x, y

    def _generate_sequence(self, idx: int) -> torch.Tensor:
        """Generate one sequence of moving digits."""
        rng = np.random.default_rng(self.seed + idx)

        # Initialize canvas
        frames = []
        positions = []
        velocities = []

        # Random digit IDs and starting positions/velocities
        digit_ids = rng.integers(0, 10, size=self.num_digits)
        for _ in range(self.num_digits):
            pos = rng.uniform(0, self.image_size - 28, size=2)
            vel = rng.uniform(-2, 2, size=2)
            positions.append(pos)
            velocities.append(vel)

        # Generate frames
        for t in range(self.seq_length):
            canvas = torch.zeros(1, self.image_size, self.image_size)

            # Move and render each digit
            for i, digit_id in enumerate(digit_ids):
                # Update position
                pos = np.array(positions[i]) + np.array(velocities[i]) * t
                pos = np.clip(pos, 0, self.image_size - 28)

                # Get random MNIST digit image
                digit_idx = rng.integers(0, len(self.mnist_digits[digit_id]))
                digit_img = self.mnist_digits[digit_id][digit_idx]  # (1, 28, 28)

                # Composite onto canvas
                y_start, x_start = int(pos[1]), int(pos[0])
                y_end, x_end = y_start + 28, x_start + 28
                canvas[:, y_start:y_end, x_start:x_end] = torch.maximum(
                    canvas[:, y_start:y_end, x_start:x_end],
                    digit_img,
                )

            # Add brightness/contrast variation (augmentation)
            brightness = rng.uniform(0.8, 1.0)
            canvas = torch.clamp(canvas * brightness, 0, 1)

            frames.append(canvas)

        return torch.stack(frames, dim=0)  # (seq_length, 1, 28, 28)


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
