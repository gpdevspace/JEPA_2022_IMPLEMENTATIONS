import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import matplotlib.pyplot as plt

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

        return torch.stack(frames, dim=0)  # (seq_length, 1, image_size, image_size)


# Function to visualize the generated sequence
def visualize_sequence(sequence):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        if i < sequence.shape[1]:
            ax.imshow(sequence[0, i].squeeze(), cmap='gray')
        else:
            ax.imshow(torch.zeros_like(sequence[0, 0]).squeeze(), cmap='gray')
        ax.axis('off')
    plt.show()

# Main execution

if __name__ == "__main__":
    # Create the dataset and dataloader
    dataset = MovingMNISTDataset(seq_length=10)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Get a sequence from the dataset
    for i, (x, y) in enumerate(dataloader):
        if i == 0:  # Display only the first sequence
            print(f"Shape of x: {x.shape}")  # Should be (1, 4, 28, 28)
            print(f"Shape of y: {y.shape}")  # Should be (1, 1, 28, 28)

            # Concatenate x and y along dimension 1
            full_sequence = torch.cat([x, y], dim=1)  # (1, 5, 28, 28)
            visualize_sequence(full_sequence)
            break