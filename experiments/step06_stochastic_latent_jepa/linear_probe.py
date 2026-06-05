"""Linear probe script to evaluate representation learning quality of JEPA."""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.step05_jepa.model import JEPA
from shared.data import CIFAR10PairDataset
from shared.device import get_device

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def plot_probe_curves(history: list[dict], out_path: Path) -> None:
    """Plot training vs testing loss and accuracy curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    test_loss = [h["test_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    test_acc = [h["test_acc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Linear Probe Training Progress on CIFAR-10", fontsize=16, fontweight="bold")

    # Loss curves
    axes[0].plot(epochs, train_loss, label="Train Loss", color="#1f77b4", marker="o")
    axes[0].plot(epochs, test_loss, label="Test Loss", color="#ff7f0e", marker="o")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=12)
    axes[0].set_title("Loss Curves", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, train_acc, label="Train Accuracy", color="#2ca02c", marker="o")
    axes[1].plot(epochs, test_acc, label="Test Accuracy", color="#d62728", marker="o")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Accuracy Curves", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {out_path}")


def plot_confusion_matrix(all_targets: list, all_preds: list, out_path: Path) -> None:
    """Plot confusion matrix of the predictions."""
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    cm = confusion_matrix(all_targets, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title("Confusion Matrix on CIFAR-10 Test Set", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probe evaluation for Step 5 JEPA")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUT_DIR / "checkpoint.pt",
        help="Path to the checkpoint file containing trained weights",
    )
    parser.add_argument("--root", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=10, help="Number of linear probe training epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate for linear probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    set_seed(args.seed)

    # Load checkpoint
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Model definition & weights load
    encoder_dim = checkpoint.get("encoder_dim", 1024)
    predictor_hidden = checkpoint.get("predictor_hidden", 2048)
    
    jepa_model = JEPA(
        encoder_dim=encoder_dim,
        predictor_hidden=predictor_hidden,
        shared_encoder=True,
    ).to(device)
    
    jepa_model.load_state_dict(checkpoint["model_state_dict"])
    jepa_model.eval()

    # Freeze encoder
    for param in jepa_model.parameters():
        param.requires_grad = False

    # Datasets with eval_mode=True (no augmentations)
    train_dataset = CIFAR10PairDataset(
        root=args.root,
        train=True,
        eval_mode=True,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=(args.workers > 0),
    )

    test_dataset = CIFAR10PairDataset(
        root=args.root,
        train=False,
        eval_mode=True,
        seed=args.seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=(args.workers > 0),
    )

    # Linear classifier
    classifier = nn.Linear(encoder_dim, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []

    print(f"\nStarting Linear Probe Training ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        # Train
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, _, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            x = x.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                features = jepa_model.embed(x)

            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # Evaluate
        classifier.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for x, _, labels in tqdm(test_loader, desc=f"Epoch {epoch} [Test]"):
                x = x.to(device)
                labels = labels.to(device)

                features = jepa_model.embed(x)
                outputs = classifier(features)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * x.size(0)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        epoch_test_loss = test_loss / test_total
        epoch_test_acc = 100.0 * test_correct / test_total

        print(
            f"Epoch {epoch:2d}: "
            f"Train Loss = {epoch_train_loss:.4f}, Train Acc = {epoch_train_acc:.2f}% | "
            f"Test Loss = {epoch_test_loss:.4f}, Test Acc = {epoch_test_acc:.2f}%"
        )

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "test_loss": epoch_test_loss,
            "test_acc": epoch_test_acc,
        })

    # Save probe results
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    results_path = OUTPUT_DIR / "linear_probe_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nLinear probe finished. Results saved to {results_path}")
    print(f"Final Test Accuracy: {history[-1]['test_acc']:.2f}%")

    # Final evaluation for confusion matrix
    print("\nComputing predictions for confusion matrix...")
    classifier.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, _, labels in tqdm(test_loader, desc="Collecting predictions"):
            x = x.to(device)
            features = jepa_model.embed(x)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(labels.tolist())

    plot_probe_curves(history, OUTPUT_DIR / "linear_probe_curves.png")
    plot_confusion_matrix(all_targets, all_preds, OUTPUT_DIR / "linear_probe_confusion_matrix.png")


if __name__ == "__main__":
    main()
