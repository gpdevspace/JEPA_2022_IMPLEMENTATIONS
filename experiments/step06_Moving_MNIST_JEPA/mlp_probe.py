import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
from experiments.step06_Moving_MNIST_JEPA.model import JEPAModel
from shared.data import MovingMNISTDataset
from shared.device import get_device

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    embedding_dim = 256
    image_size = 64
    batch_size = 256
    epochs = 150 # Probes can train longer, they are fast
    lr = 1e-2 # Higher LR for linear layer

    model = JEPAModel(embedding_dim=embedding_dim, hidden_dim=512, image_size=image_size)
    checkpoint_path = Path(__file__).resolve().parent / "outputs" / "checkpoint_jepa.pt"
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}. Please train the JEPA model first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    for p in model.parameters():
        p.requires_grad = False

    print("Generating probe datasets...")
    # Use larger sizes for probe training to get stable metrics
    train_dataset = MovingMNISTDataset(
        n_sequences=1000,
        seq_length=5,
        image_size=image_size,
        num_digits=2,
        seed=42,
        device=str(device),
        return_state=True
    )
    val_dataset = MovingMNISTDataset(
        n_sequences=200,
        seq_length=5,
        image_size=image_size,
        num_digits=2,
        seed=43,
        device=str(device),
        return_state=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Extracting embeddings...")
    def extract(loader):
        s_x_list = []
        targets_list = []
        for x, y, state in tqdm(loader, desc="Extracting"):
            print(f"Original x shape: {x.shape}")
            x = x.squeeze(1).to(device)
            print(f"Shape before encoder_x: {x.shape}")
            with torch.no_grad():
                s_x = model.encoder_x(x)
            s_x_list.append(s_x.cpu())
            
            pos = state["positions"].float()
            vel = state["velocities"].float()
            
            joint = torch.cat([
                pos[:, 0, 0:1], pos[:, 0, 1:2], vel[:, 0, 0:1], vel[:, 0, 1:2],
                pos[:, 1, 0:1], pos[:, 1, 1:2], vel[:, 1, 0:1], vel[:, 1, 1:2]
            ], dim=1)
            targets_list.append(joint)

        s_x_tensor = torch.cat(s_x_list, dim=0)
        return s_x_tensor, torch.cat(targets_list, dim=0)

    s_x_train, targets_train = extract(train_loader)
    s_x_val, targets_val = extract(val_loader)

    print(f"Extracted train size: {s_x_train.shape[0]}, val size: {s_x_val.shape[0]}")

    mean = targets_train.mean(dim=0)
    std = targets_train.std(dim=0).clamp(min=1e-6)

    targets_train = (targets_train - mean) / std
    targets_val = (targets_val - mean) / std

    probe = nn.Sequential(
        nn.Linear(embedding_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 8),
    ).to(device)
    
    criterion = nn.MSELoss()
    lr = 1e-3
    optimizer = optim.Adam(probe.parameters(), lr=lr)
            
    print(f"\nTraining MLP Probe on Joint State [x1, y1, vx1, vy1, x2, y2, vx2, vy2]")
    
    probe_train_dataset = torch.utils.data.TensorDataset(s_x_train, targets_train)
    probe_train_loader = DataLoader(probe_train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_probe_state = None
    
    for epoch in range(epochs):
        probe.train()
        
        for batch_x, batch_y in probe_train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = probe(batch_x)
            
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            probe.eval()
            with torch.no_grad():
                val_pred = probe(s_x_val.to(device))
                val_loss = criterion(val_pred, targets_val.to(device)).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_probe_state = probe.state_dict().copy()

    probe.load_state_dict(best_probe_state)
    probe.eval()
    with torch.no_grad():
        val_pred = probe(s_x_val.to(device))
        val_loss = criterion(val_pred, targets_val.to(device)).item()
        
        print(f"Final | Best Val Loss (normalized): {val_loss:.4f}\n")
        
        # Denormalize
        val_pred_orig = val_pred.cpu() * std + mean
        y_va_orig = targets_val * std + mean
        
        # Compute per-target metrics
        target_names = ["x1", "y1", "vx1", "vy1", "x2", "y2", "vx2", "vy2"]
        
        print(f"{'Target':<10} | {'RMSE':<10} | {'R2':<10}")
        print("-" * 35)
        
        results = {"RMSE": {}, "R2": {}}
        
        for i, name in enumerate(target_names):
            y_i = y_va_orig[:, i]
            pred_i = val_pred_orig[:, i]
            
            mse_i = nn.functional.mse_loss(pred_i, y_i).item()
            rmse_i = np.sqrt(mse_i)
            
            y_mean_i = y_i.mean()
            ss_tot_i = ((y_i - y_mean_i) ** 2).sum().item()
            ss_res_i = ((y_i - pred_i) ** 2).sum().item()
            
            r2_i = 1.0 - (ss_res_i / ss_tot_i) if ss_tot_i > 0 else 0.0
            
            results["RMSE"][name] = rmse_i
            results["R2"][name] = r2_i
            
            print(f"{name:<10} | {rmse_i:<10.4f} | {r2_i:<10.4f}")
            
        out_path = Path(__file__).resolve().parent / "outputs" / "mlp_probe_metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
