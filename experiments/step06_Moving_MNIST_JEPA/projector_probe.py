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
    hidden_dim = 512
    image_size = 64
    batch_size = 256
    epochs = 150 # Probes can train longer, they are fast
    lr = 1e-3 # Lower LR for stable training

    model = JEPAModel(embedding_dim=embedding_dim, hidden_dim=hidden_dim, image_size=image_size)
    checkpoint_path = Path(__file__).resolve().parent / "outputs" / "checkpoint_jepa.pt"
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}. Please train the JEPA model first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Step 1: Freeze the entire JEPA
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
        z_x_list = []
        targets_list = {
            "joint": [],
        }
        for x, y, state in tqdm(loader, desc="Extracting"):
            x = x.squeeze(1).to(device)
            # Step 2: Extract z_x instead of s_x
            with torch.no_grad():
                s_x = model.encoder_x(x)
                z_x = model.projector(s_x)
            z_x_list.append(z_x.cpu())
            
            pos = state["positions"].float()
            vel = state["velocities"].float()
            
            joint = torch.cat([
                pos[:, 0, 0:1], pos[:, 0, 1:2], vel[:, 0, 0:1], vel[:, 0, 1:2],
                pos[:, 1, 0:1], pos[:, 1, 1:2], vel[:, 1, 0:1], vel[:, 1, 1:2]
            ], dim=1)
            targets_list["joint"].append(joint)

        z_x_tensor = torch.cat(z_x_list, dim=0)
        return z_x_tensor, {k: torch.cat(v, dim=0) for k, v in targets_list.items()}

    z_x_train, targets_train = extract(train_loader)
    z_x_val, targets_val = extract(val_loader)

    print(f"Extracted train size: {z_x_train.shape[0]}, val size: {z_x_val.shape[0]}")

    # Normalize regression targets
    regression_keys = ["joint"]
    norm_stats = {}
    for k in regression_keys:
        mean = targets_train[k].mean(dim=0)
        std = targets_train[k].std(dim=0).clamp(min=1e-6)
        norm_stats[k] = {"mean": mean, "std": std}
        targets_train[k] = (targets_train[k] - mean) / std
        targets_val[k] = (targets_val[k] - mean) / std

    class LinearProbeTrain:
        def __init__(self, name, probe, criterion):
            self.name = name
            self.probe = probe.to(device)
            self.criterion = criterion
            # Step 4: Train only the probe
            self.optimizer = optim.Adam(self.probe.parameters(), lr=lr)
            
        def train(self, target_key):
            print(f"\nTraining Probe: {self.name}")
            y_tr = targets_train[target_key]
            y_va = targets_val[target_key]
            x_tr = z_x_train
            x_va = z_x_val
            
            probe_train_dataset = torch.utils.data.TensorDataset(x_tr, y_tr)
            probe_train_loader = DataLoader(probe_train_dataset, batch_size=batch_size, shuffle=True)
            
            best_val_loss = float('inf')
            best_probe_state = None
            
            for epoch in range(epochs):
                self.probe.train()
                for batch_x, batch_y in probe_train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    self.optimizer.zero_grad()
                    pred = self.probe(batch_x)
                    
                    loss = self.criterion(pred, batch_y)
                        
                    loss.backward()
                    self.optimizer.step()
                
                if epoch % 5 == 0 or epoch == epochs - 1:
                    self.probe.eval()
                    with torch.no_grad():
                        val_pred = self.probe(x_va.to(device))
                        val_loss = self.criterion(val_pred, y_va.to(device)).item()
                            
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_probe_state = self.probe.state_dict().copy()
            
            # Load best model for final evaluation
            self.probe.load_state_dict(best_probe_state)
            self.probe.eval()
            with torch.no_grad():
                val_pred = self.probe(x_va.to(device))
                val_loss = self.criterion(val_pred, y_va.to(device)).item()
                
                # Denormalize
                mean = norm_stats[target_key]["mean"].to(device)
                std = norm_stats[target_key]["std"].to(device)
                
                val_pred_orig = val_pred * std + mean
                y_va_orig = y_va.to(device) * std + mean
                
                print(f"Final | Best Val Loss (normalized): {val_loss:.4f}")
                
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
                    
                return results

    # Determine projector_dim directly from extracted embeddings
    projector_dim = z_x_train.shape[-1]
    
    # Step 3: Build a single linear probe
    probes = [
        LinearProbeTrain("Joint [x, y, vx, vy]", nn.Linear(projector_dim, 8), nn.MSELoss()),
    ]
    
    target_keys = ["joint"]
    
    all_results = {"RMSE": {}, "R2": {}}
    for probe, key in zip(probes, target_keys):
        res = probe.train(key)
        if res is not None:
            if key == "joint":
                for k in res["RMSE"]:
                    all_results["RMSE"][k] = res["RMSE"][k]
                for k in res["R2"]:
                    all_results["R2"][k] = res["R2"][k]
                    
    out_path = Path(__file__).resolve().parent / "outputs" / "projector_probe_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
