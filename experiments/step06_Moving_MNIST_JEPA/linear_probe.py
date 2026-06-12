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
    lr = 1e-3 # Lower LR for stable training

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
        targets_list = {
            "digit_ids": [],
            "x": [],
            "y": [],
            "vx": [],
            "vy": [],
            "joint": [],
        }
        for x, y, state in tqdm(loader, desc="Extracting"):
            print(f"Original x shape: {x.shape}")
            x = x.squeeze(1).to(device)
            print(f"Shape before encoder_x: {x.shape}")
            with torch.no_grad():
                s_x = model.encoder_x(x)
            s_x_list.append(s_x.cpu())
            
            targets_list["digit_ids"].append(state["digit_ids"])
            
            pos = state["positions"].float()
            vel = state["velocities"].float()
            
            targets_list["x"].append(pos[:, :, 0])
            targets_list["y"].append(pos[:, :, 1])
            targets_list["vx"].append(vel[:, :, 0])
            targets_list["vy"].append(vel[:, :, 1])
            
            joint = torch.cat([
                pos[:, 0, 0:1], pos[:, 0, 1:2], vel[:, 0, 0:1], vel[:, 0, 1:2],
                pos[:, 1, 0:1], pos[:, 1, 1:2], vel[:, 1, 0:1], vel[:, 1, 1:2]
            ], dim=1)
            targets_list["joint"].append(joint)

        s_x_tensor = torch.cat(s_x_list, dim=0)
        return s_x_tensor, {k: torch.cat(v, dim=0) for k, v in targets_list.items()}

    s_x_train, targets_train = extract(train_loader)
    s_x_val, targets_val = extract(val_loader)

    print(f"Extracted train size: {s_x_train.shape[0]}, val size: {s_x_val.shape[0]}")

    # Normalize regression targets
    regression_keys = ["x", "y", "vx", "vy", "joint"]
    norm_stats = {}
    for k in regression_keys:
        mean = targets_train[k].mean(dim=0)
        std = targets_train[k].std(dim=0).clamp(min=1e-6)
        norm_stats[k] = {"mean": mean, "std": std}
        targets_train[k] = (targets_train[k] - mean) / std
        targets_val[k] = (targets_val[k] - mean) / std

    class LinearProbeTrain:
        def __init__(self, name, probe, criterion, is_classification=False):
            self.name = name
            self.probe = probe.to(device)
            self.criterion = criterion
            self.is_classification = is_classification
            self.optimizer = optim.Adam(self.probe.parameters(), lr=lr)
            
        def train(self, target_key):
            print(f"\nTraining Probe: {self.name}")
            y_tr = targets_train[target_key]
            y_va = targets_val[target_key]
            x_tr = s_x_train
            x_va = s_x_val
            
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
                    
                    if self.is_classification:
                        pred = pred.view(-1, 2, 10).permute(0, 2, 1)
                        loss = self.criterion(pred, batch_y.long())
                    else:
                        loss = self.criterion(pred, batch_y)
                        
                    loss.backward()
                    self.optimizer.step()
                
                if epoch % 5 == 0 or epoch == epochs - 1:
                    self.probe.eval()
                    with torch.no_grad():
                        val_pred = self.probe(x_va.to(device))
                        if self.is_classification:
                            val_pred_loss = val_pred.view(-1, 2, 10).permute(0, 2, 1)
                            val_loss = self.criterion(val_pred_loss, y_va.to(device).long()).item()
                        else:
                            val_loss = self.criterion(val_pred, y_va.to(device)).item()
                            
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_probe_state = self.probe.state_dict().copy()
            
            # Load best model for final evaluation
            self.probe.load_state_dict(best_probe_state)
            self.probe.eval()
            with torch.no_grad():
                val_pred = self.probe(x_va.to(device))
                if self.is_classification:
                    val_pred_loss = val_pred.view(-1, 2, 10).permute(0, 2, 1)
                    val_loss = self.criterion(val_pred_loss, y_va.to(device).long()).item()
                    val_pred_classes = val_pred_loss.argmax(dim=1)
                    metric = (val_pred_classes == y_va.to(device).long()).float().mean().item()
                    print(f"Final | Best Val Loss: {val_loss:.4f} | Val Acc: {metric:.4f}")
                else:
                    val_loss = self.criterion(val_pred, y_va.to(device)).item()
                    
                    # Denormalize
                    mean = norm_stats[target_key]["mean"].to(device)
                    std = norm_stats[target_key]["std"].to(device)
                    
                    val_pred_orig = val_pred * std + mean
                    y_va_orig = y_va.to(device) * std + mean
                    
                    print(f"Final | Best Val Loss (normalized): {val_loss:.4f}")
                    
                    # Compute per-target metrics
                    num_targets = y_va_orig.shape[1]
                    target_names = [f"dim_{i}" for i in range(num_targets)]
                    if target_key == "joint":
                        target_names = ["x1", "y1", "vx1", "vy1", "x2", "y2", "vx2", "vy2"]
                    elif target_key == "x":
                        target_names = ["x1", "x2"]
                    elif target_key == "y":
                        target_names = ["y1", "y2"]
                    elif target_key == "vx":
                        target_names = ["vx1", "vx2"]
                    elif target_key == "vy":
                        target_names = ["vy1", "vy2"]
                    
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

    probes = [
        LinearProbeTrain("Digit Identity", nn.Linear(embedding_dim, 20), nn.CrossEntropyLoss(), is_classification=True),
        LinearProbeTrain("X Position", nn.Linear(embedding_dim, 2), nn.MSELoss()),
        LinearProbeTrain("Y Position", nn.Linear(embedding_dim, 2), nn.MSELoss()),
        LinearProbeTrain("VX Velocity", nn.Linear(embedding_dim, 2), nn.MSELoss()),
        LinearProbeTrain("VY Velocity", nn.Linear(embedding_dim, 2), nn.MSELoss()),
        LinearProbeTrain("Joint [x, y, vx, vy]", nn.Linear(embedding_dim, 8), nn.MSELoss()),
    ]
    
    target_keys = ["digit_ids", "x", "y", "vx", "vy", "joint"]
    
    all_results = {"RMSE": {}, "R2": {}}
    for probe, key in zip(probes, target_keys):
        res = probe.train(key)
        if res is not None:
            if key == "joint":
                for k in res["RMSE"]:
                    all_results["RMSE"][k] = res["RMSE"][k]
                for k in res["R2"]:
                    all_results["R2"][k] = res["R2"][k]
                    
    out_path = Path(__file__).resolve().parent / "outputs" / "linear_probe_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
