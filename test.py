import torch

# Sample input tensors for demonstration
z1 = torch.tensor([[0.5, 0.8],
                   [0.3, 0.7],
                   [0.6, 0.9]])

z2 = torch.tensor([[0.4, 0.82],
                   [0.29, 0.71],
                   [0.61, 0.89]])

# Hyperparameter lambda
lamb = 0.005

# Compute the cross-correlation matrix C
C = torch.matmul(z1.T, z2) / z1.shape[0]
print("Cross-Correlation Matrix (C):\n", C)

# Extract diagonal and off-diagonal elements
diag = torch.diagonal(C)
off_diag = C - torch.diag_embed(diag)
print("Diagonal Elements (diag):\n", diag)
print("Off-Diagonal Elements (off_diag):\n", off_diag)

# Compute the loss components
diagonal_loss = torch.mean((diag - 1) ** 2)
off_diagonal_loss = lamb * torch.mean(off_diag ** 2)
print("Diagonal Loss Component:", diagonal_loss.item())
print("Off-Diagonal Loss Component:", off_diagonal_loss.item())

# Total loss
loss = diagonal_loss + off_diagonal_loss
print("Total Barlow Twins Loss:", loss.item())