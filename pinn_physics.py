"""
pinn_physics.py
---------------
This script uses a Physics-Informed Neural Network (PINN) to discover hidden parameters.

Scenario:
    - We have noisy data of the pendulum while the butterfly is sitting on it.
    - We don't know the frequency (omega).
    - We use the differential equation: d^2x/dt^2 + w^2 * x = 0 to find 'w'.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1. PINN Model
# ===========================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard Neural Network to approximate x(t)
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
        # Learnable Parameter: Omega (Frequency)
        # Initialize at 3.0 (Normal), expecting it to drift to 2.0 (Butterfly)
        self.omega = nn.Parameter(torch.tensor([3.0]))

    def forward(self, t):
        return self.net(t)

# ===========================
# 2. Physics Loss
# ===========================
def physics_loss(model, t):
    x = model(t)
    
    # Compute derivatives using Autograd
    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]
    
    # ODE Residual: x'' + w^2 * x = 0
    residual = x_tt + (model.omega ** 2) * x
    return torch.mean(residual ** 2)

# ===========================
# 3. Execution Block
# ===========================
if __name__ == "__main__":
    # A. Generate Noisy Observation (Butterfly Data)
    # True frequency = 2.0
    t_np = np.linspace(0, 4, 100)
    x_np = np.cos(2.0 * t_np) + 0.01 * np.random.randn(100)
    
    t_tensor = torch.tensor(t_np, dtype=torch.float32).view(-1, 1).requires_grad_(True)
    x_tensor = torch.tensor(x_np, dtype=torch.float32).view(-1, 1)

    # B. Train (Discover)
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Starting Discovery. Initial Estimate: {model.omega.item():.4f}")
    history = []

    for epoch in range(2000):
        optimizer.zero_grad()
        
        # Loss 1: Data Fit
        x_pred = model(t_tensor)
        loss_data = torch.mean((x_pred - x_tensor)**2)
        
        # Loss 2: Physics Consistency
        loss_phy = physics_loss(model, t_tensor)
        
        loss = loss_data + loss_phy
        loss.backward()
        optimizer.step()
        
        history.append(model.omega.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Omega Est={model.omega.item():.4f}, Loss={loss.item():.6f}")

    print(f"Final Discovery: {model.omega.item():.4f} (True Value: 2.0)")

    # C. Plot
    plt.figure(figsize=(8, 4))
    plt.plot(history, label='Estimated Omega')
    plt.axhline(2.0, color='r', linestyle='--', label='True Value (2.0)')
    plt.title("PINN Parameter Discovery")
    plt.xlabel("Epochs")
    plt.ylabel("Frequency (Omega)")
    plt.legend()
    plt.grid(True)
    plt.show()