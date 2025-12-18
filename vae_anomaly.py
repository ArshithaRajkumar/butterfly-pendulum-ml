"""
vae_anomaly.py
--------------
This script demonstrates the use of a Variational Autoencoder (VAE) for 
Anomaly Detection.

Scenario:
    - Normal Data: A standard pendulum swinging at 3.0 Hz.
    - Anomaly: A 'Butterfly' lands, changing frequency to 2.0 Hz.
    - Goal: Detect the anomaly via Reconstruction Error.

Model: 
    - Architecture: Linear Encoder -> Latent Space (Gaussian) -> Linear Decoder
    - Loss: MSE (Reconstruction) + KL Divergence (Regularization)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1. Data Simulation
# ===========================
def generate_pendulum_data(n_samples=1000, seq_len=50, frequency=3.0):
    data = []
    for _ in range(n_samples):
        t_start = np.random.rand() * 10
        t = np.linspace(t_start, t_start + 4, seq_len)
        # Signal = Amplitude * cos(freq * t) + noise
        signal = np.cos(frequency * t) + 0.05 * np.random.randn(seq_len)
        data.append(signal)
    return torch.tensor(np.array(data, dtype=np.float32))

# ===========================
# 2. VAE Model Definition
# ===========================
class PendulumVAE(nn.Module):
    def __init__(self, seq_len=50, latent_dim=4):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, seq_len)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# ===========================
# 3. Execution Block
# ===========================
if __name__ == "__main__":
    # A. Generate Data
    print("Generating synthetic pendulum data...")
    normal_data = generate_pendulum_data(n_samples=2000, frequency=3.0)
    butterfly_data = generate_pendulum_data(n_samples=10, frequency=2.0) # Anomaly

    train_loader = torch.utils.data.DataLoader(normal_data, batch_size=32, shuffle=True)

    # B. Train Model
    model = PendulumVAE()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Training VAE on 'Normal' data...")
    for epoch in range(30):
        for batch in train_loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            
            # Loss Calculation
            recon_loss = F.mse_loss(recon, batch, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld_loss
            
            loss.backward()
            optimizer.step()

    # C. Evaluate
    print("Evaluating Anomaly Detection...")
    model.eval()
    
    # Test on one normal and one butterfly sample
    sample_norm = normal_data[0:1]
    sample_anom = butterfly_data[0:1]
    
    with torch.no_grad():
        rec_norm, _, _ = model(sample_norm)
        rec_anom, _, _ = model(sample_anom)
        
        err_norm = F.mse_loss(rec_norm, sample_norm).item()
        err_anom = F.mse_loss(rec_anom, sample_anom).item()

    print(f"Reconstruction Error (Normal):   {err_norm:.4f}")
    print(f"Reconstruction Error (Anomaly):  {err_anom:.4f}")
    
    # D. Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.plot(sample_norm[0], label='Input')
    plt.plot(rec_norm[0], '--', label='Recon')
    plt.title(f"Normal (Error: {err_norm:.2f})")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(sample_anom[0], label='Butterfly')
    plt.plot(rec_anom[0], '--', label='Recon')
    plt.title(f"Anomaly (Error: {err_anom:.2f})")
    plt.legend()
    plt.show()