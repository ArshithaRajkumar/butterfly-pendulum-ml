"""
transformer_detect.py
---------------------
This script uses a Transformer model to detect exactly WHEN a failure occurs.

Scenario:
    - A continuous timeline where physics changes halfway through.
    - Model: Uses Self-Attention to predict sequence steps.
    - Result: Instantaneous error spikes at the moment of change.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1. Positional Encoding
# ===========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ===========================
# 2. Transformer Model
# ===========================
class PendulumTransformer(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.input_linear = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer(src)
        return self.output_linear(output)

# ===========================
# 3. Execution Block
# ===========================
if __name__ == "__main__":
    # A. Generate Training Data (Normal Only)
    t = np.linspace(0, 4, 50)
    # Generate 1000 normal sequences
    train_data = []
    for _ in range(1000):
        signal = np.cos(3.0 * t) + 0.05 * np.random.randn(50)
        train_data.append(signal)
    
    train_tensor = torch.tensor(np.array(train_data), dtype=torch.float32).unsqueeze(-1) # [N, 50, 1]
    
    # B. Train Model
    model = PendulumTransformer()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("Training Transformer on Normal Sequences...")
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = criterion(output, train_tensor)
        loss.backward()
        optimizer.step()

    # C. Create "Transient" Test Scenario
    # 0-50: Normal | 50-100: Butterfly (Freq=2.0) | 100-150: Normal
    seq1 = np.cos(3.0 * np.linspace(0, 4, 50))
    seq2 = np.cos(2.0 * np.linspace(0, 4, 50)) # Butterfly
    seq3 = np.cos(3.0 * np.linspace(0, 4, 50))
    
    full_signal = np.concatenate([seq1, seq2, seq3])
    test_tensor = torch.tensor(full_signal, dtype=torch.float32).view(1, -1, 1) # [1, 150, 1]

    # D. Detect
    model.eval()
    with torch.no_grad():
        # Process in chunks (simplification for demo)
        pred1 = model(test_tensor[:, 0:50, :])
        pred2 = model(test_tensor[:, 50:100, :])
        pred3 = model(test_tensor[:, 100:150, :])
        full_pred = torch.cat([pred1, pred2, pred3], dim=1)
        
        # Calculate Error over time
        error_signal = torch.abs(test_tensor - full_pred).squeeze()

    # E. Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(full_signal, label='Sensor Data')
    plt.axvspan(50, 100, color='red', alpha=0.1, label='Butterfly Event')
    plt.legend()
    plt.title("Original Signal with Event")

    plt.subplot(2,1,2)
    plt.plot(error_signal, color='red', label='Reconstruction Error')
    plt.axhline(0.2, linestyle='--', color='grey')
    plt.legend()
    plt.title("Transformer Detection (Error Spike)")
    plt.tight_layout()
    plt.show()