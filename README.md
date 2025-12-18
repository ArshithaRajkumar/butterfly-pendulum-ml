# The Butterfly Effect: Generative AI & Physics-Informed Failure Prediction

## 🚀 Project Overview
This repository explores three advanced AI paradigms—**Variational Autoencoders (VAEs)**, **Transformers**, and **Physics-Informed Neural Networks (PINNs)**—to solve a complex time-series problem: **Detecting and diagnosing a "Butterfly Effect" on a pendulum.**

### The Scenario
1.  **Normal State:** A pendulum swings with a frequency of **3.0 Hz**.
2.  **The Anomaly:** A "butterfly" lands on the pendulum, altering the mass and changing the frequency to **2.0 Hz**.
3.  **The Challenge:** * Detect *if* the system is broken.
    * Detect *when* it broke.
    * Diagnose *how* the physics changed.

## 🧠 Methods Implemented

### 1. Global Anomaly Detection (VAE)
* **File:** `vae_anomaly.py`
* **Technique:** Uses a **Variational Autoencoder** to learn the latent distribution of normal swings.
* **Outcome:** High reconstruction error serves as a global flag for system failure.

### 2. Change-Point Detection (Transformer)
* **File:** `transformer_detect.py`
* **Technique:** Uses a **Self-Attention Transformer** to model sequential dependencies.
* **Outcome:** Detects the exact timestamp (Step 50) where the butterfly lands by identifying breaks in the temporal sequence.

### 3. Parameter Discovery (PINN)
* **File:** `pinn_physics.py`
* **Technique:** Uses a **Physics-Informed Neural Network** to solve the inverse problem using the Harmonic Oscillator equation:
    $$\frac{d^2x}{dt^2} + \omega^2 x = 0$$
* **Outcome:** Automatically discovers that the frequency parameter ($\omega$) shifted from 3.0 to 2.0, confirming a physical change in the system.

## 🛠️ How to Run
Prerequisites:
```bash
pip install torch numpy matplotlib