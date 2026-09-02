import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SPECTRAL_FILE = "training_spectral.npy"
RESERVOIR_FILE = "reservoir_states_hero_156q.npy"
INDICES_FILE = "hero_sample_indices.npy"

print("Loading Hero Data...")
try:
    spectral_data = np.load(SPECTRAL_FILE)
    reservoir_states = np.load(RESERVOIR_FILE)
    indices = np.load(INDICES_FILE)
    print(f"✓ Inputs: {spectral_data.shape}")
    print(f"✓ Reservoir: {reservoir_states.shape}")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# --- 1. PREPARE DATA ---
# We predict state t+1 from reservoir state t
# Filter out the last index if it was sampled
valid_mask = indices < (len(spectral_data) - 1)
indices = indices[valid_mask]
reservoir_states = reservoir_states[valid_mask]

# Targets: The next timestep of the 50 Fourier Modes
targets = spectral_data[indices + 1]

# Train/Test Split (Small dataset, so we use Leave-One-Out or simple split)
# For this pilot with 10 samples, we train on 8, test on 2
split = int(len(reservoir_states) * 0.8)
X_train, X_test = reservoir_states[:split], reservoir_states[split:]
y_train, y_test = targets[:split], targets[split:]

# --- 2. TRAIN READOUT (Ridge Regression) ---
print(f"\nTraining Readout Layer on {len(X_train)} samples...")
# Alpha (regularization) helps with quantum noise
model = Ridge(alpha=0.1) 
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"✓ Training Complete.")
print(f"  Global R² Score: {r2:.4f}")

# --- 3. PHYSICS VALIDATION: ENERGY CASCADE ---
# Calculate Energy Spectrum E(k) = |u_k|^2
# We compare the Spectrum of the True Turbulence vs. Quantum Prediction

def get_energy_spectrum(fourier_modes):
    # fourier_modes shape: (n_samples, 50)
    # Simple 1D spectrum proxy: average magnitude squared per mode
    # Mode index k roughly corresponds to array index
    energy = np.mean(np.abs(fourier_modes)**2, axis=0)
    return energy

E_true = get_energy_spectrum(y_test)
E_pred = get_energy_spectrum(y_pred)

# Calculate Slope (log-log) for the inertial range (e.g., modes 5-20)
k_vals = np.arange(1, len(E_true) + 1)
start_k, end_k = 5, 25 # Inertial range
slope_true = np.polyfit(np.log(k_vals[start_k:end_k]), np.log(E_true[start_k:end_k]), 1)[0]
slope_pred = np.polyfit(np.log(k_vals[start_k:end_k]), np.log(E_pred[start_k:end_k]), 1)[0]

print(f"\n=== PHYSICS VERIFICATION ===")
print(f"Inertial Range Slope (Theory: ~ -3.0)")
print(f"  True Turbulence: {slope_true:.2f}")
print(f"  Quantum Predicted: {slope_pred:.2f}")

if abs(slope_pred - slope_true) < 0.5:
    print("✅ SUCCESS: Quantum Reservoir captured the Energy Cascade!")
else:
    print("⚠ WARNING: Cascade slope deviation. Noise might be high.")

# --- 4. PLOT & SAVE ---
plt.figure(figsize=(10, 6))
plt.loglog(k_vals, E_true, 'k-', linewidth=2, label='Ground Truth (Navier-Stokes)')
plt.loglog(k_vals, E_pred, 'r--', linewidth=2, label='156-Qubit Prediction')
plt.xlabel('Wavenumber k')
plt.ylabel('Energy E(k)')
plt.title(f'Quantum Prediction of Turbulent Energy Cascade\nIBM Heron (156q) - R²={r2:.2f}')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.savefig("hero_energy_cascade.png")
print(f"\n✓ Plot saved to 'hero_energy_cascade.png'")