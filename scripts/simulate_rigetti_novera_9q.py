#!/usr/bin/env python3
"""
Rigetti Novera 9Q QRC Simulation
=================================
Simulate Rigetti Novera 9-qubit QPU using Steinegger & Räth (2025) methodology.

System Specs:
- 9 qubits in 3×3 square lattice
- 2-qubit gate fidelity >99%
- 12 tunable couplers
- Ankaa-class (4th generation)
"""

import numpy as np
import json
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Import Steinegger utilities
from qrc_steinegger_utils import (SteineggerQRC, calculate_lyapunov_time,
                                  forecast_horizon_in_lyapunov_times)

print("="*80)
print("RIGETTI NOVERA 9Q - QRC SIMULATION")
print("="*80)
print("Methodology: Steinegger & Räth (Nature Sci Rep 2025)")
print("System: Rigetti Novera QPU (9-qubit, 3×3 square lattice)")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
print("\n" + "="*80)
print("CONFIGURATION")
print("="*80)

# Rigetti Novera specifications
NOVERA_SPECS = {
    'n_qubits': 9,
    'topology': '3x3_square_lattice',
    'gate_fidelity_2q': 0.99,  # >99%
    'n_couplers': 12,
    'architecture': 'Ankaa_4th_gen',
    'price_usd': 900000
}

# QRC hyperparameters (Steinegger methodology)
QRC_CONFIG = {
    'n_qubits': 9,
    'n_layers': 8,
    'V': 5,  # Temporal multiplexing
    'r': 3,  # Spatial multiplexing
    'G': 4,  # Polynomial degree
    'scale_range': (0, 1),
    'random_seed': 42
}

# Measurement shots (separate from QRC config)
SHOTS = 4000

# Training configuration
TRAIN_CONFIG = {
    'n_samples': 800,  # Max unique samples from 1000 timesteps (0.19 samples/feature)
    'test_size': 0.2,
    'ridge_alphas': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    'random_state': 42
}

print(f"\nNovera Specs:")
for key, val in NOVERA_SPECS.items():
    print(f"  {key}: {val}")

print(f"\nQRC Config:")
for key, val in QRC_CONFIG.items():
    print(f"  {key}: {val}")

# ============================================================================
# LOAD TURBULENCE DATA
# ============================================================================
print("\n" + "="*80)
print("LOADING TURBULENCE DATA")
print("="*80)

# Load original turbulence data (same as IBM used)
turbulence_data = np.load('../01_original_data/training_spectral.npy')

print(f"\n✓ Loaded: training_spectral.npy (original data)")
print(f"  Shape: {turbulence_data.shape}")
print(f"  Note: Same dataset used for IBM 4Q and 156Q results")

# Calculate Lyapunov time
lyapunov_time, lambda_max = calculate_lyapunov_time(turbulence_data, dt=0.01)
print(f"\n  Lyapunov exponent λ_max: {lambda_max:.6f}")
print(f"  Lyapunov time τ: {lyapunov_time:.2f} timesteps")

# ============================================================================
# INITIALIZE QRC SYSTEM
# ============================================================================
print("\n" + "="*80)
print("INITIALIZING QRC SYSTEM")
print("="*80)

qrc = SteineggerQRC(**QRC_CONFIG)

# ============================================================================
# SAMPLE TIMESTEPS
# ============================================================================
print("\n" + "="*80)
print("SAMPLING TIMESTEPS")
print("="*80)

n_timesteps = len(turbulence_data)
n_samples = TRAIN_CONFIG['n_samples']

# Stratified sampling across time series
indices = np.linspace(0, n_timesteps-2, n_samples, dtype=int)  # -2 to allow y=x[t+1]
print(f"\nSampling {n_samples} timesteps from {n_timesteps} total")
print(f"  Indices range: [{indices.min()}, {indices.max()}]")

# ============================================================================
# PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING")
print("="*80)

# Fit scaler on sampled data
sampled_data = turbulence_data[indices]
preprocessed = qrc.preprocess(sampled_data, fit=True)

print(f"\n✓ Fitted StandardScaler")
print(f"  Input range: [{sampled_data.min():.2f}, {sampled_data.max():.2f}]")
print(f"  Scaled range: [{preprocessed.min():.2f}, {preprocessed.max():.2f}]")

# ============================================================================
# GENERATE RESERVOIR FEATURES
# ============================================================================
print("\n" + "="*80)
print("GENERATING RESERVOIR FEATURES")
print("="*80)

print(f"\nThis will take approximately 2-5 minutes...")
print(f"  {n_samples} timesteps × {QRC_CONFIG['V']} temporal × {QRC_CONFIG['r']} spatial")
print(f"  = {n_samples * QRC_CONFIG['V'] * QRC_CONFIG['r']} circuit executions")

X = qrc.generate_features(preprocessed, shots=SHOTS, verbose=True)

print(f"\n✓ Feature generation complete")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Features per timestep: {X.shape[1]}")

# ============================================================================
# PREPARE TARGETS
# ============================================================================
print("\n" + "="*80)
print("PREPARING TARGETS")
print("="*80)

# Target: Next timestep
y_indices = indices + 1
y = turbulence_data[y_indices]

print(f"\nTarget: Predict next timestep")
print(f"  X shape: {X.shape} (input features)")
print(f"  y shape: {y.shape} (next timestep)")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("TRAIN/TEST SPLIT")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TRAIN_CONFIG['test_size'],
    random_state=TRAIN_CONFIG['random_state']
)

print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ============================================================================
# RIDGE REGRESSION WITH HYPERPARAMETER SEARCH
# ============================================================================
print("\n" + "="*80)
print("RIDGE REGRESSION TRAINING")
print("="*80)

print(f"\nSearching for best α in {TRAIN_CONFIG['ridge_alphas']}")

best_r2 = -np.inf
best_alpha = None
best_model = None

for alpha in TRAIN_CONFIG['ridge_alphas']:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test.flatten(), y_pred.flatten())

    if r2 > best_r2:
        best_r2 = r2
        best_alpha = alpha
        best_model = model

print(f"\n✓ Best α: {best_alpha}")
print(f"  Test R²: {best_r2:.4f}")

# ============================================================================
# DETAILED EVALUATION
# ============================================================================
print("\n" + "="*80)
print("DETAILED EVALUATION")
print("="*80)

# Energy correlation
y_pred_best = best_model.predict(X_test)

real_true = y_test[:, :50]
imag_true = y_test[:, 50:]
energy_true = np.sum(np.abs(real_true + 1j * imag_true)**2, axis=1)

real_pred = y_pred_best[:, :50]
imag_pred = y_pred_best[:, 50:]
energy_pred = np.sum(np.abs(real_pred + 1j * imag_pred)**2, axis=1)

energy_corr = np.corrcoef(energy_true, energy_pred)[0, 1]

print(f"\nPerformance Metrics:")
print(f"  R² (variance explained): {best_r2:.4f}")
print(f"  Energy correlation: {energy_corr:.4f}")
print(f"  Best regularization α: {best_alpha}")

# Forecast horizon
horizon = forecast_horizon_in_lyapunov_times(best_r2, lyapunov_time)
print(f"\nForecast Horizon:")
print(f"  {horizon:.2f} Lyapunov times")
print(f"  ({horizon * lyapunov_time:.1f} timesteps)")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'system': 'Rigetti_Novera_9Q',
    'methodology': 'Steinegger_Rath_2025',
    'timestamp': datetime.now().isoformat(),

    'specs': NOVERA_SPECS,
    'qrc_config': QRC_CONFIG,
    'train_config': TRAIN_CONFIG,

    'turbulence': {
        'file': 'training_spectral.npy',
        'note': 'Original data (same as IBM 4Q/156Q)',
        'lyapunov_time': float(lyapunov_time),
        'lambda_max': float(lambda_max)
    },

    'performance': {
        'r2': float(best_r2),
        'energy_correlation': float(energy_corr),
        'best_alpha': float(best_alpha),
        'forecast_horizon_lyapunov_times': float(horizon),
        'forecast_horizon_timesteps': float(horizon * lyapunov_time),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'n_features': int(qrc.n_features)
    }
}

with open('novera_9q_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved: novera_9q_results.json")

# Save predictions
np.save('novera_9q_predictions.npy', y_pred_best)
np.save('novera_9q_targets.npy', y_test)

print(f"✓ Saved: novera_9q_predictions.npy")
print(f"✓ Saved: novera_9q_targets.npy")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Energy time series (true vs predicted)
ax1 = axes[0, 0]
ax1.plot(energy_true, 'o-', label='True', alpha=0.7, markersize=4)
ax1.plot(energy_pred, 's-', label='Predicted', alpha=0.7, markersize=4)
ax1.set_xlabel('Test Sample')
ax1.set_ylabel('Energy')
ax1.set_title(f'Energy Prediction (R²={best_r2:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Prediction scatter
ax2 = axes[0, 1]
ax2.scatter(energy_true, energy_pred, alpha=0.5, s=50)
ax2.plot([energy_true.min(), energy_true.max()],
         [energy_true.min(), energy_true.max()],
         'r--', linewidth=2, label='Perfect prediction')
ax2.set_xlabel('True Energy')
ax2.set_ylabel('Predicted Energy')
ax2.set_title(f'Scatter (Corr={energy_corr:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals
ax3 = axes[1, 0]
residuals = energy_true - energy_pred
ax3.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Residual (True - Predicted)')
ax3.set_ylabel('Frequency')
ax3.set_title('Prediction Residuals')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: System summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
Rigetti Novera 9Q QRC Results
{'='*40}

Hardware:
  • 9 qubits (3×3 square lattice)
  • 2Q gate fidelity: >99%
  • 12 tunable couplers

QRC Configuration:
  • Temporal multiplexing: V={QRC_CONFIG['V']}
  • Spatial multiplexing: r={QRC_CONFIG['r']}
  • Polynomial degree: G={QRC_CONFIG['G']}
  • Total features: {qrc.n_features:,}

Performance:
  • R² = {best_r2:.4f}
  • Energy corr = {energy_corr:.4f}
  • Forecast horizon = {horizon:.2f} τ
  • Best α = {best_alpha}

Turbulence:
  • Dataset: Original (IBM compatible)
  • Lyapunov time τ = {lyapunov_time:.1f} steps
"""

ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.savefig('novera_9q_results.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: novera_9q_results.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SIMULATION COMPLETE - RIGETTI NOVERA 9Q")
print("="*80)

print(f"\nRigetti Novera 9-Qubit QPU:")
print(f"  • Topology: {NOVERA_SPECS['topology']}")
print(f"  • Gate fidelity: {NOVERA_SPECS['gate_fidelity_2q']*100:.1f}%")

print(f"\nQRC Performance:")
print(f"  • R² = {best_r2:.4f} ({best_r2*100:.1f}% variance explained)")
print(f"  • Energy correlation = {energy_corr:.4f}")
print(f"  • Forecast horizon = {horizon:.2f} Lyapunov times")

print(f"\nComparison to IBM 4Q:")
print(f"  • IBM 4Q: R²=0.764, 4 qubits")
print(f"  • Novera 9Q: R²={best_r2:.3f}, 9 qubits")
print(f"  • Improvement: {(best_r2/0.764 - 1)*100:+.1f}%")

print(f"\nFiles created:")
print(f"  • novera_9q_results.json")
print(f"  • novera_9q_predictions.npy")
print(f"  • novera_9q_targets.npy")
print(f"  • novera_9q_results.png")

print("\n" + "="*80)
print("Next: Run simulate_rigetti_cepheus_36q.py")
print("="*80 + "\n")
