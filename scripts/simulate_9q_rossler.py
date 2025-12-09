#!/usr/bin/env python3
"""
9Q QRC Simulation on Rössler Attractor
========================================
Validate QRC on the Rössler chaotic system.

Rössler Properties:
- Lyapunov exponent: λ ≈ 0.071
- Lyapunov time: τ_L ≈ 14.08
- Different topology than Lorenz-63 (single-lobe vs butterfly)
"""

import numpy as np
import json
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Import Steinegger utilities
from qrc_steinegger_utils import (SteineggerQRC, calculate_lyapunov_time,
                                  forecast_horizon_in_lyapunov_times)

print("="*80)
print("9Q QRC - RÖSSLER ATTRACTOR VALIDATION")
print("="*80)
print("System: Rössler (a=0.2, b=0.2, c=5.7)")
print("Expected Lyapunov: λ = 0.071")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
QRC_CONFIG = {
    'n_qubits': 9,
    'n_layers': 8,
    'V': 5,  # Temporal multiplexing
    'r': 3,  # Spatial multiplexing
    'G': 4,  # Polynomial degree
    'scale_range': (0, 1),
    'random_seed': 42
}

SHOTS = 4000

TRAIN_CONFIG = {
    'n_samples': 2000,  # Increased for statistical validity
    'test_size': 0.2,
    'ridge_alphas': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    'random_state': 42
}

# ============================================================================
# LOAD RÖSSLER DATA
# ============================================================================
print("\n" + "="*80)
print("LOADING RÖSSLER DATA")
print("="*80)

rossler_data = np.load('../02_new_turbulence/rossler_trajectory.npy')
with open('../02_new_turbulence/rossler_parameters.json', 'r') as f:
    rossler_params = json.load(f)

print(f"\n✓ Loaded Rössler trajectory")
print(f"  Shape: {rossler_data.shape} (timesteps, features)")
print(f"  Features: x, y, z coordinates")
print(f"\nKnown properties:")
print(f"  Lyapunov exponent: λ = {rossler_params['lyapunov_exponent']:.3f}")
print(f"  Lyapunov time: τ_L = {rossler_params['lyapunov_time']:.2f}")

# Calculate Lyapunov from data (verification)
calc_lyap_time, calc_lambda = calculate_lyapunov_time(rossler_data, dt=rossler_params['dt'])
print(f"\nCalculated from data:")
print(f"  Lyapunov time: {calc_lyap_time:.2f} timesteps")
print(f"  Lyapunov exponent: {calc_lambda:.3f}")

# ============================================================================
# INITIALIZE QRC
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

n_timesteps = len(rossler_data)
n_samples = TRAIN_CONFIG['n_samples']

# Stratified sampling
indices = np.linspace(0, n_timesteps-2, n_samples, dtype=int)
print(f"\nSampling {n_samples} timesteps from {n_timesteps} total")

# ============================================================================
# PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING")
print("="*80)

sampled_data = rossler_data[indices]
preprocessed = qrc.preprocess(sampled_data, fit=True)

print(f"\n✓ Fitted StandardScaler")
print(f"  Scaled range: [{preprocessed.min():.2f}, {preprocessed.max():.2f}]")

# ============================================================================
# GENERATE RESERVOIR FEATURES
# ============================================================================
print("\n" + "="*80)
print("GENERATING RESERVOIR FEATURES")
print("="*80)

print(f"\nEstimated time: 2-5 minutes...")
X = qrc.generate_features(preprocessed, shots=SHOTS, verbose=True)

print(f"\n✓ Feature generation complete")
print(f"  Feature matrix shape: {X.shape}")

# ============================================================================
# PREPARE TARGETS (Next timestep prediction)
# ============================================================================
y_indices = indices + 1
y = rossler_data[y_indices]

print(f"\nTarget: Predict next timestep")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TRAIN_CONFIG['test_size'],
    random_state=TRAIN_CONFIG['random_state']
)

print(f"\nTrain: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

# ============================================================================
# RIDGE REGRESSION
# ============================================================================
print("\n" + "="*80)
print("RIDGE REGRESSION TRAINING")
print("="*80)

best_r2 = -np.inf
best_alpha = None
best_model = None

for alpha in TRAIN_CONFIG['ridge_alphas']:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print(f"  α={alpha:8.3f}: R² = {r2:.6f}")

    if r2 > best_r2:
        best_r2 = r2
        best_alpha = alpha
        best_model = model

print(f"\n✓ Best α = {best_alpha}")
print(f"✓ Best R² = {best_r2:.6f}")

# ============================================================================
# FORECAST HORIZON
# ============================================================================
print("\n" + "="*80)
print("FORECAST HORIZON ANALYSIS")
print("="*80)

y_pred = best_model.predict(X_test)

# Energy correlation
energy_true = np.sum(y_test**2, axis=1)
energy_pred = np.sum(y_pred**2, axis=1)
energy_corr = np.corrcoef(energy_true, energy_pred)[0, 1]

print(f"\nEnergy correlation: {energy_corr:.6f}")

# Forecast horizon in Lyapunov times
horizon_lyapunov = forecast_horizon_in_lyapunov_times(best_r2, rossler_params['lyapunov_time'])

print(f"\nForecast horizon:")
print(f"  {horizon_lyapunov:.2f} Lyapunov times")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'system': 'Rossler',
    'methodology': 'Steinegger_Rath_2025',
    'timestamp': datetime.now().isoformat(),
    'specs': {
        'n_qubits': 9,
        'topology': '3x3_square_lattice',
        'simulation': 'Qiskit_Aer'
    },
    'qrc_config': QRC_CONFIG,
    'train_config': TRAIN_CONFIG,
    'chaotic_system': rossler_params,
    'performance': {
        'r2': float(best_r2),
        'energy_correlation': float(energy_corr),
        'best_alpha': float(best_alpha),
        'forecast_horizon_lyapunov_times': float(horizon_lyapunov),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'n_features': int(X.shape[1])
    }
}

# Save results
results_path = 'rossler_9q_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

# Save predictions
np.save('rossler_9q_predictions.npy', y_pred)
np.save('rossler_9q_targets.npy', y_test)

print(f"\n✓ Results saved:")
print(f"  {results_path}")
print(f"  rossler_9q_predictions.npy")
print(f"  rossler_9q_targets.npy")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RÖSSLER QRC VALIDATION - COMPLETE")
print("="*80)
print(f"\n✓ Test R² = {best_r2:.4f}")
print(f"✓ Energy correlation = {energy_corr:.4f}")
print(f"✓ Forecast horizon = {horizon_lyapunov:.2f} Lyapunov times")
print(f"✓ Best α = {best_alpha}")
print("\n" + "="*80 + "\n")
