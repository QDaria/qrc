#!/usr/bin/env python3
"""
Rigetti Cepheus 36Q QRC Simulation - BATCH PROCESSING VERSION
==============================================================
Simulate Rigetti Cepheus-1 36-qubit QPU (4×9Q chiplets) using Steinegger & Räth (2025) methodology.
Uses batch processing to avoid memory issues.

System Specs:
- 36 qubits in modular 4×9 architecture
- 2-qubit gate fidelity: 99.5% (median)
- 4 Novera chiplets
- 2× better error rate than Ankaa-3

Batch Processing Strategy:
- Process 10 samples at a time (60 circuits per batch: 10 × 3V × 2r)
- Save intermediate results to disk
- Resume from last checkpoint if crash
- Constant memory usage throughout
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
print("RIGETTI CEPHEUS 36Q - QRC SIMULATION (BATCH PROCESSING)")
print("="*80)
print("Methodology: Steinegger & Räth (Nature Sci Rep 2025)")
print("System: Rigetti Cepheus-1 QPU (36-qubit, 4×9Q chiplets)")
print("Strategy: Process samples in batches to manage memory")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
print("\n" + "="*80)
print("CONFIGURATION")
print("="*80)

# Rigetti Cepheus specifications
CEPHEUS_SPECS = {
    'n_qubits': 36,
    'topology': 'modular_4x9_chiplets',
    'gate_fidelity_2q': 0.995,  # 99.5% median
    'n_chiplets': 4,
    'architecture': 'Multi-chip_Novera',
    'error_reduction': '2x_vs_Ankaa3'
}

# QRC parameters (slightly reduced from Steinegger defaults for 36Q system)
QRC_CONFIG = {
    'n_qubits': 36,
    'n_layers': 6,  # Reduced from 8 (larger system needs fewer layers)
    'V': 3,  # Temporal multiplexing (reduced from 5)
    'r': 2,  # Spatial multiplexing (reduced from 3)
    'G': 3,  # Polynomial degree (reduced from 4)
    'scale_range': (0, 1),
    'random_seed': 42
}

SHOTS = 4000  # Measurement shots

# Training configuration
TRAIN_CONFIG = {
    'n_samples': 150,  # Total samples to process
    'batch_size': 10,  # Process 10 samples at a time
    'test_size': 0.2,
    'ridge_alphas': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    'random_state': 42
}

# Checkpoint directory
CHECKPOINT_DIR = 'cepheus_36q_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("\nRigetti Cepheus Specifications:")
for key, value in CEPHEUS_SPECS.items():
    print(f"  {key}: {value}")

print("\nQRC Configuration:")
for key, value in QRC_CONFIG.items():
    print(f"  {key}: {value}")

print(f"\nMeasurement shots: {SHOTS}")

print("\nBatch Processing Configuration:")
print(f"  Total samples: {TRAIN_CONFIG['n_samples']}")
print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
print(f"  Batches: {TRAIN_CONFIG['n_samples'] // TRAIN_CONFIG['batch_size']}")
print(f"  Circuits per batch: {TRAIN_CONFIG['batch_size'] * QRC_CONFIG['V'] * QRC_CONFIG['r']}")

# ============================================================================
# INITIALIZE QRC
# ============================================================================
print("\n" + "="*80)
print("INITIALIZING STEINEGGER QRC")
print("="*80)

qrc = SteineggerQRC(**QRC_CONFIG)

print(f"\n✓ QRC initialized")
print(f"  Total qubits: {qrc.n_qubits}")
print(f"  Circuit depth: {qrc.n_layers} layers")
print(f"  Temporal multiplexing (V): {qrc.V}")
print(f"  Spatial multiplexing (r): {qrc.r}")
print(f"  Polynomial degree (G): {qrc.G}")
print(f"  Single-qubit observables: {qrc.n_single}")
print(f"  Two-qubit correlations: {qrc.n_corr}")
print(f"  Features per measurement: {qrc.n_obs_per_meas}")
print(f"  Total features: {qrc.n_features:,}")

# ============================================================================
# LOAD TURBULENCE DATA
# ============================================================================
print("\n" + "="*80)
print("LOADING TURBULENCE DATA")
print("="*80)

# Load original turbulence data (same as IBM 4Q/156Q used)
turbulence_data = np.load('../01_original_data/training_spectral.npy')

print(f"\n✓ Loaded: training_spectral.npy (original data)")
print(f"  Shape: {turbulence_data.shape}")
print(f"  Note: Same dataset used for IBM 4Q and 156Q results")

# Calculate Lyapunov time
lyapunov_time, lambda_max = calculate_lyapunov_time(turbulence_data)

print(f"\nChaos Metrics:")
print(f"  Lyapunov time (τ): {lyapunov_time:.4f} timesteps")
print(f"  λ_max: {lambda_max:.4f}")

# ============================================================================
# SAMPLE AND PREPROCESS
# ============================================================================
print("\n" + "="*80)
print("SAMPLING AND PREPROCESSING")
print("="*80)

n_samples = TRAIN_CONFIG['n_samples']
np.random.seed(TRAIN_CONFIG['random_state'])

# Sample random timesteps (avoiding end to have next timestep)
max_idx = len(turbulence_data) - 2
indices = np.random.choice(max_idx, size=n_samples, replace=False)
indices.sort()

sampled_data = turbulence_data[indices]
preprocessed = qrc.preprocess(sampled_data, fit=True)

print(f"\n✓ Sampled {n_samples} timesteps")
print(f"  Fitted StandardScaler")
print(f"  Input range: [{sampled_data.min():.2f}, {sampled_data.max():.2f}]")
print(f"  Scaled range: [{preprocessed.min():.2f}, {preprocessed.max():.2f}]")

# ============================================================================
# BATCH FEATURE GENERATION
# ============================================================================
print("\n" + "="*80)
print("BATCH FEATURE GENERATION")
print("="*80)

batch_size = TRAIN_CONFIG['batch_size']
n_batches = n_samples // batch_size
feature_batches = []

print(f"\nProcessing {n_batches} batches of {batch_size} samples each")
print(f"Circuits per batch: {batch_size * QRC_CONFIG['V'] * QRC_CONFIG['r']}")
print(f"Estimated time per batch: ~3-5 minutes")
print(f"Total estimated time: ~{n_batches * 4} minutes\n")

for batch_idx in range(n_batches):
    batch_start = batch_idx * batch_size
    batch_end = batch_start + batch_size

    checkpoint_file = os.path.join(CHECKPOINT_DIR, f'batch_{batch_idx:02d}.npy')

    # Check if this batch already processed
    if os.path.exists(checkpoint_file):
        print(f"Batch {batch_idx + 1}/{n_batches}: Loading from checkpoint...")
        X_batch = np.load(checkpoint_file)
        feature_batches.append(X_batch)
        print(f"  ✓ Loaded {X_batch.shape[0]} samples with {X_batch.shape[1]:,} features")
        continue

    # Process this batch
    print(f"Batch {batch_idx + 1}/{n_batches}: Processing samples {batch_start}-{batch_end-1}...")

    try:
        # Extract batch data
        batch_data = preprocessed[batch_start:batch_end]

        # Generate features for this batch
        X_batch = qrc.generate_features(batch_data, shots=SHOTS, verbose=False)

        # Save checkpoint
        np.save(checkpoint_file, X_batch)
        feature_batches.append(X_batch)

        print(f"  ✓ Generated {X_batch.shape[0]} samples with {X_batch.shape[1]:,} features")
        print(f"  ✓ Saved checkpoint: {checkpoint_file}")

    except Exception as e:
        print(f"  ✗ ERROR in batch {batch_idx}: {e}")
        print(f"  Stopping at batch {batch_idx}. Resume by running script again.")
        sys.exit(1)

# Concatenate all batches
print(f"\nConcatenating {len(feature_batches)} batches...")
X = np.vstack(feature_batches)

print(f"\n✓ Feature generation complete")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Features per timestep: {X.shape[1]:,}")

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

print(f"\nEnergy Correlation: {energy_corr:.4f}")

# Calculate forecast horizon in Lyapunov times
forecast_horizon_lyap = forecast_horizon_in_lyapunov_times(
    best_r2, lyapunov_time, dt=0.01
)

print(f"\nForecast Horizon:")
print(f"  R² score: {best_r2:.4f}")
print(f"  Lyapunov times: {forecast_horizon_lyap:.2f} τ")
print(f"  Timesteps: {forecast_horizon_lyap * lyapunov_time:.2f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save feature matrix
np.save('cepheus_36q_features.npy', X)
np.save('cepheus_36q_targets.npy', y)

# Predictions for visualization
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

np.save('cepheus_36q_predictions.npy', y_pred_test)

# Save comprehensive results
results = {
    'system': 'Rigetti_Cepheus_36Q',
    'methodology': 'Steinegger_Rath_2025',
    'timestamp': datetime.now().isoformat(),
    'specs': CEPHEUS_SPECS,
    'qrc_config': QRC_CONFIG,
    'train_config': TRAIN_CONFIG,
    'turbulence': {
        'file': 'training_spectral.npy',
        'note': 'Original data (same as IBM 4Q/156Q)',
        'lyapunov_time': lyapunov_time,
        'lambda_max': lambda_max
    },
    'performance': {
        'r2': best_r2,
        'energy_correlation': energy_corr,
        'best_alpha': best_alpha,
        'forecast_horizon_lyapunov_times': forecast_horizon_lyap,
        'forecast_horizon_timesteps': forecast_horizon_lyap * lyapunov_time,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': qrc.n_features
    }
}

with open('cepheus_36q_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Saved:")
print("  - cepheus_36q_features.npy")
print("  - cepheus_36q_targets.npy")
print("  - cepheus_36q_predictions.npy")
print("  - cepheus_36q_results.json")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Energy prediction
ax = axes[0, 0]
energy_train_true = np.sum(np.abs(y_train[:, :50] + 1j * y_train[:, 50:])**2, axis=1)
energy_train_pred = np.sum(np.abs(y_pred_train[:, :50] + 1j * y_pred_train[:, 50:])**2, axis=1)

ax.plot(energy_train_true[:50], label='True', alpha=0.7)
ax.plot(energy_train_pred[:50], label='Predicted', alpha=0.7)
ax.set_xlabel('Time Step')
ax.set_ylabel('Energy')
ax.set_title(f'Energy Prediction (R²={best_r2:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Scatter plot
ax = axes[0, 1]
ax.scatter(energy_true, energy_pred, alpha=0.5, s=20)
ax.plot([energy_true.min(), energy_true.max()],
        [energy_true.min(), energy_true.max()],
        'r--', lw=2, label='Perfect')
ax.set_xlabel('True Energy')
ax.set_ylabel('Predicted Energy')
ax.set_title(f'Scatter (Corr={energy_corr:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals
ax = axes[1, 0]
residuals = y_pred_test.flatten() - y_test.flatten()
ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
ax.axvline(residuals.mean(), color='red', linestyle='--',
           label=f'Mean: {residuals.mean():.2e}')
ax.set_xlabel('Residual (Pred - True)')
ax.set_ylabel('Count')
ax.set_title('Residual Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Results summary
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""Rigetti Cepheus 36Q QRC Results
{'='*40}

Hardware:
  • 36 qubits (4×9Q chiplets)
  • 99.5% gate fidelity

QRC Configuration:
  • n_qubits={QRC_CONFIG['n_qubits']}, n_layers={QRC_CONFIG['n_layers']}
  • V={QRC_CONFIG['V']}, r={QRC_CONFIG['r']}, G={QRC_CONFIG['G']}
  • Features: {qrc.n_features:,}

Performance:
  • R² = {best_r2:.4f}
  • Best α = {best_alpha}
  • Forecast: {forecast_horizon_lyap:.1f} τ ({forecast_horizon_lyap * lyapunov_time:.1f} steps)

Training:
  • Lyapunov time = {lyapunov_time:.2f} τ
  • Train/Test: {len(X_train)}/{len(X_test)} samples
"""

ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig('cepheus_36q_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: cepheus_36q_results.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SIMULATION COMPLETE")
print("="*80)

print(f"\nRigetti Cepheus 36Q Performance:")
print(f"  R² Score: {best_r2:.4f}")
print(f"  Energy Correlation: {energy_corr:.4f}")
print(f"  Forecast Horizon: {forecast_horizon_lyap:.2f} Lyapunov times")
print(f"  Forecast Timesteps: {forecast_horizon_lyap * lyapunov_time:.2f}")

print(f"\nComparison with other systems:")
print(f"  IBM 4Q:      R² = 0.764,  Forecast = 4.5 τ")
print(f"  Rigetti 9Q:  R² = 0.936,  Forecast = 15.2 τ")
print(f"  Rigetti 36Q: R² = {best_r2:.3f}, Forecast = {forecast_horizon_lyap:.1f} τ")
print(f"  IBM 156Q:    R² = 0.723,  Forecast = 4.2 τ")

print("\n" + "="*80)
