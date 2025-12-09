#!/usr/bin/env python3
"""
Generate Phase Space Trajectory Plots for Chaotic Systems
===========================================================
Create publication-quality 3D phase space visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-v0_8-paper')

print("="*80)
print("GENERATING PHASE SPACE TRAJECTORY PLOTS")
print("="*80)

# Load data
lorenz_data = np.load('../02_new_turbulence/lorenz63_trajectory.npy')
rossler_data = np.load('../02_new_turbulence/rossler_trajectory.npy')

print(f"\n✓ Loaded Lorenz-63: {lorenz_data.shape}")
print(f"✓ Loaded Rössler: {rossler_data.shape}")

# ============================================================================
# FIGURE: Combined Phase Space Plots
# ============================================================================
print("\n📊 Creating combined phase space figure...")

fig = plt.figure(figsize=(14, 5))

# --- Lorenz-63 Butterfly Attractor ---
ax1 = fig.add_subplot(131, projection='3d')

# Plot trajectory (use subset for clarity)
n_plot = 5000
x, y, z = lorenz_data[:n_plot, 0], lorenz_data[:n_plot, 1], lorenz_data[:n_plot, 2]

# Color by time for visual effect
colors = plt.cm.viridis(np.linspace(0, 1, n_plot))
for i in range(n_plot - 1):
    ax1.plot(x[i:i+2], y[i:i+2], z[i:i+2],
             color=colors[i], alpha=0.6, linewidth=0.5)

ax1.set_xlabel('x', fontsize=11, fontweight='bold')
ax1.set_ylabel('y', fontsize=11, fontweight='bold')
ax1.set_zlabel('z', fontsize=11, fontweight='bold')
ax1.set_title('Lorenz-63 Attractor\n(λ=0.906, butterfly topology)',
              fontsize=12, fontweight='bold')
ax1.view_init(elev=20, azim=45)
ax1.grid(True, alpha=0.3)

# --- Rössler Single-Lobe Attractor ---
ax2 = fig.add_subplot(132, projection='3d')

# Plot trajectory
n_plot = 5000
x, y, z = rossler_data[:n_plot, 0], rossler_data[:n_plot, 1], rossler_data[:n_plot, 2]

colors = plt.cm.plasma(np.linspace(0, 1, n_plot))
for i in range(n_plot - 1):
    ax2.plot(x[i:i+2], y[i:i+2], z[i:i+2],
             color=colors[i], alpha=0.6, linewidth=0.5)

ax2.set_xlabel('x', fontsize=11, fontweight='bold')
ax2.set_ylabel('y', fontsize=11, fontweight='bold')
ax2.set_zlabel('z', fontsize=11, fontweight='bold')
ax2.set_title('Rössler Attractor\n(λ=0.071, spiral topology)',
              fontsize=12, fontweight='bold')
ax2.view_init(elev=20, azim=45)
ax2.grid(True, alpha=0.3)

# --- Trajectory Comparison (2D projection) ---
ax3 = fig.add_subplot(133)

# Plot x-y projections
ax3.plot(lorenz_data[:3000, 0], lorenz_data[:3000, 1],
         'b-', alpha=0.4, linewidth=0.5, label='Lorenz-63')
ax3.plot(rossler_data[:3000, 0], rossler_data[:3000, 1],
         'r-', alpha=0.4, linewidth=0.5, label='Rössler')

ax3.set_xlabel('x', fontsize=11, fontweight='bold')
ax3.set_ylabel('y', fontsize=11, fontweight='bold')
ax3.set_title('Phase Space Projection (x-y)\nDifferent Topologies',
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=10, loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

plt.tight_layout()
plt.savefig('phase_space_attractors.pdf', dpi=300, bbox_inches='tight')
plt.savefig('phase_space_attractors.png', dpi=300, bbox_inches='tight')
print("✓ Saved: phase_space_attractors.pdf")
print("✓ Saved: phase_space_attractors.png")

# ============================================================================
# FIGURE: Lorenz-63 Detailed View
# ============================================================================
print("\n📊 Creating Lorenz-63 detailed view...")

fig = plt.figure(figsize=(12, 4))

# 3D view
ax1 = fig.add_subplot(131, projection='3d')
n_plot = 3000
x, y, z = lorenz_data[:n_plot, 0], lorenz_data[:n_plot, 1], lorenz_data[:n_plot, 2]
colors = plt.cm.coolwarm(np.linspace(0, 1, n_plot))
for i in range(n_plot - 1):
    ax1.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], alpha=0.7, linewidth=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('3D Phase Space', fontweight='bold')
ax1.view_init(elev=20, azim=45)

# x-z projection
ax2 = fig.add_subplot(132)
ax2.plot(lorenz_data[:n_plot, 0], lorenz_data[:n_plot, 2],
         'b-', alpha=0.6, linewidth=0.5)
ax2.set_xlabel('x', fontweight='bold')
ax2.set_ylabel('z', fontweight='bold')
ax2.set_title('x-z Projection (Butterfly)', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Time series
ax3 = fig.add_subplot(133)
time = np.arange(n_plot) * 0.01
ax3.plot(time, lorenz_data[:n_plot, 0], 'b-', alpha=0.8, linewidth=1, label='x(t)')
ax3.plot(time, lorenz_data[:n_plot, 1], 'r-', alpha=0.8, linewidth=1, label='y(t)')
ax3.plot(time, lorenz_data[:n_plot, 2], 'g-', alpha=0.8, linewidth=1, label='z(t)')
ax3.set_xlabel('Time', fontweight='bold')
ax3.set_ylabel('State', fontweight='bold')
ax3.set_title('Chaotic Dynamics', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lorenz63_detailed.pdf', dpi=300, bbox_inches='tight')
plt.savefig('lorenz63_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: lorenz63_detailed.pdf")
print("✓ Saved: lorenz63_detailed.png")

# ============================================================================
# FIGURE: Rössler Detailed View
# ============================================================================
print("\n📊 Creating Rössler detailed view...")

fig = plt.figure(figsize=(12, 4))

# 3D view
ax1 = fig.add_subplot(131, projection='3d')
n_plot = 3000
x, y, z = rossler_data[:n_plot, 0], rossler_data[:n_plot, 1], rossler_data[:n_plot, 2]
colors = plt.cm.viridis(np.linspace(0, 1, n_plot))
for i in range(n_plot - 1):
    ax1.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], alpha=0.7, linewidth=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('3D Phase Space', fontweight='bold')
ax1.view_init(elev=20, azim=45)

# x-y projection (spiral)
ax2 = fig.add_subplot(132)
ax2.plot(rossler_data[:n_plot, 0], rossler_data[:n_plot, 1],
         'r-', alpha=0.6, linewidth=0.5)
ax2.set_xlabel('x', fontweight='bold')
ax2.set_ylabel('y', fontweight='bold')
ax2.set_title('x-y Projection (Spiral)', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Time series
ax3 = fig.add_subplot(133)
time = np.arange(n_plot) * 0.1
ax3.plot(time, rossler_data[:n_plot, 0], 'b-', alpha=0.8, linewidth=1, label='x(t)')
ax3.plot(time, rossler_data[:n_plot, 1], 'r-', alpha=0.8, linewidth=1, label='y(t)')
ax3.plot(time, rossler_data[:n_plot, 2], 'g-', alpha=0.8, linewidth=1, label='z(t)')
ax3.set_xlabel('Time', fontweight='bold')
ax3.set_ylabel('State', fontweight='bold')
ax3.set_title('Chaotic Dynamics', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rossler_detailed.pdf', dpi=300, bbox_inches='tight')
plt.savefig('rossler_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rossler_detailed.pdf")
print("✓ Saved: rossler_detailed.png")

print("\n" + "="*80)
print("PHASE SPACE VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. phase_space_attractors.pdf (combined 3-panel)")
print("  2. lorenz63_detailed.pdf (3-panel detailed)")
print("  3. rossler_detailed.pdf (3-panel detailed)")
print("\n" + "="*80 + "\n")
