#!/usr/bin/env python3
"""
QDaria QRC - Phase 3: Hero Run (156 Qubits) - FIXED VERSION
=============================================================
CRITICAL FIX: Creates a unique circuit for each input sample.

The original version used the same circuit 10 times, which is why
all reservoir states were nearly identical (correlation > 0.98).
"""

import numpy as np
from datetime import datetime
import json
import os
import time

# === CONFIGURATION ===
N_QUBITS = 156
N_SHOTS = 4000
N_SAMPLES = 50  # Increased from 10 to get meaningful statistics
N_LAYERS = 3    # Must match generate_hero_qasm.py
MAX_EXECUTION_TIME = 3600  # 1 hour for 50 samples
COST_PER_SHOT = 0.0016

TARGET_BACKEND = 'ibm_kingston'
FALLBACK_BACKENDS = ['ibm_torino', 'ibm_fez', 'ibm_marrakesh']


def generate_heavy_hex_coupling(n_qubits: int = 156):
    """Generate Heavy-Hex coupling map for IBM Heron."""
    edges = []
    n_cols = 13
    n_rows = 12

    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx >= n_qubits:
                break

            if col < n_cols - 1 and idx + 1 < n_qubits:
                edges.append((idx, idx + 1))

            if row < n_rows - 1:
                below = idx + n_cols
                if below < n_qubits:
                    edges.append((idx, below))

                if col % 2 == row % 2:
                    if col > 0:
                        diag = below - 1
                        if diag < n_qubits:
                            edges.append((idx, diag))
                else:
                    if col < n_cols - 1:
                        diag = below + 1
                        if diag < n_qubits:
                            edges.append((idx, diag))

    return edges


def build_qrc_circuit(input_data: np.ndarray, reservoir_seed: int = 156):
    """
    Build a QRC circuit with UNIQUE input encoding.
    
    CRITICAL: Each input sample must produce a different circuit!
    
    Args:
        input_data: 1D array of input values (e.g., spectral coefficients)
        reservoir_seed: Seed for reservoir parameters (same across all samples)
    
    Returns:
        QuantumCircuit with input-dependent encoding
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    
    coupling_map = generate_heavy_hex_coupling(N_QUBITS)
    
    qr = QuantumRegister(N_QUBITS, 'q')
    cr = ClassicalRegister(N_QUBITS, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # Fixed reservoir parameters (same for all samples - this is correct)
    np.random.seed(reservoir_seed)
    n_params = N_LAYERS * (N_QUBITS * 2 + len(coupling_map))
    reservoir_params = np.random.uniform(0, 2*np.pi, n_params)
    
    # CRITICAL: Normalize input to [-π, π] for rotation angles
    # This is where each sample becomes UNIQUE
    data_min, data_max = input_data.min(), input_data.max()
    if data_max - data_min > 1e-10:
        scaled = 2 * np.pi * (input_data - data_min) / (data_max - data_min) - np.pi
    else:
        scaled = np.zeros_like(input_data)
    
    param_idx = 0
    input_idx = 0
    n_inputs = len(input_data)
    
    for layer in range(N_LAYERS):
        # INPUT ENCODING - This is what makes each circuit unique!
        for q in range(N_QUBITS):
            qc.sx(q)
            qc.rz(float(scaled[input_idx % n_inputs]), q)  # <-- Different per sample!
            input_idx += 1
        
        # Reservoir dynamics (fixed across samples)
        for q in range(N_QUBITS):
            qc.rz(reservoir_params[param_idx], q)
            param_idx += 1
            qc.sx(q)
            qc.rz(reservoir_params[param_idx], q)
            param_idx += 1
        
        # Entangling layer
        edge_set_1 = coupling_map[::3]
        edge_set_2 = coupling_map[1::3]
        edge_set_3 = coupling_map[2::3]
        
        for q1, q2 in edge_set_1:
            qc.cz(q1, q2)
        for q1, q2 in edge_set_2:
            qc.cz(q1, q2)
        for q1, q2 in edge_set_3:
            qc.cz(q1, q2)
    
    qc.measure(qr, cr)
    return qc


def run_hero_fixed(spectral_data: np.ndarray, n_samples: int = N_SAMPLES):
    """
    Execute Hero Run with UNIQUE circuits per sample.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    
    print("=" * 60)
    print("HERO RUN - FIXED VERSION")
    print("Each sample gets a UNIQUE circuit!")
    print("=" * 60)
    
    # Connect - try open instance first
    print("\nConnecting to IBM Quantum...")
    
    # Try different instance configurations
    service = None
    instance_used = None
    
    for instance in ['ibm-q/open/main', 'open-instance', None]:
        try:
            if instance:
                service = QiskitRuntimeService(instance=instance)
                instance_used = instance
            else:
                service = QiskitRuntimeService(instance='open-instance')
                instance_used = 'default'
            print(f"  Connected using instance: {instance_used}")
            break
        except Exception as e:
            print(f"  Instance {instance} failed: {e}")
            continue
    
    if service is None:
        print("ERROR: Could not connect to IBM Quantum")
        return None
    
    # List available backends first
    print("\nAvailable 100+ qubit backends:")
    all_backends = service.backends()
    large_backends = [b for b in all_backends if b.configuration().n_qubits >= 156]
    
    for b in large_backends:
        status = b.status()
        print(f"  {b.name}: {b.configuration().n_qubits}q, operational={status.operational}, queue={status.pending_jobs}")
    
    if not large_backends:
        print("\nNo 100+ qubit backends available on this instance.")
        print("Available backends:")
        for b in all_backends[:10]:
            print(f"  {b.name}: {b.configuration().n_qubits}q")
        return None
    
    # Select best available backend
    backend = None
    for target in [TARGET_BACKEND] + FALLBACK_BACKENDS:
        for b in large_backends:
            if b.name == target and b.status().operational:
                backend = b
                break
        if backend:
            break
    
    if not backend:
        # Use first operational large backend
        for b in large_backends:
            if b.status().operational:
                backend = b
                break
    
    if not backend:
        print("ERROR: No operational 156+ qubit backend found")
        return None
    
    print(f"  Backend: {backend.name}")
    print(f"  Qubits: {backend.configuration().n_qubits}")
    print(f"  Pending jobs: {backend.status().pending_jobs}")
    
    # Sample indices (evenly spaced through the time series)
    indices = np.linspace(0, len(spectral_data) - 2, n_samples, dtype=int)
    print(f"\n  Sampling {n_samples} timesteps: {indices[:5]}...{indices[-3:]}")
    
    # Build UNIQUE circuits for each sample
    print(f"\nBuilding {n_samples} unique circuits...")
    circuits = []
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    
    for i, idx in enumerate(indices):
        input_vec = spectral_data[idx]  # This is different for each sample!
        qc = build_qrc_circuit(input_vec, reservoir_seed=156)
        transpiled = pm.run(qc)
        circuits.append(transpiled)
        
        if (i + 1) % 10 == 0:
            print(f"    Built {i+1}/{n_samples} circuits...")
    
    print(f"  ✓ All {n_samples} circuits built")
    print(f"  Circuit depth: {circuits[0].depth()}")
    
    # Verify circuits are different by checking gate parameters
    print("\nVerifying circuits are unique...")
    
    # Compare the RZ angles in the first layer (should differ due to different inputs)
    def get_rz_params(circ, max_gates=10):
        params = []
        for inst in circ.data[:50]:  # Check first 50 instructions
            if inst.operation.name == 'rz' and len(params) < max_gates:
                params.append(float(inst.operation.params[0]))
        return params
    
    params_0 = get_rz_params(circuits[0])
    params_1 = get_rz_params(circuits[1])
    
    print(f"  Circuit 0 first RZ angles: {[f'{p:.3f}' for p in params_0[:5]]}")
    print(f"  Circuit 1 first RZ angles: {[f'{p:.3f}' for p in params_1[:5]]}")
    
    if params_0 == params_1:
        print("  ⚠ WARNING: First two circuits have identical parameters!")
        print("  This means the bug is NOT fixed. Aborting.")
        return None
    else:
        print("  ✓ Circuits have different parameters (bug is fixed!)")
    
    # Cost estimate
    total_shots = n_samples * N_SHOTS
    estimated_cost = total_shots * COST_PER_SHOT
    print(f"\nCost estimate: ${estimated_cost:.2f}")
    print(f"  {n_samples} samples × {N_SHOTS} shots = {total_shots:,} total shots")
    
    # Confirm before running
    confirm = input("\nProceed? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Aborted.")
        return None
    
    # Submit job
    print(f"\nSubmitting to {backend.name}...")
    sampler = Sampler(backend)
    job = sampler.run(circuits, shots=N_SHOTS)
    
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")
    
    # Save job ID for recovery
    with open('hero_job_id.txt', 'w') as f:
        f.write(job_id)
    print(f"  Job ID saved to hero_job_id.txt")
    
    # Wait for results
    print("  Waiting for results...")
    start_time = time.time()
    
    while not job.done():
        elapsed = time.time() - start_time
        print(f"    Status: {job.status()} ({elapsed/60:.1f} min elapsed)")
        time.sleep(60)
    
    # Extract results with checkpointing
    print("\nExtracting results...")
    result = job.result()
    
    reservoir_states = np.zeros((n_samples, N_QUBITS))
    
    for i, pub_result in enumerate(result):
        counts = pub_result.data.c.get_counts()
        total_shots = sum(counts.values())
        
        # Extract single-qubit expectation values
        z_exp = np.zeros(N_QUBITS)
        for bitstring, count in counts.items():
            for q in range(min(N_QUBITS, len(bitstring))):
                bit = int(bitstring[-(q+1)]) if q < len(bitstring) else 0
                z_exp[q] += (1 - 2*bit) * count
        z_exp /= total_shots
        
        reservoir_states[i] = z_exp
        
        # Checkpoint every 10 samples
        if (i + 1) % 10 == 0:
            np.save('reservoir_states_checkpoint.npy', reservoir_states[:i+1])
            print(f"    Extracted {i+1}/{n_samples} samples (checkpoint saved)")
    
    print(f"  ✓ Reservoir shape: {reservoir_states.shape}")
    
    # Quick validation
    corr_matrix = np.corrcoef(reservoir_states)
    off_diag = corr_matrix[np.triu_indices(n_samples, k=1)]
    mean_corr = off_diag.mean()
    print(f"  Mean sample correlation: {mean_corr:.4f}")
    
    if mean_corr > 0.95:
        print("  ⚠ WARNING: Samples still highly correlated - check circuit!")
    else:
        print("  ✓ Samples show diversity (circuit is working)")
    
    # Save results
    np.save('reservoir_states_hero_156q_fixed.npy', reservoir_states)
    np.save('hero_sample_indices_fixed.npy', indices)
    
    output = {
        'status': 'SUCCESS',
        'backend': backend.name,
        'job_id': job_id,
        'n_qubits': N_QUBITS,
        'n_samples': n_samples,
        'n_shots': N_SHOTS,
        'mean_sample_correlation': float(mean_corr),
        'estimated_cost': estimated_cost,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open('hero_run_fixed_result.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("HERO RUN COMPLETE")
    print(f"{'=' * 60}")
    print("Files saved:")
    print("  - reservoir_states_hero_156q_fixed.npy")
    print("  - hero_sample_indices_fixed.npy")
    print("  - hero_run_fixed_result.json")
    
    return reservoir_states, indices


if __name__ == "__main__":
    # Load spectral data
    if os.path.exists('training_spectral.npy'):
        spectral_data = np.load('training_spectral.npy')
    elif os.path.exists('../data/training_spectral.npy'):
        spectral_data = np.load('../data/training_spectral.npy')
    else:
        print("ERROR: training_spectral.npy not found")
        exit(1)
    
    print(f"Spectral data: {spectral_data.shape}")
    
    run_hero_fixed(spectral_data, n_samples=50)