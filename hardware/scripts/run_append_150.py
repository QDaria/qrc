#!/usr/bin/env python3
"""
Run 150 additional samples and append to existing 50.
"""

import numpy as np
from datetime import datetime
import json
import time

N_QUBITS = 156
N_SHOTS = 4000
N_LAYERS = 3

def generate_heavy_hex_coupling(n_qubits=156):
    edges = []
    n_cols, n_rows = 13, 12
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
                    if col > 0 and below - 1 < n_qubits:
                        edges.append((idx, below - 1))
                else:
                    if col < n_cols - 1 and below + 1 < n_qubits:
                        edges.append((idx, below + 1))
    return edges

def build_qrc_circuit(input_data, reservoir_seed=156):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    
    coupling_map = generate_heavy_hex_coupling(N_QUBITS)
    qr = QuantumRegister(N_QUBITS, 'q')
    cr = ClassicalRegister(N_QUBITS, 'c')
    qc = QuantumCircuit(qr, cr)
    
    np.random.seed(reservoir_seed)
    n_params = N_LAYERS * (N_QUBITS * 2 + len(coupling_map))
    reservoir_params = np.random.uniform(0, 2*np.pi, n_params)
    
    data_min, data_max = input_data.min(), input_data.max()
    if data_max - data_min > 1e-10:
        scaled = 2 * np.pi * (input_data - data_min) / (data_max - data_min) - np.pi
    else:
        scaled = np.zeros_like(input_data)
    
    param_idx, input_idx = 0, 0
    n_inputs = len(input_data)
    
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            qc.sx(q)
            qc.rz(float(scaled[input_idx % n_inputs]), q)
            input_idx += 1
        for q in range(N_QUBITS):
            qc.rz(reservoir_params[param_idx], q)
            param_idx += 1
            qc.sx(q)
            qc.rz(reservoir_params[param_idx], q)
            param_idx += 1
        for q1, q2 in coupling_map[::3]:
            qc.cz(q1, q2)
        for q1, q2 in coupling_map[1::3]:
            qc.cz(q1, q2)
        for q1, q2 in coupling_map[2::3]:
            qc.cz(q1, q2)
    
    qc.measure(qr, cr)
    return qc

def main():
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    
    print("=" * 60)
    print("APPEND 150 SAMPLES TO EXISTING 50")
    print("=" * 60)
    
    # Load data
    spectral = np.load('training_spectral.npy')
    new_indices = np.load('new_150_indices.npy')
    n_samples = len(new_indices)
    
    print(f"New samples to run: {n_samples}")
    print(f"Indices: {new_indices[:5]}...{new_indices[-3:]}")
    
    # Connect
    print("\nConnecting to IBM Quantum...")
    service = QiskitRuntimeService(instance='open-instance')
    
    # Get backend
    backend = service.backend('ibm_fez')
    print(f"Backend: {backend.name}")
    print(f"Queue: {backend.status().pending_jobs} jobs")
    
    # Build circuits
    print(f"\nBuilding {n_samples} circuits...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    circuits = []
    
    for i, idx in enumerate(new_indices):
        qc = build_qrc_circuit(spectral[idx], reservoir_seed=156)
        transpiled = pm.run(qc)
        circuits.append(transpiled)
        if (i + 1) % 30 == 0:
            print(f"    Built {i+1}/{n_samples} circuits...")
    
    print(f"  ✓ All {n_samples} circuits built")
    
    # Verify different
    def get_rz_params(circ):
        params = []
        for inst in circ.data[:50]:
            if inst.operation.name == 'rz' and len(params) < 5:
                params.append(float(inst.operation.params[0]))
        return params
    
    p0, p1 = get_rz_params(circuits[0]), get_rz_params(circuits[1])
    print(f"  Circuit 0 RZ: {[f'{p:.2f}' for p in p0]}")
    print(f"  Circuit 1 RZ: {[f'{p:.2f}' for p in p1]}")
    
    if p0 == p1:
        print("  ERROR: Circuits identical!")
        return
    print("  ✓ Circuits are unique")
    
    # Confirm
    print(f"\nEstimated time: ~3-4 minutes")
    confirm = input("Proceed? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Aborted.")
        return
    
    # Submit
    print(f"\nSubmitting to {backend.name}...")
    sampler = Sampler(backend)
    job = sampler.run(circuits, shots=N_SHOTS)
    
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")
    
    with open('append_job_id.txt', 'w') as f:
        f.write(job_id)
    
    # Wait
    print("  Waiting...")
    start = time.time()
    while not job.done():
        print(f"    Status: {job.status()} ({(time.time()-start)/60:.1f} min)")
        time.sleep(60)
    
    # Extract
    print("\nExtracting results...")
    result = job.result()
    
    new_reservoir = np.zeros((n_samples, N_QUBITS))
    for i, pub_result in enumerate(result):
        counts = pub_result.data.c.get_counts()
        total = sum(counts.values())
        z_exp = np.zeros(N_QUBITS)
        for bitstring, count in counts.items():
            for q in range(min(N_QUBITS, len(bitstring))):
                bit = int(bitstring[-(q+1)])
                z_exp[q] += (1 - 2*bit) * count
        z_exp /= total
        new_reservoir[i] = z_exp
        if (i + 1) % 30 == 0:
            print(f"    Extracted {i+1}/{n_samples}")
    
    # Load existing and append
    existing_reservoir = np.load('reservoir_states_hero_156q_fixed.npy')
    existing_indices = np.load('hero_sample_indices_fixed.npy')
    
    combined_reservoir = np.vstack([existing_reservoir, new_reservoir])
    combined_indices = np.concatenate([existing_indices, new_indices])
    
    # Sort by index for cleaner data
    sort_order = np.argsort(combined_indices)
    combined_reservoir = combined_reservoir[sort_order]
    combined_indices = combined_indices[sort_order]
    
    # Save
    np.save('reservoir_states_hero_156q_200.npy', combined_reservoir)
    np.save('hero_sample_indices_200.npy', combined_indices)
    
    print(f"\n✓ Combined reservoir: {combined_reservoir.shape}")
    print(f"✓ Saved: reservoir_states_hero_156q_200.npy")
    print(f"✓ Saved: hero_sample_indices_200.npy")
    
    # Quick validation
    corr = np.corrcoef(combined_reservoir)
    off_diag = corr[np.triu_indices(len(corr), k=1)]
    print(f"\nMean sample correlation: {off_diag.mean():.3f}")

if __name__ == "__main__":
    main()
