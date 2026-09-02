#!/usr/bin/env python3
"""
Generate 156-Qubit Hero Circuit as OpenQASM 3.0
================================================
Exports the verified Heavy-Hex circuit to QASM format.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.qasm3 import dumps as qasm3_dumps

# === CONFIGURATION ===
N_QUBITS = 156
N_LAYERS = 3


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


def build_hero_circuit(input_data: np.ndarray) -> QuantumCircuit:
    """Build the 156-qubit Heavy-Hex QRC circuit."""
    coupling_map = generate_heavy_hex_coupling(N_QUBITS)

    qr = QuantumRegister(N_QUBITS, 'q')
    cr = ClassicalRegister(N_QUBITS, 'c')
    qc = QuantumCircuit(qr, cr)

    # Initialize reservoir parameters
    np.random.seed(156)
    n_params = N_LAYERS * (N_QUBITS * 2 + len(coupling_map))
    reservoir_params = np.random.uniform(0, 2*np.pi, n_params)

    # Normalize input
    data_min, data_max = input_data.min(), input_data.max()
    scaled = 2 * np.pi * (input_data - data_min) / (data_max - data_min + 1e-10)

    param_idx = 0
    input_idx = 0
    n_inputs = len(input_data)

    for layer in range(N_LAYERS):
        # Input encoding
        for q in range(N_QUBITS):
            qc.sx(q)
            qc.rz(scaled[input_idx % n_inputs], q)
            input_idx += 1

        # Reservoir dynamics
        for q in range(N_QUBITS):
            qc.rz(reservoir_params[param_idx], q)
            param_idx += 1
            qc.sx(q)
            qc.rz(reservoir_params[param_idx], q)
            param_idx += 1

        # Entangling layer following Heavy-Hex edges
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


def main():
    """Generate and export the Hero circuit."""
    print("=" * 60)
    print("Generating 156-Qubit Hero Circuit")
    print("=" * 60)

    # Use a sample input (first timestep of spectral data)
    # In production, this would be loaded from training_spectral.npy
    np.random.seed(0)
    sample_input = np.random.randn(100)

    print(f"Building circuit...")
    print(f"  Qubits: {N_QUBITS}")
    print(f"  Layers: {N_LAYERS}")

    circuit = build_hero_circuit(sample_input)

    print(f"  Depth: {circuit.depth()}")
    print(f"  Gates: {sum(circuit.count_ops().values())}")

    # Export to QASM 3.0
    print(f"\nExporting to OpenQASM 3.0...")
    qasm_str = qasm3_dumps(circuit)

    with open('3_hero_156q_kingston.qasm', 'w') as f:
        f.write(qasm_str)

    print(f"✓ Saved to 3_hero_156q_kingston.qasm")
    print(f"  File size: {len(qasm_str):,} bytes")

    # Also save circuit stats
    stats = {
        'n_qubits': N_QUBITS,
        'n_layers': N_LAYERS,
        'depth': circuit.depth(),
        'gate_counts': dict(circuit.count_ops()),
        'total_gates': sum(circuit.count_ops().values()),
    }

    import json
    with open('hero_circuit_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Saved stats to hero_circuit_stats.json")


if __name__ == "__main__":
    main()
