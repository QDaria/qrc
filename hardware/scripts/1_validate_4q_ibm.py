#!/usr/bin/env python3
"""
QDaria QRC - Phase 1: 4-Qubit IBM Validation
=============================================
Runs the validated 4-qubit QRC circuit on IBM real hardware.

Target: ibm_osaka, ibm_kyoto, or ibm_brisbane (Free Tier)
Goal: Validate Forecast Horizon > 4.0 Lyapunov times with real noise
"""

import numpy as np
from datetime import datetime
import json
import os

# === CONFIGURATION ===
N_QUBITS = 4
N_LAYERS = 8
N_SHOTS = 4000
N_SAMPLES = 50  # Number of timesteps to sample

# Preferred backends (Free Tier)
PREFERRED_BACKENDS = ['ibm_osaka', 'ibm_kyoto', 'ibm_brisbane', 'ibm_sherbrooke']

# === IMPORTS ===
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class IBMValidator:
    """4-Qubit QRC validation on IBM hardware."""

    def __init__(self):
        self.service = None
        self.backend = None
        self.backend_name = None

    def connect(self) -> bool:
        """Connect to IBM Quantum - tries both channels."""
        print("Connecting to IBM Quantum...")

        # Try 1: IBM Quantum Platform (standard free tier)
        try:
            self.service = QiskitRuntimeService(channel='ibm_quantum')
            print("  ✓ Connected via ibm_quantum channel")
            return True
        except Exception as e1:
            print(f"  ibm_quantum failed: {e1}")

        # Try 2: IBM Cloud
        try:
            self.service = QiskitRuntimeService(channel='ibm_cloud')
            print("  ✓ Connected via ibm_cloud channel")
            return True
        except Exception as e2:
            print(f"  ibm_cloud failed: {e2}")

        # Try 3: Default (auto-detect saved credentials)
        try:
            self.service = QiskitRuntimeService()
            print("  ✓ Connected via default channel")
            return True
        except Exception as e3:
            print(f"  Default failed: {e3}")

        print("\nERROR: Could not connect to IBM Quantum")
        print("Please run:")
        print("  from qiskit_ibm_runtime import QiskitRuntimeService")
        print("  QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
        return False

    def select_backend(self) -> bool:
        """Select the best available backend."""
        print("\nSelecting backend...")

        try:
            backends = self.service.backends(
                filters=lambda b: (
                    b.status().operational
                    and not b.configuration().simulator
                    and b.configuration().n_qubits >= N_QUBITS
                )
            )

            if not backends:
                print("  No operational backends available")
                return False

            # Sort by preference and queue length
            backend_info = []
            for b in backends:
                name = b.name
                queue = b.status().pending_jobs
                priority = PREFERRED_BACKENDS.index(name) if name in PREFERRED_BACKENDS else 100
                backend_info.append((name, queue, priority, b))

            backend_info.sort(key=lambda x: (x[2], x[1]))

            self.backend = backend_info[0][3]
            self.backend_name = backend_info[0][0]

            print(f"  ✓ Selected: {self.backend_name}")
            print(f"    Queue: {backend_info[0][1]} jobs")
            print(f"    Qubits: {self.backend.configuration().n_qubits}")

            return True

        except Exception as e:
            print(f"  Error selecting backend: {e}")
            return False

    def build_qrc_circuit(self, input_data: np.ndarray) -> QuantumCircuit:
        """Build the validated 4-qubit QRC circuit."""
        qr = QuantumRegister(N_QUBITS, 'q')
        cr = ClassicalRegister(N_QUBITS, 'c')
        qc = QuantumCircuit(qr, cr)

        # Fixed reservoir parameters (same as simulator)
        np.random.seed(42)
        n_params = N_LAYERS * (N_QUBITS * 2 + N_QUBITS)
        reservoir_params = np.random.uniform(0, 2*np.pi, n_params)

        # Normalize input
        data_min, data_max = input_data.min(), input_data.max()
        scaled_all = 2 * np.pi * (input_data - data_min) / (data_max - data_min + 1e-10)

        param_idx = 0
        input_idx = 0
        n_inputs = len(input_data)

        # Interleaved input encoding + reservoir dynamics
        for layer in range(N_LAYERS):
            # Input encoding
            for i in range(N_QUBITS):
                qc.ry(scaled_all[input_idx % n_inputs], i)
                input_idx += 1
                qc.rz(scaled_all[input_idx % n_inputs], i)
                input_idx += 1

            # Reservoir dynamics
            for i in range(N_QUBITS):
                qc.rz(reservoir_params[param_idx], i)
                param_idx += 1
                qc.ry(reservoir_params[param_idx], i)
                param_idx += 1

            # Entangling layer (ring topology)
            for i in range(N_QUBITS):
                j = (i + 1) % N_QUBITS
                qc.cx(i, j)
                qc.rz(reservoir_params[param_idx], j)
                param_idx += 1

        qc.measure(qr, cr)
        return qc

    def get_expectation_values(self, counts: dict) -> np.ndarray:
        """Extract expectation values from measurement counts."""
        total_shots = sum(counts.values())

        # Single-qubit <Z_i>
        z_exp = np.zeros(N_QUBITS)
        for bitstring, count in counts.items():
            for i in range(N_QUBITS):
                bit = int(bitstring[-(i+1)]) if i < len(bitstring) else 0
                z_exp[i] += (1 - 2*bit) * count
        z_exp /= total_shots

        # Two-qubit <Z_i Z_j>
        n_corr = N_QUBITS * (N_QUBITS - 1) // 2
        zz_exp = np.zeros(n_corr)
        corr_idx = 0
        for i in range(N_QUBITS):
            for j in range(i+1, N_QUBITS):
                for bitstring, count in counts.items():
                    bit_i = int(bitstring[-(i+1)]) if i < len(bitstring) else 0
                    bit_j = int(bitstring[-(j+1)]) if j < len(bitstring) else 0
                    zz_exp[corr_idx] += (1 - 2*bit_i) * (1 - 2*bit_j) * count
                corr_idx += 1
        zz_exp /= total_shots

        return np.concatenate([z_exp, zz_exp])

    def run_validation(self, spectral_data: np.ndarray):
        """Run validation on real hardware."""
        print(f"\n{'='*60}")
        print("Running 4-Qubit QRC on Real Hardware")
        print(f"{'='*60}")
        print(f"Backend: {self.backend_name}")
        print(f"Samples: {N_SAMPLES}")
        print(f"Shots: {N_SHOTS}")

        # Select timesteps
        indices = np.linspace(0, len(spectral_data)-1, N_SAMPLES, dtype=int)

        # Build circuits
        print("\nBuilding circuits...")
        circuits = []
        for idx in indices:
            qc = self.build_qrc_circuit(spectral_data[idx])
            circuits.append(qc)

        # Transpile
        print("Transpiling for target backend...")
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=3)
        transpiled = pm.run(circuits)

        print(f"  Circuit depth: {transpiled[0].depth()}")
        print(f"  Gate count: {sum(transpiled[0].count_ops().values())}")

        # Execute
        print(f"\nSubmitting {len(transpiled)} circuits to {self.backend_name}...")
        sampler = Sampler(self.backend)
        job = sampler.run(transpiled, shots=N_SHOTS)

        job_id = job.job_id()
        print(f"  Job ID: {job_id}")
        print("  Waiting for results (this may take several minutes)...")

        result = job.result()

        # Extract reservoir states
        n_features = N_QUBITS + N_QUBITS * (N_QUBITS - 1) // 2
        reservoir_states = np.zeros((N_SAMPLES, n_features))

        for i, pub_result in enumerate(result):
            counts = pub_result.data.c.get_counts()
            reservoir_states[i] = self.get_expectation_values(counts)

        print(f"\n✓ Execution complete")
        print(f"  Reservoir states shape: {reservoir_states.shape}")

        return reservoir_states, indices, job_id


def main():
    """Main execution."""
    print("=" * 60)
    print("QDaria QRC - Phase 1: IBM 4-Qubit Validation")
    print("=" * 60)

    # Load spectral data
    if os.path.exists('training_spectral.npy'):
        spectral_data = np.load('training_spectral.npy')
    elif os.path.exists('../data/training_spectral.npy'):
        spectral_data = np.load('../data/training_spectral.npy')
    else:
        print("ERROR: training_spectral.npy not found")
        print("Please upload this file from the original project")
        return

    print(f"Loaded spectral data: {spectral_data.shape}")

    # Initialize validator
    validator = IBMValidator()

    if not validator.connect():
        return

    if not validator.select_backend():
        return

    # Run validation
    reservoir_states, indices, job_id = validator.run_validation(spectral_data)

    # Save results
    np.save('reservoir_states_ibm_4q.npy', reservoir_states)
    np.save('ibm_sample_indices.npy', indices)

    result = {
        'status': 'SUCCESS',
        'backend': validator.backend_name,
        'job_id': job_id,
        'n_samples': N_SAMPLES,
        'n_shots': N_SHOTS,
        'reservoir_shape': list(reservoir_states.shape),
        'timestamp': datetime.now().isoformat(),
    }

    with open('ibm_4q_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Backend: {validator.backend_name}")
    print(f"Job ID: {job_id}")
    print(f"Results saved to:")
    print(f"  - reservoir_states_ibm_4q.npy")
    print(f"  - ibm_sample_indices.npy")
    print(f"  - ibm_4q_result.json")
    print("\nNext: Run readout training to compute Forecast Horizon")


if __name__ == "__main__":
    main()
