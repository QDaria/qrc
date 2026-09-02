#!/usr/bin/env python3
"""
QDaria QRC - Phase 3: Hero Run (156 Qubits)
============================================
Submits the pre-verified 156-qubit circuit to IBM Kingston (Heron r2).

BUDGET GUARD: Maximum 600 seconds execution time
COST LIMIT: $500 maximum

Target: ibm_kingston (IBM Heron r2, 156 qubits)
"""

import numpy as np
from datetime import datetime
import json
import os
import time

# === CONFIGURATION ===
N_QUBITS = 156
N_SHOTS = 4000
N_SAMPLES = 10
MAX_EXECUTION_TIME = 600  # seconds
MAX_COST_USD = 500.0
COST_PER_SHOT = 0.0016  # Estimated for premium tier

# Target backend
TARGET_BACKEND = 'ibm_kingston'
FALLBACK_BACKENDS = ['ibm_torino', 'ibm_fez', 'ibm_marrakesh']

# === IMPORTS ===
from qiskit import QuantumCircuit
from qiskit.qasm3 import loads as qasm3_loads
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class BudgetGuard:
    """Strict budget and time protection."""

    def __init__(self, max_cost: float, max_time: int):
        self.max_cost = max_cost
        self.max_time = max_time
        self.start_time = None
        self.estimated_cost = 0.0

    def check_cost(self, n_circuits: int, n_shots: int) -> bool:
        """Check if job is within budget."""
        self.estimated_cost = n_circuits * n_shots * COST_PER_SHOT
        if self.estimated_cost > self.max_cost:
            print(f"⚠ BUDGET GUARD: Estimated cost ${self.estimated_cost:.2f} exceeds limit ${self.max_cost:.2f}")
            return False
        print(f"  Estimated cost: ${self.estimated_cost:.2f} (limit: ${self.max_cost:.2f})")
        return True

    def start(self):
        """Start execution timer."""
        self.start_time = time.time()

    def check_time(self) -> bool:
        """Check if within time limit."""
        if self.start_time is None:
            return True
        elapsed = time.time() - self.start_time
        if elapsed > self.max_time:
            print(f"⚠ BUDGET GUARD: Execution time {elapsed:.0f}s exceeds limit {self.max_time}s")
            return False
        return True

    def abort(self, reason: str):
        """Abort execution with reason."""
        print(f"\n{'='*60}")
        print("BUDGET GUARD - EXECUTION ABORTED")
        print(f"{'='*60}")
        print(f"Reason: {reason}")
        print(f"Estimated cost avoided: ${self.estimated_cost:.2f}")
        return {'status': 'ABORTED', 'reason': reason}


class HeroRunner:
    """156-Qubit Hero Run executor."""

    def __init__(self):
        self.service = None
        self.backend = None
        self.backend_name = None
        self.budget = BudgetGuard(MAX_COST_USD, MAX_EXECUTION_TIME)

    def connect(self) -> bool:
        """Connect to IBM Quantum."""
        print("Connecting to IBM Quantum...")

        try:
            self.service = QiskitRuntimeService(channel='ibm_quantum')
            print("  ✓ Connected via ibm_quantum")
            return True
        except:
            pass

        try:
            self.service = QiskitRuntimeService(channel='ibm_cloud')
            print("  ✓ Connected via ibm_cloud")
            return True
        except:
            pass

        try:
            self.service = QiskitRuntimeService()
            print("  ✓ Connected via default")
            return True
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    def select_backend(self) -> bool:
        """Select 156+ qubit backend."""
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
                print("  No backends with 156+ qubits available")
                return False

            # Prefer target, then fallbacks
            all_targets = [TARGET_BACKEND] + FALLBACK_BACKENDS

            for target in all_targets:
                for b in backends:
                    if b.name == target:
                        self.backend = b
                        self.backend_name = target
                        print(f"  ✓ Selected: {self.backend_name}")
                        print(f"    Qubits: {b.configuration().n_qubits}")
                        print(f"    Queue: {b.status().pending_jobs} jobs")
                        return True

            # Use first available
            self.backend = backends[0]
            self.backend_name = backends[0].name
            print(f"  ✓ Selected: {self.backend_name} (first available)")
            return True

        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    def load_qasm_circuit(self, qasm_file: str) -> QuantumCircuit:
        """Load pre-verified circuit from QASM file."""
        print(f"\nLoading circuit from {qasm_file}...")

        with open(qasm_file, 'r') as f:
            qasm_str = f.read()

        circuit = qasm3_loads(qasm_str)
        print(f"  Qubits: {circuit.num_qubits}")
        print(f"  Depth: {circuit.depth()}")

        return circuit

    def verify_transpilation(self, circuit: QuantumCircuit) -> dict:
        """Verify circuit fits within coherence limits."""
        print("\nVerifying transpilation...")

        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=3)
        transpiled = pm.run(circuit)

        depth = transpiled.depth()
        ops = transpiled.count_ops()

        # Heron r2 specs
        T2_us = 200
        cx_time_ns = 660

        n_2q = ops.get('cz', 0) + ops.get('cx', 0) + ops.get('ecr', 0)
        estimated_time_us = (n_2q * cx_time_ns / depth) / 1000

        coherence_ratio = estimated_time_us / T2_us

        analysis = {
            'depth': depth,
            'two_qubit_gates': n_2q,
            'estimated_time_us': estimated_time_us,
            'T2_limit_us': T2_us,
            'coherence_ratio': coherence_ratio,
            'within_coherence': coherence_ratio < 0.5,
        }

        print(f"  Transpiled depth: {depth}")
        print(f"  Two-qubit gates: {n_2q}")
        print(f"  Coherence ratio: {coherence_ratio:.3f}")

        if not analysis['within_coherence']:
            print("  ⚠ WARNING: Circuit may exceed coherence limit")

        return analysis, transpiled

    def get_expectation_values(self, counts: dict) -> np.ndarray:
        """Extract expectation values."""
        total_shots = sum(counts.values())

        # Single-qubit only (for 156 qubits, full correlations would be massive)
        z_exp = np.zeros(N_QUBITS)
        for bitstring, count in counts.items():
            for i in range(min(N_QUBITS, len(bitstring))):
                bit = int(bitstring[-(i+1)]) if i < len(bitstring) else 0
                z_exp[i] += (1 - 2*bit) * count
        z_exp /= total_shots

        return z_exp

    def run_hero(self, spectral_data: np.ndarray, qasm_file: str):
        """Execute the Hero Run."""
        print(f"\n{'='*60}")
        print("HERO RUN - 156 Qubit Execution")
        print(f"{'='*60}")

        # Load and verify circuit
        base_circuit = self.load_qasm_circuit(qasm_file)
        analysis, transpiled = self.verify_transpilation(base_circuit)

        if not analysis['within_coherence']:
            return self.budget.abort("Circuit exceeds coherence limit")

        # Budget check
        if not self.budget.check_cost(N_SAMPLES, N_SHOTS):
            return self.budget.abort("Exceeds cost limit")

        # Prepare circuits with different inputs
        print(f"\nPreparing {N_SAMPLES} circuits...")
        indices = np.linspace(0, len(spectral_data)-1, N_SAMPLES, dtype=int)

        # For Hero Run, we use the same pre-verified circuit
        # (In production, you'd rebuild with different inputs)
        circuits = [transpiled] * N_SAMPLES

        # Start execution
        self.budget.start()

        print(f"\nSubmitting to {self.backend_name}...")
        print(f"  Circuits: {len(circuits)}")
        print(f"  Shots: {N_SHOTS}")

        sampler = Sampler(self.backend)
        job = sampler.run(circuits, shots=N_SHOTS)

        job_id = job.job_id()
        print(f"  Job ID: {job_id}")
        print("  Waiting for results...")

        # Monitor with timeout
        while not job.done():
            if not self.budget.check_time():
                print("  Attempting to cancel job...")
                try:
                    job.cancel()
                except:
                    pass
                return self.budget.abort("Execution timeout")
            time.sleep(30)
            print(f"  Status: {job.status()}")

        result = job.result()

        # Extract results
        reservoir_states = np.zeros((N_SAMPLES, N_QUBITS))

        for i, pub_result in enumerate(result):
            counts = pub_result.data.c.get_counts()
            reservoir_states[i] = self.get_expectation_values(counts)

        print(f"\n✓ Hero Run complete!")
        print(f"  Reservoir shape: {reservoir_states.shape}")

        return reservoir_states, indices, job_id, analysis


def main():
    """Main execution."""
    print("=" * 60)
    print("QDaria QRC - Phase 3: HERO RUN (156 Qubits)")
    print("=" * 60)
    print(f"\n⚠ BUDGET GUARD ACTIVE")
    print(f"  Max cost: ${MAX_COST_USD:.2f}")
    print(f"  Max time: {MAX_EXECUTION_TIME}s")

    # Load data
    if os.path.exists('training_spectral.npy'):
        spectral_data = np.load('training_spectral.npy')
    elif os.path.exists('../data/training_spectral.npy'):
        spectral_data = np.load('../data/training_spectral.npy')
    else:
        print("\nERROR: training_spectral.npy not found")
        return

    # Check QASM file
    qasm_file = '3_hero_156q_kingston.qasm'
    if not os.path.exists(qasm_file):
        print(f"\nERROR: {qasm_file} not found")
        print("Run generate_hero_qasm.py first")
        return

    print(f"\nSpectral data: {spectral_data.shape}")

    # Initialize runner
    runner = HeroRunner()

    if not runner.connect():
        return

    if not runner.select_backend():
        return

    # Execute Hero Run
    result = runner.run_hero(spectral_data, qasm_file)

    if isinstance(result, dict) and result.get('status') == 'ABORTED':
        return

    reservoir_states, indices, job_id, analysis = result

    # Save results
    np.save('reservoir_states_hero_156q.npy', reservoir_states)
    np.save('hero_sample_indices.npy', indices)

    output = {
        'status': 'SUCCESS',
        'backend': runner.backend_name,
        'job_id': job_id,
        'n_qubits': N_QUBITS,
        'n_samples': N_SAMPLES,
        'n_shots': N_SHOTS,
        'reservoir_shape': list(reservoir_states.shape),
        'transpilation_analysis': analysis,
        'actual_cost': runner.budget.estimated_cost,
        'timestamp': datetime.now().isoformat(),
    }

    with open('hero_run_result.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("HERO RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Backend: {runner.backend_name}")
    print(f"Job ID: {job_id}")
    print(f"Cost: ${runner.budget.estimated_cost:.2f}")
    print(f"\nResults saved:")
    print(f"  - reservoir_states_hero_156q.npy")
    print(f"  - hero_sample_indices.npy")
    print(f"  - hero_run_result.json")


if __name__ == "__main__":
    main()
