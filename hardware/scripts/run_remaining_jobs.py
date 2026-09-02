"""
Run remaining 2 jobs (8-9) to complete the 10-sample dataset.
Uses the fast client.get_job_results() method.
"""

import numpy as np
import json
import os
from qbraid.runtime import QbraidProvider
from braket.circuits import Circuit

# Config
DATA_FILE = "training_spectral.npy"
OUTPUT_DIR = "quantum_results"
DEVICE_ID = "rigetti_ankaa_3"
SHOTS = 1000
N_QUBITS = 36
N_SAMPLES = 10
NORM_RANGE = (-np.pi, np.pi)

# Load existing results
COUNTS_FILE = os.path.join(OUTPUT_DIR, "raw_counts.json")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

with open(COUNTS_FILE) as f:
    results_dict = json.load(f)
with open(CHECKPOINT_FILE) as f:
    checkpoint = json.load(f)

start_index = len(results_dict)
print(f"Found {start_index}/10 completed. Running jobs {start_index}-{N_SAMPLES-1}.")

if start_index >= N_SAMPLES:
    print("All 10 jobs done!")
    exit()

# Load and normalize data
spectral_data = np.load(DATA_FILE)
input_data = spectral_data[:N_SAMPLES].copy()
data_min, data_max = input_data.min(), input_data.max()
input_data = (input_data - data_min) / (data_max - data_min)
input_data = input_data * (NORM_RANGE[1] - NORM_RANGE[0]) + NORM_RANGE[0]
print(f"Data normalized to [{NORM_RANGE[0]:.2f}, {NORM_RANGE[1]:.2f}]")

# Connect
provider = QbraidProvider()
device = provider.get_device(DEVICE_ID)
print(f"Device: {DEVICE_ID} - {device.status()}")

# Circuit builder
def build_circuit(features):
    qc = Circuit()
    for i in range(N_QUBITS):
        qc.ry(i, float(features[i % len(features)]))
    for i in range(N_QUBITS - 1):
        qc.cphaseshift(i, i + 1, np.pi / 2)
    qc.cphaseshift(N_QUBITS - 1, 0, np.pi / 2)
    return qc

# Cost estimate
jobs_remaining = N_SAMPLES - start_index
print(f"\nCost: {jobs_remaining} jobs × 120 = {jobs_remaining * 120} credits")

# Run remaining jobs
for i in range(start_index, N_SAMPLES):
    print(f"\nJob {i}/{N_SAMPLES-1}:")
    
    try:
        circuit = build_circuit(input_data[i])
        job = device.run(circuit, shots=SHOTS)
        print(f"  Submitted: {job.id}")
        
        # Wait for completion
        job.wait_for_final_state()
        
        # Get results via fast client method
        result = provider.client.get_job_results(job.id)
        counts = result['measurementCounts']
        print(f"  ✓ {len(counts)} outcomes")
        
        # Save
        results_dict[str(i)] = counts
        checkpoint['completed'][str(i)] = {'job_id': job.id}
        
        with open(COUNTS_FILE, 'w') as f:
            json.dump(results_dict, f)
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f)
        print(f"  ✓ Saved")
        
    except Exception as e:
        print(f"  ✗ {e}")
        break

print(f"\n{'='*50}")
print(f"COMPLETE: {len(results_dict)}/10 jobs")
print(f"Results in {OUTPUT_DIR}/")
