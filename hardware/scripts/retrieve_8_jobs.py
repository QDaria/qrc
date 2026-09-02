"""
Memory-safe retrieval - processes one job at a time.
"""

import json
import os
import gc

OUTPUT_DIR = "quantum_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

JOB_IDS = [
    "rigetti_ankaa_3-mo-qjob-49mgxkgjcwfwpoys705r",
    "rigetti_ankaa_3-mo-qjob-7imh7s6c820gfs7iecx3",
    "rigetti_ankaa_3-mo-qjob-vapek33y5g0uz13kaba9",
    "rigetti_ankaa_3-mo-qjob-8jo99hi0n9vg19z50zkb",
    "rigetti_ankaa_3-mo-qjob-cpr1ao2li7gwdc0lpwwx",
    "rigetti_ankaa_3-mo-qjob-lxj0nxxd3d6i84iojxlp",
    "rigetti_ankaa_3-mo-qjob-b55nr4utkc8fj1q3qaq1",
    "rigetti_ankaa_3-mo-qjob-8jtydehwfzzkiw0uyejb",
]

CHECKPOINT = os.path.join(OUTPUT_DIR, "checkpoint.json")
COUNTS_FILE = os.path.join(OUTPUT_DIR, "raw_counts.json")

# Load existing progress
completed = {}
results = {}
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        completed = json.load(f).get('completed', {})
if os.path.exists(COUNTS_FILE):
    with open(COUNTS_FILE) as f:
        results = json.load(f)

start = len(completed)
print(f"Starting from job {start}/8")

# Import here to control memory
from qbraid.runtime import load_job

for i in range(start, len(JOB_IDS)):
    job_id = JOB_IDS[i]
    print(f"\nJob {i}: {job_id[-24:]}")
    
    try:
        job = load_job(job_id)
        result = job.result()
        counts = result.data.get_counts()
        
        n = len(counts)
        print(f"  ✓ {n} outcomes")
        
        # Save immediately
        completed[str(i)] = {'job_id': job_id, 'n': n}
        results[str(i)] = {k: int(v) for k, v in counts.items()}
        
        with open(CHECKPOINT, 'w') as f:
            json.dump({'completed': completed}, f)
        with open(COUNTS_FILE, 'w') as f:
            json.dump(results, f)
        
        # Clear memory
        del job, result, counts
        gc.collect()
        
    except Exception as e:
        print(f"  ✗ {e}")
        break

print(f"\nDone: {len(completed)}/8 retrieved")