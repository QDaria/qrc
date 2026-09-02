"""
Fast retrieval using provider.client.get_job_results()
"""

import json
import os
from qbraid.runtime import QbraidProvider

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

provider = QbraidProvider()
results_dict = {}

print("Retrieving 8 jobs...")

for i, job_id in enumerate(JOB_IDS):
    try:
        result = provider.client.get_job_results(job_id)
        counts = result['measurementCounts']
        results_dict[str(i)] = counts
        print(f"Job {i}: ✓ {len(counts)} outcomes")
    except Exception as e:
        print(f"Job {i}: ✗ {e}")

# Save
with open(os.path.join(OUTPUT_DIR, "raw_counts.json"), 'w') as f:
    json.dump(results_dict, f)

with open(os.path.join(OUTPUT_DIR, "checkpoint.json"), 'w') as f:
    json.dump({'completed': {str(i): {'job_id': JOB_IDS[i]} for i in range(len(results_dict))}}, f)

print(f"\n✓ Saved {len(results_dict)}/8 jobs to {OUTPUT_DIR}/")
