# Hardware run provenance

Reconstructed 2026-09-02 from the original working directory of this paper (result JSONs, execution
scripts and reservoir-state arrays recovered from the author's archive). This file exists because the
paper, the README, the Zenodo description and the company blog post gave three incompatible accounts
of the hardware (Heron r2 vs r3; `ibm_fez` vs `ibm_pittsburgh`; open plan vs IBM Quantum Network).
The result files settle it.

## What actually ran

| Run | Backend | Job ID | Timestamp (result file) | Samples | Shots | Result file |
|---|---|---|---|---|---|---|
| 4-qubit validation | `ibm_fez` | `d4kdprd74pkc7386uov0` | 2025-11-27T23:22:23 | 50 | 4,000 | `jobs/ibm_4q_result.json` |
| 156-qubit pilot | `ibm_fez` | `d4kqhgh0i6jc73df4690` | 2025-11-28T13:51:31 | 10 | 4,000 | `jobs/hero_run_result.json` |
| 156-qubit run | `ibm_fez` | `d4m7kg10i6jc73dgg1n0` | 2025-11-30T17:12:35 | 50 | 4,000 | `jobs/hero_run_fixed_result.json` |
| 156-qubit extension | `ibm_fez` | `d4mu8ash0bas73fatacg` | 2025-12-01T18:54:33 | 300 | 4,000 | `jobs/qrc_checkpoint.json`, `jobs/qrc_results.json` |

- **Processor:** `ibm_fez` is an IBM Heron r2, 156 qubits, heavy-hex lattice. There was no Heron r3
  and no `ibm_pittsburgh` in any run. The blog post (Heron r2) was right; the paper and README were wrong.
- **Access route:** `QiskitRuntimeService` on the `ibm_quantum_platform` channel, "Standard Free
  Tier" per `scripts/setup_ibm.py` in the original archive — i.e. the IBM Quantum Platform **open
  plan**, not the IBM Quantum Network. (That setup script is deliberately not included here.)
- **Primitive:** `SamplerV2`, 4,000 shots per circuit, transpiled with
  `generate_preset_pass_manager(optimization_level=3)`. The 156-qubit pilot transpiled to 9,395
  two-qubit gates (`jobs/hero_circuit_stats.json`, `jobs/hero_run_result.json`).
- **Job IDs** are visible only to the IBM account that submitted them; they are published here as the
  audit trail the author can produce on request.

## Data lineage

| File | Shape | Origin |
|---|---|---|
| `../data/training_spectral.npy` | 1000 × 100 | Spectral turbulence series used as input |
| `../data/reservoir_states_ibm_4q.npy` | 50 × 10 | 4Q validation, job `d4kdprd74pkc7386uov0`; sample indices in `../data/ibm_sample_indices.npy` |
| `../data/reservoir_states_hero_156q_200.npy` | 200 × 156 | 156Q, assembled 2025-11-30 from the 50-sample run (`d4m7kg10i6jc73dgg1n0`) plus 150 samples submitted by `scripts/run_append_150.py`; indices in `../data/hero_sample_indices_200.npy` |
| `jobs/raw_counts.json` | 425 KB | Raw measurement counts for the 156Q runs |

`../data/validation_results.json` (validation timestamp 2025-12-07) recomputes the paper's R² values
from these two reservoir-state files: 4Q test R² 0.7635 (paper: 0.764), 156Q test R² 0.7228
(paper: 0.723).

The job identifier of the 150-sample extension was written by `run_append_150.py` to a local
`append_job_id.txt` that was not archived; the 300-sample run of 2025-12-01 (`d4mu8ash0bas73fatacg`)
post-dates the 200-sample file and is not used in the paper.

## What was attempted and not used

Ten jobs on Rigetti Ankaa-3 via qBraid (`rigetti_ankaa_3-mo-qjob-…`, listed in `jobs/checkpoint.json`
and `scripts/retrieve_8_jobs.py`) were a 36-qubit hardware attempt. Their data do not appear in the
paper; every Rigetti result reported is a Qiskit Aer simulation with a Novera-style noise model.

## What this does and does not support

- Supported: "156-qubit QRC on IBM Heron r2 hardware (`ibm_fez`), 200 samples, 4,000 shots", with
  dates and job IDs.
- Supported with the qualifier the paper now uses: "to our knowledge the largest QRC demonstration
  on **gate-based superconducting** hardware". Kornjača et al. (2024) ran 108 neutral atoms on an
  analog device; Yasuda et al. (2023) ran 120 superconducting qubits and call it preliminary.
- Not supported: "largest on real quantum hardware" without qualification, "Heron r3",
  "ibm_pittsburgh", "IBM Quantum Network".
- Not re-verified: the calibration values in the paper's Appendix table were not checked against
  `ibm_fez` calibration data from the run dates.
