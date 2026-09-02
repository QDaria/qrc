# Hardware run provenance

Reconstructed 2026-09-02 from the original working directory of this paper (result JSONs, execution
scripts and reservoir-state arrays recovered from the author's archive). This file exists because the
paper, the README, the Zenodo description and the company blog post gave three incompatible accounts
of the hardware (Heron r2 vs r3; `ibm_fez` vs `ibm_pittsburgh`; open plan vs IBM Quantum Network).
The result files settle it. Nothing on disk or in any deposit names `ibm_pittsburgh` in code, results
or logs; that name, "Heron r3", "IBM Quantum Network" and a calibration table entered the text between
the 2 December and 12 December 2025 drafts.

## What actually ran

| Run | Backend | Job ID | Timestamp (result file) | Samples | Shots | Result file |
|---|---|---|---|---|---|---|
| 4-qubit validation | `ibm_fez` | `d4kdprd74pkc7386uov0` | 2025-11-27T23:22:23 | 50 | 4,000 | `jobs/ibm_4q_result.json` |
| 156-qubit pilot (discarded: same circuit ×10) | `ibm_fez` | `d4kqhgh0i6jc73df4690` | 2025-11-28T13:51:31 | 10 | 4,000 | `jobs/hero_run_result.json` |
| 156-qubit run | `ibm_fez` | `d4m7kg10i6jc73dgg1n0` | 2025-11-30T17:12:35 | 50 | 4,000 | `jobs/hero_run_fixed_result.json` |
| 156-qubit extension | `ibm_fez` | `d4m87dl74pkc7388nll0` | 2025-11-30 (npy written 19:03) | 150 | 4,000 | recorded in the author's `retrieve_156q_full.py`; submitted by `scripts/run_append_150.py` |

- **Processor:** `ibm_fez` is an IBM Heron r2, 156 qubits, heavy-hex lattice. There was no Heron r3
  and no `ibm_pittsburgh` in any run. The blog post (Heron r2) was right; the paper and README were wrong.
- **Access route:** `QiskitRuntimeService` on the `ibm_quantum_platform` channel, `instance='open-instance'`,
  "Standard Free Tier" per the author's `setup_ibm.py` — i.e. the IBM Quantum Platform **open plan**,
  not the IBM Quantum Network. (That setup script is deliberately not included here.)
- **Primitive:** `SamplerV2`, 4,000 shots per circuit, transpiled with
  `generate_preset_pass_manager(optimization_level=3)`. All reservoir-state values are exact multiples
  of 1/4000. The 156-qubit circuit: 3 layers, depth 73, 1,257 CZ pre-transpile
  (`jobs/hero_circuit_stats.json`); 9,395 two-qubit gates post-transpile (`jobs/hero_run_result.json`).
- **4-qubit circuit:** 4 qubits of `ibm_fez`, 8 layers of parameterised single-qubit rotations each
  followed by a ring-coupled CX entangling layer; readout 4 ⟨Z⟩ + 6 ⟨ZZ⟩ = 10 features
  (`scripts/1_validate_4q_ibm.py`). The paper's earlier "IBM Canary r2, 4Q linear chain" description
  was not real.
- **Calibration:** no `backend.properties()` snapshot was saved for any run. The former Appendix C
  calibration table was removed for that reason.
- **Job IDs** are visible only to the IBM account that submitted them; they are published here as the
  audit trail the author can produce on request.

## Data lineage

| File | Shape | Origin |
|---|---|---|
| `../data/training_spectral.npy` | 1000 × 100 | Spectral turbulence series used as input |
| `../data/reservoir_states_ibm_4q.npy` | 50 × 10 | 4Q validation, job `d4kdprd74pkc7386uov0`; indices in `../data/ibm_sample_indices.npy` |
| `../data/reservoir_states_hero_156q_200.npy` | 200 × 156 | 156Q, assembled 2025-11-30 from the 50-sample run (`d4m7kg10i6jc73dgg1n0`) plus the 150-sample extension (`d4m87dl74pkc7388nll0`); indices in `../data/hero_sample_indices_200.npy` |
| `jobs/raw_counts.json` | 425 KB | Raw counts of the **unused Rigetti Ankaa-3** attempt (36-bit strings), not the IBM runs |
| IBM raw counts (not in repo) | 130 MB | `raw_counts_156q_archive.json`, archived by the author; available on request |

`../data/validation_results.json` (validation timestamp 2025-12-07) recomputes the paper's numbers
from the two reservoir-state files: 4Q test R² 0.7635 (paper 0.764), 156Q test R² 0.7228 (paper 0.723),
RMSE 608.05 and 566.43. Those R² values are scikit-learn `r2_score` on the **flattened** test-set
spectra (all wavenumbers pooled, variance about the global mean). The per-wavenumber averaged R²
(scikit-learn's default multi-output setting) from the same predictions is 0.32 (4Q) and 0.14 (156Q).
The paper now states this definition.

## What was attempted and not used

- Ten jobs on Rigetti Ankaa-3 via qBraid (`rigetti_ankaa_3-mo-qjob-…`, `jobs/checkpoint.json`,
  `scripts/retrieve_8_jobs.py`): a 36-qubit hardware attempt that returned unusable (NaN) reservoirs.
  Every Rigetti result in the paper is a Qiskit Aer simulation with a Novera-style noise model.
- An earlier 156-qubit submission (`d4kpqr10i6jc73df3gdg`) was cancelled; `scripts/3_rescue_hero_job.py`
  refers to it.
- A 4-qubit Lorenz-63 hardware run on `ibm_fez`, 1 Dec 2025 (six jobs including `d4mu8ash0bas73fatacg`,
  500 samples, `jobs/qrc_checkpoint.json`, `jobs/qrc_results.json`): abandoned (test R² strongly
  negative). Not in the paper.
- Post-publication runs in January 2026 on `ibm_torino`, `ibm_fez` and `ibm_marrakesh` are not part of
  this preprint.

## What this does and does not support

- Supported: "156-qubit QRC on IBM Heron r2 hardware (`ibm_fez`), 200 samples, 4,000 shots", with
  dates and job IDs, and R² 0.723 under the flattened-spectrum definition.
- Supported, qualified: "a larger qubit count than previously reported QRC hardware experiments, to
  our knowledge" — Kornjača et al. (2024) ran 108 neutral atoms on an analog device; Yasuda et al.
  (2023) ran 120 superconducting qubits and call it preliminary. The count is larger; the learning is
  not better, which is the paper's own point.
- Not supported: "largest QRC demonstration on real hardware" unqualified, "first", "Heron r3",
  "ibm_pittsburgh", "IBM Quantum Network", the "Canary r2" 4-qubit processor, the former calibration
  table, and the former "18–36 hour queue" claim.
