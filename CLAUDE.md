# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantum Reservoir Computing (QRC) research project: a 156-qubit experiment on IBM Heron r2 (`ibm_fez`, open plan) — a larger qubit count than previously reported QRC hardware runs, not better learning — plus 4-qubit hardware and 9-qubit simulation baselines. Hardware provenance (job IDs, dates) lives in `hardware/PROVENANCE.md`. Implements the Steinegger-Räth (2025) feature engineering methodology for chaotic system prediction.

## Build/Run Commands

### Prerequisites
```bash
pip install qiskit qiskit-aer numpy scipy scikit-learn matplotlib
```

### Run Simulations
```bash
cd scripts
python simulate_rigetti_novera_9q.py      # 9Q turbulence simulation
python simulate_9q_lorenz63.py            # Lorenz-63 chaotic attractor
python simulate_9q_rossler.py             # Rössler system
python simulate_rigetti_cepheus_36q.py    # 36Q batch simulation
```

### Generate Figures
```bash
cd scripts
python generate_figure2.py                # Forecast trajectories
python generate_figure3.py                # Sample efficiency analysis
python generate_figure4.py                # Hardware topology comparison
python generate_comprehensive_figures.py  # All figures
```

### Build Paper (LaTeX)
```bash
cd paper
pdflatex qrc_paper_clean.tex
bibtex qrc_paper_clean
pdflatex qrc_paper_clean.tex
pdflatex qrc_paper_clean.tex
```

## Architecture

### Core Components

**Steinegger-Räth Feature Engineering Pipeline:**
- Temporal multiplexing (V=5): 5 virtual nodes per physical qubit
- Spatial multiplexing (r=3): 3 independent reservoir initializations
- Polynomial expansion (G=4): Degree-4 polynomial features
- Ridge regularization with cross-validated alpha

**Simulation Scripts** (`scripts/`):
- `simulate_*.py`: QRC simulations using Qiskit Aer with noise models
- `generate_figure*.py`: Publication figure generation from JSON results
- Import `qrc_steinegger_utils` module for shared QRC utilities (SteineggerQRC class, Lyapunov time calculations)

**Data Flow:**
1. Load time series data from `data/training_spectral.npy` (1000×100)
2. Run QRC simulation → output to `data/*.json`
3. Generate figures → output to `paper/figures/`

### Key Parameters (from scripts)
```python
QRC_CONFIG = {
    'n_qubits': 9,
    'n_layers': 8,
    'V': 5,           # Temporal multiplexing
    'r': 3,           # Spatial multiplexing
    'G': 4,           # Polynomial degree
    'scale_range': (0, 1),
    'random_seed': 42
}
SHOTS = 4000
```

## Directory Structure

- `scripts/` - Python simulation and figure generation scripts
- `data/` - JSON results from simulations (validation_results.json, etc.)
- `paper/` - LaTeX source, compiled PDF, and figures
- `arxiv/` - arXiv submission files
## 🚨 MANDATORY: Visual Validation Before Delivery

**CRITICAL**: Before suggesting to publish or deliver ANY PDF:
1. **Build LaTeX properly**: `pdflatex` → `bibtex` → `pdflatex` → `pdflatex` (3-4 passes)
2. **Check for [?] citations**: `grep "undefined" *.log`
3. **Visually verify with Playwright**: Take screenshot of PDF to confirm citations, figures render
4. **Never trust "it should work"** - PROVE IT with visual evidence

If uncertain about output quality, apply iterative refinement (up to 12 cycles) until confident in the result.