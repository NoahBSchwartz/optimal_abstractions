# Artifact: Optimized Piecewise Affine Abstractions of Neural Networks with Learnable Activation Functions

This artifact accompanies the paper **"Optimized Piecewise Affine Abstractions of Neural Networks with Learnable Activation Functions"**. It contains all code, pre-trained models, pre-computed logs, and generated plots required to understand and reproduce the experimental results.

---

## Table of Contents

1. [Artifact Structure](#artifact-structure)
2. [Resource Requirements](#resource-requirements)
3. [Setup Instructions](#setup-instructions)
4. [Running Experiments](#running-experiments)
5. [Reproducing Paper Results from Logs](#reproducing-paper-results-from-logs)
6. [Example: Quick Smoke Test](#example-quick-smoke-test)
7. [Configuration Reference](#configuration-reference)
8. [Tested System](#tested-system)

---

## Artifact Structure

```
project_artifact/
├── LICENSE                          # MIT License
├── README.md                        # This file
├── requirements.txt                 # Python package dependencies
├── main.py                          # Main experiment script (paper results)
│
├── src/                             # Supporting modules (used by notebooks)
│   ├── custom_fastkan.py            # FastKAN model definition
│   ├── kan_verification_fitter.py   # PWL fitting (vanilla + DP)
│   ├── kan_verification_milp.py     # MILP encoding and bound checking
│   ├── kan_verification_trainer.py  # Model training routines
│   ├── kan_verification_grapher.py  # Visualization utilities
│   └── kan_verification_experiment_runner.py  # Older experiment runner
│
├── model_pkls/                      # Pre-trained FastKAN and MLP checkpoints
│   ├── pinnheat_kan_model.pkl       # Physics-Informed NN (heat equation)
│   ├── functionxy_kan_model.pkl     # f(x,y) = x*y regression
│   ├── functionexp_kan_model.pkl    # f(x) = exp(x)
│   ├── functionexp4_kan_model.pkl   # f(x) = exp(4x)
│   ├── functionexp100_kan_model.pkl # f(x) = exp(100x)
│   ├── functionlegendre_kan_model.pkl
│   ├── functionbessel_kan_model.pkl
│   ├── funcsphharm_kan_model.pkl
│   ├── funcellipeinc_kan_model.pkl
│   ├── funcellipkinc_kan_model.pkl
│   ├── funcnoise_kan_model.pkl
│   ├── prosthetic_kan_model.pkl
│   ├── weather_kan_model.pkl
│   ├── pm25_kan_model.pkl
│   ├── mnist_kan_model.pkl
│   ├── acopf_ml4acopf_kan_model.pkl
│   ├── acopf_ml4aconpf2_kan_model.pkl
│   └── *_mlp_model.pkl              # Corresponding MLP baselines
│
├── logs/                            # Pre-computed experimental results
│   ├── verification_log.txt         # Full console log from our paper run
│   ├── graph_data_pinnheat.json     # Parsed results for pinnheat network
│   ├── graph_data_functionxy.json
│   ├── graph_data_functionexp.json
│   ├── graph_data_functionexp4.json
│   ├── graph_data_functionexp100.json
│   ├── graph_data_functionlegendre.json
│   ├── graph_data_functionbessel.json
│   ├── graph_data_funcsphharm.json
│   ├── graph_data_funcellipeinc.json
│   ├── graph_data_funcellipkinc.json
│   └── graph_data_pinnheat.json
│
├── plots/                           # All figures from the paper
│   ├── plot_segments_vs_output_width.pdf
│   ├── plot_input_vs_output_width.pdf
│   ├── plot_timing_summary.pdf
│   ├── plot_input_vs_output_width.tex
│   ├── timing_table.tex
│   ├── verification_multi_kan.png   # Fig: 3-panel comparison
│   ├── true_vanilla_comparison.png  # Fig: 3-method comparison
│   └── verification_*.png           # Per-network verification plots
│
└── notebooks/
    ├── results_analysis.ipynb       # Regenerate all paper figures from logs
    └── 3_methods_over_10_networks.ipynb  # Cross-network comparison analysis
```

### What Each File Does

| File | Purpose |
|---|---|
| `main.py` | Runs the full verification pipeline: DP abstraction → ILP allocation → MILP verification → plotting. Produces all paper figures and JSON logs. |
| `src/custom_fastkan.py` | FastKAN model class (copied from [ZiyaoLi/fast-kan](https://github.com/ZiyaoLi/fast-kan)) with `use_layernorm=False` option for verification compatibility. |
| `src/kan_verification_fitter.py` | Vanilla (equal-segment) and DP-optimal PWL fitting algorithms. |
| `src/kan_verification_milp.py` | Gurobi MILP encoding of the PWL-abstracted KAN; interval-analysis solver. |
| `logs/verification_log.txt` | Full stdout from our experiment run (can be used to inspect all numerical results without re-running). |
| `logs/graph_data_*.json` | Structured results (output widths, times) for each network, keyed by segment budget and input width. |
| `notebooks/results_analysis.ipynb` | Reads `logs/graph_data_*.json` and regenerates every figure from the paper. |

---

## Resource Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU cores | 4 | 8+ (parallel MILP solving) |
| Disk space | 500 MB | 1 GB |
| Gurobi license | Academic or commercial | Academic (free for researchers) |

**Gurobi is required.** A free academic license can be obtained from [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/). The software will automatically detect a license installed at the default path (`~/gurobi.lic` on Linux/Mac, `C:\gurobi\gurobi.lic` on Windows) or via the `GRB_LICENSE_FILE` environment variable.

**Runtime estimates (on our test machine: Apple M3 Pro, 16 GB RAM, 11 cores):**

| Experiment scope | Approx. time |
|---|---|
| Single network (pinnheat), all segment budgets | ~20 minutes |
| All 10 function networks, all segment budgets | ~3–5 hours |
| Quick smoke test (k=1 only) | ~2 minutes |

---

## Setup Instructions

Follow these steps exactly. No prior knowledge of the project is assumed.

### Step 1: Install Python

You need Python 3.9 or later. Check your version:

```bash
python3 --version
```

If you don't have Python 3.9+, download it from [python.org](https://www.python.org/downloads/).

### Step 2: Install Gurobi and Activate a License

1. Download Gurobi from [gurobi.com/downloads](https://www.gurobi.com/downloads/).
2. Install it and follow the instructions to place a license file at `~/gurobi.lic` (Linux/Mac) or `C:\gurobi\gurobi.lic` (Windows).
3. For an academic license, register at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/) and run `grbgetkey <key>` as instructed after registration.

Verify Gurobi works:

```bash
python3 -c "import gurobipy; print('Gurobi OK')"
```

### Step 3: Create a Virtual Environment (recommended)

```bash
cd project_artifact          # navigate into this directory
python3 -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
```

### Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch installation may differ depending on your hardware. If the above fails for torch, visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct install command for your OS and GPU setup, then re-run `pip install -r requirements.txt`.

### Step 5: Verify the Setup

Run this quick import check from the `project_artifact/` directory:

```bash
python3 -c "
import gurobipy, torch, numpy, matplotlib, joblib
print('gurobipy:', gurobipy.gurobi.version())
print('torch:', torch.__version__)
print('All imports OK')
"
```

You should see version numbers printed without errors.

---

## Running Experiments

All experiments are run by executing `main.py` from the `project_artifact/` directory.

### Configure Which Models to Run

At the top of `main.py`, edit `FILENAME_FILTER` to select which pre-trained networks to verify:

```python
FILENAME_FILTER = ["pinn"]       # Only the heat-equation PINN (fastest, ~20 min)
FILENAME_FILTER = ["functionxy"] # x*y regression network
FILENAME_FILTER = ["functionexp", "functionlegendre", "functionbessel"]  # several at once
FILENAME_FILTER = ["function", "func", "pinn", "prosthetic", "weather", "pm25"]  # all
```

Any network name listed in `model_pkls/` can be selected by including a substring of its filename.

### Other Tunable Parameters

```python
SEGMENT_COUNTS = [1, 5, 10, 20, 30, 40, 50]  # segment budgets k to sweep
INPUT_WIDTHS   = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]  # input interval widths
TIME_LIMIT     = 500    # Gurobi time limit per MILP solve (seconds)
MIP_GAP        = 0.15   # Gurobi optimality gap (0.15 = 15%)
MAX_SEGMENTS   = 50     # maximum segments for DP precomputation
FIXED_IW       = 5.0    # fixed input width for the segment-sweep experiment
```

### Run the Experiment

```bash
cd project_artifact
python3 main.py
```

Output is written to both the terminal and `verification_log.txt`. When finished, the script also writes one `graph_data_<name>.json` per network and saves two figures:
- `verification_multi_kan.png` — 3-panel plot (timing, segments vs. output width, input width vs. output width)
- `true_vanilla_comparison.png` — comparison of Opt, Vanilla-DP, and True-Vanilla methods

### Expected Output (pinnheat, k=1)

```
=== pinnheat ===
  theoretical min error: 0.2411, n_splines: 42, dp_time: 1.006s
  Testing total segment budget: 42 (k=1 per spline for Van)
    Opt k=1 (budget=42): width=2.0603, ilp=0.044s, milp=0.015s
    Van k=1: width=2.0603, time=0.058s
  ...
```

The `width` value is the mean output interval width (lower is tighter/better). `Opt` is our proposed method; `Van` is the vanilla baseline.

---

## Reproducing Paper Results from Logs

To regenerate all paper figures **without re-running experiments**, use the pre-computed logs:

### Option 1: Notebooks (recommended)

```bash
cd project_artifact
jupyter notebook notebooks/results_analysis.ipynb
```

Run all cells. The notebook reads from `logs/graph_data_*.json` and regenerates the figures saved in `plots/`.

### Option 2: Read the Log Directly

`logs/verification_log.txt` contains the complete console output from our experiment run, including all numerical values for every network, segment budget, and input width tested. All numbers in the paper tables can be read directly from this file.

### Mapping Paper Elements to Log/Data Files

| Paper element | Source file | Key fields |
|---|---|---|
| Table: Timing (DP, ILP, MILP per network) | `logs/verification_log.txt` | `dp_time`, `ilp=`, `milp=` |
| Table: Output widths per k | `logs/graph_data_<name>.json` | `graph_data[name].opt_pareto`, `.van_pareto` |
| Fig: Segments vs Output Width | `plots/plot_segments_vs_output_width.pdf` | generated by notebook |
| Fig: Input Width vs Output Width | `plots/plot_input_vs_output_width.pdf` | generated by notebook |
| Fig: Timing Summary | `plots/plot_timing_summary.pdf` | generated by notebook |
| Fig: 3-method comparison | `plots/true_vanilla_comparison.png` | generated by `main.py` |

### JSON Log Format

Each `logs/graph_data_<name>.json` file has this structure:

```json
{
  "filename": "pinnheat",
  "SEGMENT_COUNTS": [1, 5, 10, 20, 30, 40, 50],
  "INPUT_WIDTHS": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
  "graph_data": {
    "pinnheat": {
      "n_params": <int>,
      "opt_pareto": [[time, width], ...],   // Opt method, indexed by SEGMENT_COUNTS
      "van_pareto": [[time, width], ...],   // Vanilla method
      "opt_input_sweep": [[iw, width, time], ...],  // Opt, indexed by INPUT_WIDTHS
      "van_input_sweep": [[iw, width, time], ...],  // Vanilla
      "dp_time": <float>,
      "ilp_time": [<float>, ...],
      "milp_time": [<float>, ...],
      "van_time": [<float>, ...]
    }
  }
}
```

---

## Example: Test run on heatPDE example with k=1

To verify the setup works end-to-end in under 5 minutes, edit `main.py` to use only k=1:

```python
SEGMENT_COUNTS = [1]
INPUT_WIDTHS   = [1.0]
FILENAME_FILTER = ["pinn"]
```

Then run:

```bash
python3 main.py
```

Expected: the script completes without error, prints `width=2.0603` for `Opt k=1`, and writes `verification_multi_kan.png`.

---

## Configuration Reference

### Method Descriptions

The script compares three verification methods:

| Method | Description |
|---|---|
| **Opt (Optimized-by-Budget)** | ILP allocates segments across all splines to minimize output error given a total segment budget. Segments are placed using minimax DP. |
| **Van (Vanilla-DP)** | All splines receive the same number of segments k; segments are placed using minimax DP. |
| **True-Van (Vanilla-Uniform)** | All splines receive k uniformly-spaced breakpoints (no DP optimization). |

For each method and each segment budget k, the script solves a Gurobi MILP to find tight output bounds, then reports mean output interval width.

### Gurobi Parameters

`TIME_LIMIT` and `MIP_GAP` control the Gurobi solver. With `MIP_GAP=0.15`, Gurobi stops when the solution is within 15% of optimal — this is the setting used in the paper. Tightening `MIP_GAP` to 0.05 gives tighter bounds at higher computational cost.

---

## Tested System

| Component | Version |
|---|---|
| OS | macOS 15.0 (Darwin 24.6.0) |
| CPU | Apple M3 Pro (11 cores) |
| RAM | 16 GB |
| Python | 3.11 |
| PyTorch | 2.3.0 |
| Gurobi | 11.0 |
| gurobipy | 11.0.0 |
| numpy | 1.26.4 |
| matplotlib | 3.8.4 |
| joblib | 1.4.2 |
