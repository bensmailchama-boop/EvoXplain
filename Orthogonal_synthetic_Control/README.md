# EvoXplain Synthetic Orthogonal Control Experiment

This README describes how to run the SYNTH_ORTHO control experiment to test whether mechanistic multiplicity arises in the absence of feature correlations.

## Overview

The experiment generates synthetic data with orthogonal (uncorrelated) features and tests whether logistic regression models trained on this data exhibit mechanistic multiplicity (k > 1) or collapse to a single explanation basin (k = 1).

**Hypothesis:** With orthogonal features, the null hypothesis k=1 should be accepted, confirming that multiplicity in real datasets stems from collinearity/non-identifiability rather than stochastic training alone.

## Quick Start

Two experimental conditions are provided:

1. **Varied C**: Regularisation strength sampled log-uniformly from [0.01, 100]
2. **Fixed C**: Regularisation strength fixed at C=1.0

---

## Experiment 1: Varied C (C ~ LogUniform[0.01, 100])

### Step 1: Run all 1000 models

```tcsh
python evoxplain_Orthogonal_Synthetic_data_k1_null.py --mode chunk --dataset SYNTH_ORTHO --model logreg --c_mode varied --c_min 0.01 --c_max 100 --n_runs 1000 --chunk_size 1000 --chunk_id 0 --split_seed 101 --output_dir ./results_ortho_varied_C --synth_n_samples 5000 --synth_n_features 10 --synth_snr 50 --seed 42
```

### Step 2: Aggregate and cluster with k=1 null hypothesis testing

```tcsh
python evoxplain_Orthogonal_Synthetic_data_k1_null.py --mode aggregate_split --dataset SYNTH_ORTHO --model logreg --split_seed 101 --output_dir ./results_ortho_varied_C --synth_n_samples 5000 --synth_n_features 10 --synth_snr 50 --seed 42 --k_max 8 --silhouette_threshold 0.25
```

### Output

Results saved to `./results_ortho_varied_C/aggregate_split101.npz`

---

## Experiment 2: Fixed C (C = 1.0)

### Step 1: Run all 1000 models

```tcsh
python evoxplain_Orthogonal_Synthetic_data_k1_null.py --mode chunk --dataset SYNTH_ORTHO --model logreg --c_mode fixed --C 1.0 --n_runs 1000 --chunk_size 1000 --chunk_id 0 --split_seed 101 --output_dir ./results_ortho_fixed_C --synth_n_samples 5000 --synth_n_features 10 --synth_snr 50 --seed 42
```

### Step 2: Aggregate and cluster with k=1 null hypothesis testing

```tcsh
python evoxplain_Orthogonal_Synthetic_data_k1_null.py --mode aggregate_split --dataset SYNTH_ORTHO --model logreg --split_seed 101 --output_dir ./results_ortho_fixed_C --synth_n_samples 5000 --synth_n_features 10 --synth_snr 50 --seed 42 --k_max 8 --silhouette_threshold 0.25
```

### Output

Results saved to `./results_ortho_fixed_C/aggregate_split101.npz`

---

## Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--synth_n_samples` | Number of synthetic samples | 1000 |
| `--synth_n_features` | Number of features (all orthogonal) | 10 |
| `--synth_snr` | Signal-to-noise ratio | 2.0 |
| `--n_runs` | Total number of model runs | 1000 |
| `--c_mode` | `fixed` or `varied` | fixed |
| `--C` | Fixed C value (when c_mode=fixed) | 1.0 |
| `--c_min` | Min C (when c_mode=varied) | 0.01 |
| `--c_max` | Max C (when c_mode=varied) | 100 |
| `--k_max` | Maximum k to consider for clustering | 8 |
| `--silhouette_threshold` | Min silhouette to reject k=1 null | 0.25 |
| `--use_bic` | Use BIC criterion (default: True) | True |
| `--use_gap` | Use Gap statistic (slower) | False |

---

## Inspecting Results

Load the NPZ file in Python:

```python
import numpy as np

# Load results
data = np.load('./results_ortho_varied_C/aggregate_split101.npz', allow_pickle=True)

# Key results
print(f"Best k: {data['best_k'][0]}")
print(f"k=1 accepted (null): {data['k1_accepted'][0]}")
print(f"Normalised entropy: {data['entropy_norm'][0]:.4f}")

# View silhouette scores
for k in range(2, 9):
    key = f'silhouette_k{k}'
    if key in data:
        print(f"Silhouette k={k}: {data[key][0]:.4f}")

# View BIC scores
for k in range(1, 9):
    key = f'bic_k{k}'
    if key in data:
        print(f"BIC k={k}: {data[key][0]:.1f}")

# If k=1 was accepted, see why
if 'k1_reason' in data:
    print(f"k=1 reason: {data['k1_reason'][0]}")
```

---

## Expected Results

For orthogonal synthetic data with high SNR (e.g., SNR=50):

| Condition | Expected k* | Interpretation |
|-----------|-------------|----------------|
| Varied C | 1 | Multiplicity from C variation alone insufficient |
| Fixed C | 1 | No multiplicity without collinearity |

If k > 1 is found, this suggests multiplicity can arise from sources other than feature correlation (e.g., low SNR, underspecification).

---

## HPC Usage (Optional)

For large-scale runs on HPC, submit chunks in parallel:

```tcsh
foreach i (0 1 2 3 4 5 6 7 8 9)
    python evoxplain_Orthogonal_Synthetic_data_k1_null.py --mode chunk --dataset SYNTH_ORTHO --model logreg --c_mode varied --c_min 0.01 --c_max 100 --n_runs 1000 --chunk_size 100 --chunk_id $i --split_seed 101 --output_dir ./results_ortho_varied_C --synth_n_samples 5000 --synth_n_features 10 --synth_snr 50 --seed 42
end
```

Then aggregate after all chunks complete.

---

## Citation

If using this code, please cite the EvoXplain paper (arXiv:2512.22240).
