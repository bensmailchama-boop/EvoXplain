# EvoXplain

**Measuring Mechanistic Non-Identifiability in Machine Learning Models**

EvoXplain is a framework for detecting and quantifying *mechanistic multiplicity* — the phenomenon where machine learning models achieve similar predictive performance while using fundamentally different explanation mechanisms. This has critical implications for AI safety, regulatory compliance, and the trustworthiness of model explanations.

> **Key Finding:** Models achieving 97%+ accuracy can exhibit completely unstable feature attributions across training runs. Predictive stability ≠ mechanistic stability.

## Paper

This repository accompanies the preprint:

> **EvoXplain: When Machine Learning Models Agree on Predictions but Disagree on Why — Measuring Mechanistic Multiplicity Across Training Runs**  
> arXiv:2512.22240

## Overview

EvoXplain works by:

1. Training many instances of the same model architecture on the same data (with different random seeds or hyperparameters)
2. Computing SHAP-based feature importance vectors for each model
3. Clustering these explanation vectors to identify distinct "mechanistic basins"
4. Quantifying mechanistic diversity using normalized entropy and silhouette scores

The framework reveals that seemingly equivalent models can rely on entirely different features to make predictions — a finding with implications for explainable AI and regulatory frameworks like the EU AI Act.

## Installation

### Requirements

```bash
pip install numpy pandas scikit-learn shap matplotlib scipy
```

### Clone the Repository

```bash
git clone https://github.com/bensmailchama-boop/EvoXplain.git
cd EvoXplain
```

## Directory Structure

```
EvoXplain/
├── evoxplain_core_engine.py           # Main engine: training, SHAP, aggregation
├── evoxplain_disagreement_within_split.py  # Within-split disagreement analysis
├── evoxplain_visualize_logreg_clustered.py # Visualization for LogReg experiments
├── evoxplain_visualize_rf_clustered.py     # Visualization for Random Forest experiments
├── evoxplain_per_split_summary.py          # Generate per-split summary CSV
├── data/
│   └── compas-scores-two-years.csv    # COMPAS dataset (download separately)
├── results/                           # Output directory for experiments
│   ├── bc_lr_shap_variedC/           # Breast Cancer LogReg varied C results
│   ├── bc_lr_shap_fixedC_C1.0/       # Breast Cancer LogReg fixed C control
│   ├── compas_lr_shap_variedC/       # COMPAS LogReg varied C results
│   └── ...
├── hpc_scripts/                       # SLURM batch scripts (optional)
│   ├── batches_BC_LogReg_variedC.sh
│   ├── aggregate_BC_LogReg_variedC.sh
│   └── ...
└── logs/                              # Log files from HPC runs
```

## Quick Start

### Local Execution (Small Scale)

For testing or small-scale experiments on a local machine:

```bash
# 1. Run a single chunk of experiments (e.g., 20 runs)
python evoxplain_core_engine.py \
    --dataset breast_cancer \
    --model logreg \
    --mode chunk \
    --split_seed 100 \
    --n_runs 100 \
    --chunk_id 0 \
    --chunk_size 20 \
    --c_mode varied \
    --c_min 0.01 \
    --c_max 100 \
    --output_dir results/test_run

# 2. Aggregate chunks into a single split file
python evoxplain_core_engine.py \
    --dataset breast_cancer \
    --model logreg \
    --mode aggregate_split \
    --split_seed 100 \
    --output_dir results/test_run

# 3. Visualize results
python evoxplain_visualize_logreg_clustered.py \
    --input_dir results/test_run \
    --overlay log_C \
    --space normed
```

### HPC Execution (Full Scale)

For reproducing paper results with 10,000 runs per dataset (or 5000 runs for RF):

```bash
# Submit array job for parallel chunk execution
sbatch hpc_scripts/batches_BC_LogReg_variedC.sh

# After all chunks complete, run aggregation
sbatch hpc_scripts/aggregate_BC_LogReg_variedC.sh
```

## Core Components

### evoxplain_core_engine.py

The main engine supporting three modes:

| Mode | Description |
|------|-------------|
| `chunk` | Train models and compute SHAP importance for a subset of runs |
| `aggregate_split` | Combine chunks into a single split NPZ file |
| `aggregate_universal` | Combine multiple splits, cluster, and compute disagreements |

**Key Arguments:**

```
--dataset          Dataset name (breast_cancer, compas, adult)
--model            Model type (logreg, rf)
--mode             Execution mode (chunk, aggregate_split, aggregate_universal)
--split_seed       Random seed for train/test split
--n_runs           Total number of runs per split
--chunk_id         Chunk index for parallel execution
--chunk_size       Number of runs per chunk
--c_mode           Regularization mode: 'fixed' or 'varied'
--C                Fixed C value (when c_mode=fixed)
--c_min, --c_max   C range for log-uniform sampling (when c_mode=varied)
--output_dir       Output directory for results
--disagreement     Compute disagreement report (with aggregate_universal)
```

### evoxplain_disagreement_within_split.py

Computes disagreement analysis **within** each split — the methodologically correct approach since it compares models trained on the same train/test split.

```bash
python evoxplain_disagreement_within_split.py \
    --input_dir results/bc_lr_shap_variedC \
    --dataset breast_cancer \
    --model logreg
```

### Visualization Scripts

Generate publication-quality figures:

```bash
# LogReg experiments
python evoxplain_visualize_logreg_clustered.py \
    --input_dir results/bc_lr_shap_variedC \
    --overlay log_C \
    --space normed \
    --universal \
    --param_grid

# Random Forest experiments  
python evoxplain_visualize_rf_clustered.py \
    --input_dir results/bc_rf_shap_varied \
    --overlay n_estimators \
    --space normed \
    --param_grid
```

## Datasets

### Breast Cancer Wisconsin (Built-in)

Loaded automatically via scikit-learn. Features are standardized.

### COMPAS

Download the COMPAS dataset and place it in `data/`:

```bash
mkdir -p data
wget -O data/compas-scores-two-years.csv \
    https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
```

### Adult Income

Download and place in `data/adult.csv`.

## Output Files

### Per-Chunk (during `chunk` mode)

- `importance_split{seed}_chunk{id}.npy` — SHAP importance vectors
- `meta_split{seed}_chunk{id}.json` — Metadata (accuracy, C values, etc.)

### Per-Split (after `aggregate_split`)

- `aggregate_split{seed}.npz` — Combined importance, accuracy, hyperparameters

### Universal (after `aggregate_universal`)

- `universal_importance.npy` — All importance vectors
- `universal_normed_importance.npy` — L2-normalized vectors
- `universal_labels.npy` — Cluster assignments
- `universal_summary.json` — Clustering metrics (k, entropy, silhouette)
- `disagreement_report_split{seed}.csv` — Instance-level disagreements

## Methodology

### k=1 Null Hypothesis

EvoXplain treats **k=1 as the null hypothesis** when discovering mechanistic basins. This ensures we *discover* whether multiple basins exist rather than *assume* they do.

Forcing k≥2 (as many clustering approaches do by default) would be methodologically incorrect for mechanistic analysis:

- **Convex models** (e.g., Logistic Regression with fixed C) have a unique global optimum — all runs converge to identical explanations
- Forcing k≥2 would artificially split a single basin, inflating entropy estimates

#### Silhouette Threshold

We accept k>1 only if the silhouette score exceeds a threshold (default: 0.25):

| Silhouette | Interpretation | Decision |
|------------|----------------|----------|
| < 0.25 | No substantial cluster structure | Accept k=1 |
| ≥ 0.25 | Evidence of multiple clusters | Accept best k |

The threshold of 0.25 follows standard silhouette interpretation where values below 0.25 indicate no meaningful structure.

#### Algorithm

```
1. If all explanation vectors are identical → return k=1
2. If variance is negligible (< 1e-10) → return k=1  
3. For k = 2 to k_max:
   - Fit K-means, compute silhouette score
   - Track best k by silhouette
4. If best_silhouette < 0.25 → return k=1 (null hypothesis)
5. Otherwise → return best_k
```

#### Validation

The fixed-C control experiment validates this approach: Logistic Regression with C=1.0 correctly returns k=1 for all splits, with `k1_reason: "degenerate_identical_vectors"`.

### Clustering

1. **Normalization:** Center by mean, L2-normalize each explanation vector
2. **K-selection:** Silhouette score optimization over k ∈ [2, k_max], with k=1 null hypothesis
3. **Entropy:** Normalized Shannon entropy over cluster membership

### Key Metrics

| Metric | Interpretation |
|--------|----------------|
| `best_k` | Number of distinct mechanistic basins |
| `entropy_norm` | Uniformity of basin populations (1.0 = perfectly uniform) |
| `silhouette` | Cluster separation quality |
| `disagreement` | Max probability difference between basin representatives |

### Important Constraint

> **Only cluster WITHIN a split.** Clustering across splits is a category error — each split represents an independent causal context.

## Reproducing Paper Results

### Breast Cancer + LogReg (Varied C)

```bash
# 1. Run all chunks (500 array jobs × 20 runs = 10,000 runs)
sbatch hpc_scripts/batches_BC_LogReg_variedC.sh

# 2. Aggregate and cluster
sbatch hpc_scripts/aggregate_BC_LogReg_variedC.sh

# 3. Generate summary
python evoxplain_per_split_summary.py --input_dir results/bc_lr_shap_variedC

# 4. Visualize
python evoxplain_visualize_logreg_clustered.py \
    --input_dir results/bc_lr_shap_variedC \
    --universal --param_grid
```

### Fixed C Control Experiment

To verify that mechanistic multiplicity arises from hyperparameter variation:

```bash
sbatch hpc_scripts/batches_BC_LogReg_FixedC.sh
sbatch hpc_scripts/aggregate_BC_LogReg_fixedC.sh
```

Expected result: All splits show k=1 (explanations collapse to single basin).

## HPC Configuration

The provided SLURM scripts are configured for a specific HPC environment. Before use, modify:

1. **Working directory:** Change `#SBATCH --chdir=` to your path
2. **Partition:** Adjust `#SBATCH --partition=` for your cluster
3. **Python version:** Update `python3.9` if needed

Example modification:

```bash
# In batches_BC_LogReg_variedC.sh, change:
#SBATCH --chdir=/home2/chamabens/HPC_EvoXplain
# To:
#SBATCH --chdir=/your/path/to/EvoXplain
```

## Citation

If you use EvoXplain in your research, please cite:

```bibtex
@article{evoxplain2025,
  title={EvoXplain: When Machine Learning Models Agree on Predictions but Disagree on Why -- Measuring Mechanistic Multiplicity Across Training Runs},
  author={[Chama Bensmail]},
  journal={arXiv preprint arXiv:2512.22240},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Patent Notice

**EvoXplain is the subject of a UK provisional patent application.** The methodology for detecting and quantifying mechanistic non-identifiability in machine learning models is patent-pending. Commercial use may require a separate license agreement. For licensing inquiries, please contact the authors.

## Acknowledgments

This work was conducted by Chama Bensmail, using University of Hertfordshire HPC resources.
