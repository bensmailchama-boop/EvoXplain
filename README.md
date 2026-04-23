# EvoXplain

**Measuring Mechanistic Non-Identifiability in Machine Learning Models**

EvoXplain is a framework for detecting and quantifying *mechanistic multiplicity* — the phenomenon where machine learning models achieve similar predictive performance while relying on fundamentally different explanation mechanisms. This has direct implications for AI safety, regulatory compliance (e.g., the EU AI Act), and the trustworthiness of post-hoc model explanations in high-stakes domains such as genomics, healthcare, and criminal justice.

> **Key finding.** Models achieving ≥98% test accuracy can exhibit completely unstable feature attributions across training runs. Predictive stability is *not* mechanistic stability. On TCGA pan-cancer RNA-seq, we observe a **~39× decoupling ratio** between accuracy variation and attribution-cosine variation within a single pipeline.

---

## Paper

This repository accompanies the preprint:

> **EvoXplain: When Machine Learning Models Agree on Predictions but Disagree on Why — Measuring Mechanistic Multiplicity Across Training Runs**
> arXiv:2512.22240 (December 2025)

---

## What's new (2026)

- **TCGA pan-cancer benchmark.** End-to-end pipeline for 11,060 samples × 1,000 top-variance genes (tumour-vs-normal) with a Xena-format adapter.
- **TCGA-BRCA subtype benchmark.** Luminal-vs-Basal loader for intrinsic molecular subtype analysis.
- **Multi-lens attributions.** SHAP, LIME, Integrated Gradients, and Gini can now be computed in a single pass by passing a comma-separated `--attribution` list.
- **Cross-pipeline coverage.** Logistic Regression (fixed-C, C-grid, ElasticNet grid), Random Forest, XGBoost, DNN, and SVM are all first-class citizens.
- **Boundary-set methodology.** Attributions are computed on a decision-boundary subset of the held-out test set (default: probability band 0.45–0.55, k=200) to sharpen multiplicity signals.
- **Cross-split stability mode.** Hungarian-matched centroid cosine similarity across splits, per-lens, for detecting stable vs data-driven basins.
- **9-block falsification battery.** A consolidated adversarial test suite that rules out stochastic noise, sampling variance, background-set artefacts, and model instability as alternative explanations for observed multiplicity.

---

## Overview

EvoXplain operates in five conceptual steps:

1. **Train a large ensemble** of the same model class on the same train/test split, varying a hyperparameter or random seed axis (a *pipeline*).
2. **Compute attribution vectors** on a boundary subset of the test set using one or more *lenses* (SHAP, LIME, IG, Gini).
3. **Cluster the L2-normalised attribution vectors** *within a split* to identify distinct mechanistic basins.
4. **Quantify mechanistic diversity** using silhouette score, normalised Shannon entropy, and pairwise centroid cosine similarity.
5. **Falsify** the finding against a battery of null hypotheses (randomness, sampling noise, background-set artefacts, model instability) before claiming genuine multiplicity.

### Pipelines vs. lenses — an important distinction

The *pipeline* is the combination of **(data split) + (model class) + (hyperparameters)**. Mechanistic multiplicity is a property of the pipeline. Attribution methods are **post-hoc lenses**: they do not cause, suppress, or generate multiplicity — they only *resolve* (or fail to resolve) multiplicity that already exists in the pipeline. Different lenses may reveal different facets of the same underlying non-identifiability.

---

## Installation

### Core requirements

```bash
pip install numpy pandas scikit-learn scipy matplotlib
pip install shap lime      # attribution lenses
pip install torch          # required for DNN and IG
pip install xgboost        # required for XGB pipeline
```

### Clone the repository

```bash
git clone https://github.com/bensmailchama-boop/EvoXplain.git
cd EvoXplain
```

---

## Supported pipelines

| Component | Options |
|-----------|---------|
| **Models** (`--model`) | `lr`, `rf`, `xgb`, `dnn`, `svm` |
| **LR penalty** (`--lr_penalty`) | `l2`, `l1`, `elasticnet`, `none` |
| **LR C mode** (`--lr_C_mode`) | `fixed`, `grid` (log-uniform over `--lr_C_min`…`--lr_C_max`) |
| **LR ElasticNet l1_ratio** (`--lr_l1_ratio_mode`) | `fixed`, `grid` (0→1 across runs) |
| **Attribution lenses** (`--attribution`) | `shap`, `lime`, `ig`, `gini` — comma-separated for multi-lens |
| **Boundary set** (`--boundary_method`) | `prob_band` (default, 0.45–0.55), `full` |

## Supported datasets

| Name (`--dataset`) | Task | Notes |
|--------------------|------|-------|
| `breast_cancer` | Benign vs malignant | Sklearn built-in |
| `compas` | Recidivism prediction | Requires `data/compas-scores-two-years.csv` |
| `adult` | Income >50K | UCI Adult |
| `german_credit` | Credit risk | OpenML credit-g |
| `acs_income` | ACS income (US) | via `folktables`, configurable by state/year |
| `cmnist` | Colored MNIST | Spurious-correlation benchmark |
| `mimic` | MIMIC-CXR pneumonia | Requires preprocessed `.npz` |
| `synthetic_single` | Controlled single-mechanism | Ground-truth k=1 |
| `synthetic_two` | Controlled two-mechanism | Ground-truth k=2 |
| `synthetic_two_mixture` | Two-mechanism mixture | Ground-truth k=2 with overlap |
| `tcga` | TCGA tumour vs normal | 11,060 samples × 1,000 genes (Xena adapter) |
| `tcga_brca_luminal_vs_basal` | BRCA intrinsic subtypes | Luminal A/B vs Basal |

---

## Quick start

### Local execution — single chunk

```bash
python evoxplain_core_engine.py \
    --dataset breast_cancer \
    --model lr \
    --mode chunk \
    --split_seed 100 \
    --n_runs 100 \
    --chunk_id 0 \
    --chunk_size 20 \
    --attribution shap,lime \
    --lr_C_mode grid --lr_C_min 0.01 --lr_C_max 100 \
    --output_dir results/test_run
```

### Aggregate a split (per-lens clustering)

```bash
python evoxplain_core_engine.py \
    --dataset breast_cancer \
    --model lr \
    --mode aggregate_split \
    --split_seed 100 \
    --attribution shap,lime \
    --output_dir results/test_run
```

### Cross-split stability (Hungarian-matched)

```bash
python evoxplain_core_engine.py \
    --dataset breast_cancer \
    --model lr \
    --mode cross_split_stability \
    --split_seeds 100,101,102,103,104 \
    --attribution_lens shap \
    --output_dir results/test_run
```

### HPC execution

SLURM array job templates for the full 100×100 and 10,000-run designs are provided in `hpc_scripts/`. Update `#SBATCH --chdir=` and `#SBATCH --partition=` for your cluster before use. Note: the example scripts assume a **tcsh** shell — adjust for bash if your environment differs.

---

## Core components

### `evoxplain_core_engine.py`

Single-file engine covering the full experiment lifecycle.

| Mode (`--mode`) | Purpose |
|-----------------|---------|
| `chunk` | Train models and compute attributions for a subset of runs within a split |
| `aggregate_split` | Combine chunks into a per-split `aggregate_split{seed}.npz` and cluster per-lens |
| `report` | Emit a per-split summary report (k\*, entropy, silhouette, centroid cosine) |
| `analyze_subbasins` | Sub-structure analysis within a target basin |
| `cross_split_stability` | Match centroids across splits via the Hungarian algorithm; per-lens |
| `load_only` | Sanity-check dataset loading (no training) |

**Key arguments:**

```
--dataset               Dataset name (see table above)
--model                 Model class (lr, rf, xgb, dnn, svm)
--attribution           Comma-separated list of lenses (shap, lime, ig, gini)
--attribution_lens      Lens to use for cross_split_stability when multi-lens
--split_seed            Random seed for a single train/test split
--split_seeds           Comma-separated seeds (cross_split_stability)
--n_runs / --chunk_id / --chunk_size   Parallel chunking controls
--lr_C / --lr_C_mode    Fixed or log-uniform grid over C
--lr_penalty            l1 | l2 | elasticnet | none
--lr_l1_ratio_mode      fixed | grid (for ElasticNet)
--boundary_method       prob_band (default) | full
--boundary_prob_low / --boundary_prob_high / --boundary_k
--shap_bg_mode          per_split | fixed
--cosine_collapse       Collapse-to-k=1 threshold on 1−cos_min (default 0.99)
--tcga_gz_path          Path to TCGA Xena-format expression matrix
--tcga_top_n            Number of top-variance genes (default 1000)
--tcga_subtype_path     Path to BRCA subtype annotations
```

---

## Methodology

### k=1 as the null hypothesis

EvoXplain treats **k=1 as the null hypothesis** when discovering mechanistic basins. We *discover* whether multiple basins exist rather than *assume* they do.

Forcing k≥2 (as many clustering approaches do by default) would be a category error here:

- **Convex pipelines** (e.g., LR with fixed C) have a unique global optimum and must collapse to k=1.
- Forcing k≥2 would artificially split a single basin within a same-data-split experiment and inflate entropy.

### Silhouette threshold

We accept k>1 only if the silhouette score exceeds a threshold (default: 0.25).

| Silhouette | Interpretation | Decision |
|------------|----------------|----------|
| < 0.25 | No substantial cluster structure | Accept k=1 |
| ≥ 0.25 | Evidence of multiple basins | Accept best k |

### Collapse guards

Two additional guards prevent spurious k>1:

1. **Cosine collapse guard.** If `1 − cosine_min < cosine_collapse` (default 0.99), return k=1.
2. **Centroid cosine check.** If any pair of candidate centroids has cosine similarity > 0.99, collapse to k=1.

### Clustering algorithm

```
1. If all attribution vectors are identical → return k=1
2. If variance is negligible (< 1e-10) → return k=1
3. For k = 2 to k_max:
     Fit K-means; compute silhouette (euclidean or cosine)
     Track best k by silhouette
4. If best_silhouette < 0.25 → return k=1
5. If any centroid pair has cosine > 0.99 → return k=1
6. Otherwise → return best_k
```

### Boundary-set methodology

Attributions are computed on a subset of the test set that sits near the decision boundary (default: probability band 0.45–0.55, up to k=200 points). This sharpens the multiplicity signal because bulk regions of feature space tend to be assigned the same label by all basins — the disagreement lives at the margin. The boundary set is computed from `X_test` only and cached per split for reproducibility.

### Within-split vs cross-split

> **Cluster ONLY within a split** (same causal context). Clustering across splits is a category error — each split is an independent causal context. Cross-split *comparison* (via `cross_split_stability` mode) uses Hungarian-matched centroid cosine similarity, not joint clustering.

### Metrics

| Metric | Interpretation |
|--------|----------------|
| `best_k` | Number of mechanistic basins (per lens) |
| `entropy_norm` | Uniformity of basin populations (1.0 = perfectly uniform) |
| `silhouette` | Cluster separation quality |
| `centroid_cosine` | Pairwise similarity between basin representatives |
| `pairwise_avg_matched_cosine` | Cross-split basin stability (Hungarian-matched) |

---

## Headline empirical results

### Breast Cancer Wisconsin (LR, SHAP + LIME, C-grid)

- 100 splits × 100 runs; C log-uniform over [0.01, 100].
- SHAP k\*=2 in 84/100 splits; LIME k\*=2 in 98/100 splits; label agreement between the two lenses 97.3%.
- Multiplicity is intrinsic to the (data-split + LR + C-grid) pipeline — both lenses resolve it.

### TCGA pan-cancer tumour vs normal (cross-pipeline)

Four model classes, 100 splits × 100 runs, dual-lens SHAP + LIME.

| Pipeline | Test accuracy | Multiplicity signature |
|----------|---------------|------------------------|
| XGBoost | 0.9878 | k\*=1 (single tissue-of-origin story; SHAP cosine ~0.976) |
| DNN | 0.9873 | k\*=2 SHAP (sign-flip), k\*=2 LIME (weight-fork) |
| LR C-grid | 0.9832 | **k\*=3 SHAP / k\*=2 LIME**, anti-aligned basins |
| ElasticNet grid | 0.9763 | k\*=2 LIME (clean); SHAP fragments under C-grid over-resolution |

For LR C-grid, basin assignment is monotonically driven by the regularisation strength. Pathway enrichment (g:Profiler) of the top genes in each basin confirms biologically distinct mechanisms:

- **Basin 0** (high C, ~100% split recurrence) — **extracellular region / matrix** terms dominant.
- **Basin 1** (low C, 35–40% recurrence) — **PPAR signalling / lipid metabolism** dominant.

The decoupling ratio between test-accuracy variation and attribution-cosine variation within the LR C-grid pipeline is **~41×**.

### TCGA-BRCA Luminal vs Basal

100 splits × 100 runs, LR C-grid, SHAP + LIME:

- k\*=2 in 94/95 usable splits.
- Inter-basin cosine **+0.85 SHAP / +0.81 LIME** — this is *geometric non-semantic multiplicity* (a sparsity fork, not a sign-flip).
- AGR3, TFF1, DSCAM-AS1, SRARP recur as luminal markers across basins.

---

## Falsification battery

A genuine finding of mechanistic multiplicity must survive adversarial falsification. EvoXplain ships a 9-block battery (consolidated from an earlier 12-test grid) that tests whether the observed k>1 could be explained away by any of the following null accounts:

| Block | Null hypothesis under test |
|-------|----------------------------|
| 1 — Replay / seed / C decomposition | Multiplicity is due to stochastic seed variation, not to the hyperparameter axis |
| 2 — Sampling noise | Multiplicity is due to small-sample boundary-set variance |
| 3 — Background-set artefact | Multiplicity is an artefact of the SHAP background set |
| 4 — Model instability | Multiplicity reflects unstable training, not genuine mechanism choice |
| 5 — LOO-AUC sensitivity | Top-basin features are noise (should fail to transfer) |
| 6 — Cross-split cosine | Basins are data-split-specific rather than structural |
| 7 — Top-k patient stability | Instance-level attributions are noise-driven |
| 8 — Anti-alignment check | Worst-pair cosine reveals true sign-reversal structure |
| 9 — Null / permuted labels | Multiplicity persists on meaningless labels (a failure of the pipeline, not a finding) |

All nine blocks were killed on the Breast Cancer benchmark (full numbers in the preprint). A hardened rerun on TCGA pan-cancer is currently underway.

---

## Output files

### Per-chunk (`chunk` mode)

- `chunk_{id}.npz` — per-lens attribution vectors (`expvec_{lens}`), test accuracies, run IDs, hyperparameter trajectories (`run_C_values`, `run_l1_values`).

### Per-split (`aggregate_split` mode)

- `aggregate_split{seed}.npz` — combined attribution vectors, per-lens cluster labels, centroids (`centroids_normed_{lens}`), k\* (`k_star_{lens}`), silhouette, entropy.

### Cross-split (`cross_split_stability` mode)

- `cross_split_stability.json` — per-lens Hungarian-matched centroid cosine similarity across splits.

---

## Reproducing paper results

### Breast Cancer + LR (C-grid, dual-lens)

```bash
# Run chunks in parallel across the SLURM array
sbatch hpc_scripts/batches_BC_LR_Cgrid.sh

# Aggregate per split
sbatch hpc_scripts/aggregate_BC_LR_Cgrid.sh
```

### TCGA pan-cancer + LR (C-grid, dual-lens)

```bash
sbatch hpc_scripts/batches_TCGA_LR_Cgrid.sh
sbatch hpc_scripts/aggregate_TCGA_LR_Cgrid.sh
```

### Fixed-C control (should return k=1 everywhere)

```bash
sbatch hpc_scripts/batches_BC_LR_FixedC.sh
```

Expected result: all splits return k=1 (explanations collapse to a single basin), validating the k=1 null hypothesis.

---

## Citation

If you use EvoXplain in your research, please cite:

```bibtex
@article{bensmail2025evoxplain,
  title   = {EvoXplain: When Machine Learning Models Agree on Predictions but
             Disagree on Why -- Measuring Mechanistic Multiplicity Across Training Runs},
  author  = {Bensmail, Chama},
  journal = {arXiv preprint arXiv:2512.22240},
  year    = {2025}
}
```

---

## License

EvoXplain is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — free for research, academic, teaching, and open-source use. See `LICENSE` for the full text.

## Patent notice

EvoXplain is the subject of a **UK provisional patent application** covering the methodology for detecting and quantifying mechanistic non-identifiability in machine learning models.

For commercial use that is incompatible with AGPL-3.0, or for patent licensing enquiries, please contact the author: **bensmail.chama@gmail.com**.

## Acknowledgments

This work was conducted by Chama Bensmail (Omics Data Solutions Ltd, Sheffield) using University of Hertfordshire HPC resources, with support from Prof. Volker Steuber and Dr Epaminondas Kapetanios (Bio-computation group and H-XAI Lab, University of Hertfordshire).
