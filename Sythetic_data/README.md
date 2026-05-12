# EvoXplain — Synthetic Ground-Truth Validation (LR)

This document describes how to reproduce the synthetic logistic regression
basin-geometry validation experiments. These experiments provide a
**self-contained, data-independent proof** that EvoXplain's bootstrap-and-cluster
procedure correctly recovers k\* under known ground-truth conditions: 2 when
redundant pathways exist (SYNTH-K2), 1 when they do not (SYNTH-K1).

## Why synthetic data?

Real datasets conflate pipeline geometry with dataset-specific confounds.
The synthetic experiments isolate the mechanism: by construction, either
`signal_copy_A` or `signal_copy_B` alone is sufficient to predict `y`
(SYNTH-K2), or both are jointly required (SYNTH-K1). The engine should
recover k\*=2 and k\*=1 respectively under the same pipeline and the same
parsimony threshold. No external data download is required — both datasets
are generated deterministically inside `evoxplain_core_engine.py`.

---

## Dataset definitions

### SYNTH-K2 — ground-truth k\* = 2

| Property | Value |
|---|---|
| CLI name | `synth_k2` |
| Samples | 5,000 |
| Features | 20 (`signal_copy_A`, `signal_copy_B`, `noise_0` … `noise_17`) |
| Latent structure | `x0 = z + ε(0.005)`, `x1 = z + ε(0.005)` — near-identical proxies |
| Label | `y = (z + 0.3·ε > 0)` |
| Expected k\* | **2** — under L1 regularisation the optimiser snaps to either `[1,0,…]` or `[0,1,…]` |

> **Note:** Under pure L2 regularisation the optimiser averages weight across
> `signal_copy_A` and `signal_copy_B` and the multiplicity is hidden. The
> pipeline **must** use L1 (via `--lr_penalty elasticnet --lr_l1_ratio 1.0`)
> to expose the basin geometry. This is a feature, not a bug: it demonstrates
> that regularisation choice is part of the pipeline definition.

---

### SYNTH-K1 — ground-truth k\* = 1

| Property | Value |
|---|---|
| CLI name | `synth_k1` |
| Samples | 5,000 |
| Features | 20 (`signal_required_A`, `signal_required_B`, `noise_0` … `noise_17`) |
| Latent structure | `x0` and `x1` are independent; `y = (x0 + x1 + 0.3·ε > 0)` |
| Expected k\* | **1** — no redundant pathway; every bootstrap converges to the same support |

---

## Prerequisites

```bash
pip install numpy scipy scikit-learn shap
```

No dataset download needed. Both datasets are fully self-contained in the engine.

---

## Reproducing the results — step by step

All commands use split seed 101 and 50 runs, matching the published figure.
Run SYNTH-K2 and SYNTH-K1 independently; output directories are separate.

---

### Step 1 — Run chunk (training + attribution)

**SYNTH-K2**

```bash
python evoxplain_core_engine.py \
  --dataset synth_k2 \
  --model lr \
  --lr_penalty elasticnet \
  --lr_l1_ratio 1.0 \
  --lr_C 0.01 \
  --attribution shap \
  --mode chunk \
  --chunk_id 0 \
  --chunk_size 50 \
  --n_runs 50 \
  --split_seed 101 \
  --output_dir results/synth_k2
```

**SYNTH-K1**

```bash
python evoxplain_core_engine.py \
  --dataset synth_k1 \
  --model lr \
  --lr_penalty elasticnet \
  --lr_l1_ratio 1.0 \
  --lr_C 0.01 \
  --attribution shap \
  --mode chunk \
  --chunk_id 0 \
  --chunk_size 50 \
  --n_runs 50 \
  --split_seed 101 \
  --output_dir results/synth_k1
```

Each command produces `results/synth_k2/split101/chunk_0.npz` (and analogously
for SYNTH-K1) containing the per-run attribution vectors and test accuracies.

---

### Step 2 — Aggregate and cluster

**SYNTH-K2**

```bash
python evoxplain_core_engine.py \
  --dataset synth_k2 \
  --model lr \
  --lr_penalty elasticnet \
  --lr_l1_ratio 1.0 \
  --lr_C 0.01 \
  --attribution shap \
  --mode aggregate_split \
  --split_seed 101 \
  --output_dir results/synth_k2
```

**SYNTH-K1**

```bash
python evoxplain_core_engine.py \
  --dataset synth_k1 \
  --model lr \
  --lr_penalty elasticnet \
  --lr_l1_ratio 1.0 \
  --lr_C 0.01 \
  --attribution shap \
  --mode aggregate_split \
  --split_seed 101 \
  --output_dir results/synth_k1
```

Produces `aggregate_split101.npz` with k\*, cluster labels, and normalised
centroids per lens.

---

### Step 3 — Generate report

**SYNTH-K2**

```bash
python evoxplain_core_engine.py \
  --dataset synth_k2 \
  --model lr \
  --lr_penalty elasticnet \
  --lr_l1_ratio 1.0 \
  --lr_C 0.01 \
  --attribution shap \
  --mode report \
  --split_seed 101 \
  --output_dir results/synth_k2
```

**SYNTH-K1**

```bash
python evoxplain_core_engine.py \
  --dataset synth_k1 \
  --model lr \
  --lr_penalty elasticnet \
  --lr_l1_ratio 1.0 \
  --lr_C 0.01 \
  --attribution shap \
  --mode report \
  --split_seed 101 \
  --output_dir results/synth_k1
```

---

## Expected outputs

### SYNTH-K2

```
[PASS]  expected k* = 2  |  detected k* = 2
lens = shap (local)  |  silhouette {k=2: 0.74, k=3: 0.67, k=4: 0.74}
basin 1 (n=18)  —  signal_copy_A dominant  →  simplex corner [1, 0, …]
basin 2 (n=32)  —  signal_copy_B dominant  →  simplex corner [0, 1, …]
```

The basin scatter plot (|attribution| on `signal_copy_A` vs `signal_copy_B`)
shows two clusters at opposite simplex corners, separated along the L1
simplex edge (dashed). This is the canonical mechanistic non-identifiability
signature: two explanation strategies with identical predictive performance.

### SYNTH-K1

```
[PASS]  expected k* = 1  |  detected k* = 1
lens = shap (local)  |  k* from engine aggregate
basin 1 (n=50)  —  both signals required  →  interior of simplex
```

All 50 runs cluster at a single interior point where both
`signal_required_A` and `signal_required_B` carry comparable attribution
weight. No basin-switching; the mechanism is unique.

---

## Interpreting the figure

The x-axis and y-axis show the L2-normalised absolute SHAP attribution on
the two signal features. The dashed line is the L1 simplex edge (attributions
sum to 1 after normalisation). Under multiplicity (SYNTH-K2), runs cluster at
opposite corners. Under uniqueness (SYNTH-K1), runs cluster at a single
interior point well away from the corners.

This figure constitutes the ground-truth validation of the basin-detection
procedure. It is dataset-agnostic and fully reproducible from a clean
environment with no external dependencies beyond scikit-learn and SHAP.

---

## HPC (SLURM) one-liner

For larger run budgets (e.g. 200 runs across 4 chunks of 50):

```bash
# Submit 4 chunks in parallel, then aggregate
for i in 0 1 2 3; do
  sbatch --partition=core32 --wrap="python evoxplain_core_engine.py \
    --dataset synth_k2 --model lr --lr_penalty elasticnet \
    --lr_l1_ratio 1.0 --lr_C 0.01 --attribution shap \
    --mode chunk --chunk_id $i --chunk_size 50 --n_runs 200 \
    --split_seed 101 --output_dir results/synth_k2"
done
# After all jobs complete:
python evoxplain_core_engine.py \
  --dataset synth_k2 --model lr --lr_penalty elasticnet \
  --lr_l1_ratio 1.0 --lr_C 0.01 --attribution shap \
  --mode aggregate_split --split_seed 101 --output_dir results/synth_k2
```

> **Note for HPC users:** the engine uses Tcsh on headnode1/headnode2.
> Replace the `for` loop above with the Tcsh equivalent:
> ```tcsh
> foreach i (0 1 2 3)
>   sbatch --partition=core32 --wrap="python evoxplain_core_engine.py ..."
> end
> ```

---

## Citation

If you use these experiments, please cite the EvoXplain preprint:

```
Bensmail, C. (2025) | arXiv:2512.22240 | www.evoxplain.com
```

UK provisional patent filed 4 December 2025. Commercial enquiries:
contact@evoxplain.com
