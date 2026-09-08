# EvoXplain — analysis package

Code and analysis scripts for the EvoXplain preliminary results: explanation
multiplicity in accuracy-equivalent model populations, and what standard
aggregation does to it.

This package is organised by **claim**. Each script below produces a specific
result; the claim it supports is stated with it. If a number appears in the
paper, the script that produced it is here.

---

## Data

Nothing large is stored in this repository.

| Data | Source | Size |
|---|---|---|
| `tcga_RSEM_gene_tpm.gz` | UCSC Xena TOIL RSEM recompute, public download | 707 MB |
| TCGA split aggregates (`aggregate_split*.npz`, 100 splits) | produced by `evoxplain_core_engine.py`; archived | 980 MB |
| Lewis refit pickles (`split_*.pickle`, 20 refits) | produced by `lewis_baselearner_faithful.py`; archived | 656 MB |
| Lewis input matrix (`gene_lewis_exact.pickle`) | GSE62944 `GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz`, columns matched on 16-char TCGA barcode to `lewis_labels_915.tsv` (915 samples, 199 resistant); built by `build_lewis_exact.py` | — |
| Lewis independent reconstruction (`gene_all.pickle`) | Xena/TOIL log2(RSEM-TPM+0.001), 885 samples, 40,950 genes, best-mean ENSG per symbol, HGNC 2026 freeze | — |

Archived artifacts: local at `~/evoxplain/archive/` (tcga_lr_cgrid, lewis_refits), each with MANIFEST.sha256. Public deposit not yet arranged.

The Xena matrix is a public download and is not redistributed here. Record its
checksum when you fetch it; the effect sizes in `tissue_composition_check.py`
depend on that exact release.

---

## Pipeline

**`evoxplain_core_engine.py`** — the analysis pipeline. Fits a population of
models across a declared sweep, computes SHAP and LIME attributions on a
boundary set, clusters the resulting explanation vectors per split, and writes
`aggregate_split<seed>.npz` containing per-run attribution vectors
(`expvec_normed_{lens}`), cluster labels, `run_C_values`, `test_acc` and
`probs_test`.

**`tcga_xena_adapter.py`** — loads the Xena TOIL matrix, derives tumour/normal
labels from TCGA barcode sample-type codes (01 = primary tumour,
11 = solid tissue normal), selects the top-N variable genes, standardises.

Everything downstream reads the `.npz` aggregates. No script other than the
engine and `lewis_baselearner_faithful.py` fits a model.

---

## TCGA results

### 1. The regimes are determined by the pipeline, not by sample composition

```
python path_structure_check.py <aggregates_dir> --lens lime
```

Within a split, all runs share one train/test partition (`probs_test` is
`(n_runs, n_test)` over the same samples). This checks whether cluster
membership is a single threshold in C.

*Result:* 99 of 100 splits monotone; the exception is one run inside the
transition window. Boundary at C ≈ 0.23–0.35 across splits.

*Claim supported:* sample composition — tumour type, histology, purity,
adjacent-normal availability — is held fixed across the regime boundary and
cannot cause the divergence.

### 2. The branch genes are tumour-vs-normal markers, not tissue markers

```
python tissue_composition_check.py <path/to/tcga_RSEM_gene_tpm.gz>
```

Reports the sample-type inventory, the TSS composition of the normal class,
and per-gene Cohen's d computed **within** each tissue source site.

*Result:* 727 normals against 9186 primary tumours, concentrated in breast,
kidney, head-and-neck, thyroid, prostate, liver, colon and lung. Every branch
gene separates consistently across most of 30 tissue sites in one direction
(e.g. HMGCS2 down in 24/30, CST1 up in 22/30 with zero opposing), rather than
only in the tissues over-represented among normals.

*Claim supported:* the branches are not normal-tissue-composition artifacts.

### 3. The two regimes recruit non-overlapping biological programmes

```
python tcga_basin_recurrence_check.py     # paths in the CONFIG block
```

Per-split, per-basin enrichment via g:Profiler, aligned across splits, with
term–gene intersections retained.

*Result (LIME, 100 splits, top-25 terms per split-basin):*

| | Basin 0 | Basin 1 |
|---|---|---|
| extracellular region / space | 100% / 97% | 99% / 50% |
| ACSM2B+ACSM5 CoA-ligase and conjugation terms | 25–36% | 37–45% |
| cornified envelope formation / keratinization | 35–44% / 30% | absent |
| epithelium development | 20% | absent |
| PPAR signalling (PCK1, HMGCS2, FABP1) | absent | 29% |
| neuropeptide receptor binding (EDN3, TAC1, NTS) | absent | 44% |
| secreted cystatins (CST1, CST4) | absent | 45–54% |

*Claim supported:* a shared trunk plus two mutually exclusive branch
programmes across the accuracy-equivalent region.

### 4. Standard aggregation reports neither branch

```
python simulate_practitioner_phantoms.py <aggregates_dir> <out_dir> --lens lime
python enrich_phantoms.py <out_dir> --lens lime
```

Simulates 50 practitioners, each drawing 10 splits and scanning a 10-point
log-spaced C grid, then averaging the (already unit-normalised) explanation
vectors into one reported panel. **No models are refit** — every vector is
indexed out of the existing aggregates.

*Result:* PPAR signalling, present in 29 of 100 individual split-basins,
appears in 0 of 50 aggregated reports. Keratinization likewise absent. The
extracellular trunk and the shared ACSM signal survive in 100% and ~46%.
Seventeen of 50 practitioners report the trunk alone.

*Also tested:* every term in every phantom report traces back to a real basin
term. Averaging subtracts and displaces; it does not fabricate.

### 5. The gradient-boosted control is degenerate — do not cite it

```
python xgb_control_check.py     # run from the XGB results directory
```

*Result:* within each split, all 100 runs have identical accuracy, identical
attribution vectors (pairwise cosine 1.000000) and one unique C value. k*=1
there is arithmetic, not evidence that boosted trees resist multiplicity. A
proper control with varied seeds and subsampling has not been run.

---

## Lewis & Kemp retrospective audit

Faithful reproduction of the classifier in Lewis, J. E. & Kemp, M. L.,
*Nature Communications* 12, 2700 (2021), doi:10.1038/s41467-021-22989-1.
Same model, same tuning budget, 20 independent resamples.

```
python lewis_baselearner_faithful.py      # produces the 20 refits
python lewis_build_genelists.py           # gene lists + agreement statistics
python lewis_enrichment.py                # enrichment, measured-gene background
python lewis_listsize_sweep.py            # list-size sensitivity
```

*Result:* 20 refits, AUROC 0.745–0.883 (mean 0.813). Signed attribution cosine
mean 0.018, median 0.002, 49% of pairs anti-correlated. Top-50 gene lists share
a mean Jaccard of 0.099. The averaged panel's genes appear in a mean of 7.3 of
20 individual models; 15 of 50 appear in 5 or fewer.

At the pathway level, against the measured-gene background: 10 of 20 models
yield **zero** significant terms, top recurrence is 2 of 20, and the averaged
panel yields **zero** significant terms.

*Claim supported:* this is the dispersed regime — no reproducible explanation
at gene or pathway level, in contrast to the structured TCGA case.

### List-size sensitivity (`lewis_listsize_sweep.py`)

| top-N | Jaccard | zero-term models | top recurrence | panel gene appearance |
|---|---|---|---|---|
| 50 | 0.099 | 10/20 | 2/20 | 7.3/20 |
| 100 | 0.104 | 5/20 | 4/20 | 6.7/20 |
| 200 | 0.234 | 2/20 | 15/20 | 5.2/20 |
| 500 | 0.577 | 0/20 | 20/20 | 3.4/20 |

Apparent convergence at large N is carried by broad housekeeping terms
(histone deacetylase activity, RNA Pol II pre-initiation) and coincides with
the averaged panel matching individual models *less* often, not more — the
signature of overlap by volume rather than shared signal. The null at
publication-realistic list sizes is the finding.

---

## Environment

```
pip install -r requirements.txt
```

Enrichment results depend on both the `gprofiler-official` version and the
**g:Profiler database release**. Run against `gprofiler-official` 1.0.0, g:Profiler database release
`e114_eg62_p19_27110d83`. Annotations change between releases; term
recurrence figures are not reproducible without matching both.

g:Profiler settings used throughout: `organism=hsapiens`,
`significance_threshold_method=g_SCS`, `user_threshold=0.05`,
`sources=[GO:BP, GO:MF, GO:CC, KEGG, REAC]`. The Lewis enrichment additionally
passes the 22,835 measured genes as an explicit background with
`domain_scope=custom`.

---

## Known issues in this package

- **Hardcoded paths.** `lewis_baselearner_faithful.py`,
  `lewis_build_genelists.py`, `lewis_listsize_sweep.py` and the CONFIG block of
  `tcga_basin_recurrence_check.py` contain absolute or working-directory-relative
  paths that must be edited before use. The other scripts take arguments.
- **`xgb_control_check.py`** globs relative to the working directory; run it
  from inside the XGB results directory.
- One gene symbol (ENSG00000162896) is unverified and marked with `?` in
  `tissue_composition_check.py`.

See `LIMITATIONS.md` for limitations of the *results*, as distinct from the
code.
