#!/usr/bin/env python3
"""
tcga_xena_adapter.py  –  TCGA / UCSC Xena Expression Matrix Adapter for EvoXplain
====================================================================================
Prepares the TCGA TOIL RSEM TPM matrix (tcga_RSEM_gene_tpm.gz) downloaded from
  https://toil.xenahubs.net/download/tcga_RSEM_gene_tpm.gz
for ingestion by evoxplain_core_engine.py.

This module does NOT modify the EvoXplain core logic.  It is a self-contained
preprocessing adapter that produces arrays compatible with the (X, y, feature_names)
interface expected by run_chunk() / load_dataset() in the core engine.

Returns
-------
X            : np.ndarray, shape (n_samples, n_genes_selected)
feature_names: list[str]   – Ensembl gene IDs (or symbols if remapped)
sample_ids   : list[str]   – TCGA barcodes, one per row of X

Label helpers (optional, separate step)
----------------------------------------
infer_tumour_normal_labels(sample_ids)  → np.ndarray of int (1=tumour, 0=normal, -1=unknown)
merge_external_labels(sample_ids, label_path, sample_col, label_col) → np.ndarray

Usage example
-------------
    from tcga_xena_adapter import load_tcga_expression, infer_tumour_normal_labels

    X, feature_names, sample_ids = load_tcga_expression(
        gz_path   = "/data/tcga_toil/tcga_RSEM_gene_tpm.gz",
        top_n     = 1000,
        standardize = True,
    )
    y = infer_tumour_normal_labels(sample_ids)

    # --- Filter to labelled samples only before passing to EvoXplain ---
    mask = y != -1
    X, y, sample_ids = X[mask], y[mask], [s for s, m in zip(sample_ids, mask) if m]

    # --- Then pass X, y, feature_names straight into EvoXplain core ---
"""

import os
import sys
import gzip
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _print_banner(msg: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {msg}\n{bar}")


def _check_file(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"[tcga_adapter] File not found: {path}\n"
            "  Download with:\n"
            "  wget https://toil.xenahubs.net/download/tcga_RSEM_gene_tpm.gz"
        )
    size_gb = p.stat().st_size / 1e9
    print(f"[tcga_adapter] Found: {p.name}  ({size_gb:.2f} GB on disk)")
    return p


def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column to float64, replacing non-numeric with NaN."""
    n_before = df.shape
    df = df.apply(pd.to_numeric, errors="coerce")
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        print(f"[tcga_adapter] WARNING: {n_nan:,} non-numeric cells coerced to NaN "
              f"(will be imputed with column median).")
        # Impute column-wise medians – avoids dropping samples
        col_medians = df.median(axis=0)
        df = df.fillna(col_medians)
    return df


def _remove_near_zero_variance(df: pd.DataFrame,
                                variance_threshold: float = 1e-6) -> pd.DataFrame:
    """Drop genes whose variance across samples is <= variance_threshold."""
    gene_var = df.var(axis=0)
    mask = gene_var > variance_threshold
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"[tcga_adapter] Dropped {n_dropped:,} near-zero-variance genes "
              f"(var <= {variance_threshold}).")
    return df.loc[:, mask]


def _deduplicate_genes(df: pd.DataFrame,
                        strategy: str = "keep_max_variance") -> pd.DataFrame:
    """
    Handle duplicate gene identifiers in the row index (pre-transpose).

    Parameters
    ----------
    strategy : 'keep_max_variance' (default) or 'keep_first' or 'mean'
    """
    n_before = len(df.columns)
    dup_mask = df.columns.duplicated(keep=False)
    n_dups = dup_mask.sum()

    if n_dups == 0:
        return df

    print(f"[tcga_adapter] Found {n_dups:,} duplicate gene IDs. "
          f"Deduplication strategy: '{strategy}'.")

    if strategy == "keep_first":
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    elif strategy == "mean":
        df = df.T.groupby(level=0).mean().T

    else:  # keep_max_variance (default)
        var_series = df.var(axis=0)
        # For each group of duplicates, keep the one with highest variance
        keep_idx = (
            var_series
            .groupby(df.columns)
            .idxmax()
        )
        df = df[keep_idx.values]

    n_after = len(df.columns)
    print(f"[tcga_adapter] Genes after deduplication: "
          f"{n_before:,} → {n_after:,}")
    return df


def _select_top_variable_genes(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep the top_n most variable genes (by variance across samples)."""
    gene_var = df.var(axis=0).sort_values(ascending=False)
    selected = gene_var.index[:top_n].tolist()
    print(f"[tcga_adapter] Selecting top {top_n:,} most variable genes "
          f"(from {df.shape[1]:,} remaining).")
    return df[selected]


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_tcga_expression(
    gz_path: "data/tcga_RSEM_gene_tpm.gz",
    top_n: int = 1000,
    standardize: bool = True,
    variance_threshold: float = 1e-6,
    log2_transform: bool = False,
    deduplicate_strategy: str = "keep_max_variance",
    chunksize: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load and preprocess the TCGA TOIL RSEM TPM expression matrix.

    The raw matrix has layout:
        rows    = Ensembl gene IDs  (first column = 'sample' or gene ID header)
        columns = TCGA sample barcodes

    This function transposes to the standard ML orientation:
        rows    = samples
        columns = genes (features)

    Parameters
    ----------
    gz_path           : Path to tcga_RSEM_gene_tpm.gz
    top_n             : Number of most variable genes to retain (default 1000)
    standardize       : If True, z-score each gene across samples (default True)
    variance_threshold: Genes with var <= this are removed (default 1e-6)
    log2_transform    : Apply log2(x + 1) to raw TPM values (default True).
                        Set False if the file is already log2-transformed
                        (Xena TOIL files ARE already log2(TPM+0.001), so this
                        applies an additional shift – see note below).
    deduplicate_strategy : How to handle duplicate gene IDs:
                        'keep_max_variance' | 'keep_first' | 'mean'
    chunksize         : If set, read in row chunks to reduce peak RAM.
                        None = load all at once (faster, needs ~12 GB RAM).
    verbose           : Print progress messages (default True)

    Returns
    -------
    X            : np.ndarray  shape (n_samples, top_n)
    feature_names: list[str]   length top_n
    sample_ids   : list[str]   length n_samples

    Notes
    -----
    The Xena TOIL file stores values as log2(TPM + 0.001), so the data is
    already log-scaled.  If log2_transform=True (default), this function
    applies log2(x + 1) on top of the stored values – this is a conservative
    re-scaling that is safe but not standard.  If you know your file is the
    raw TOIL output, set log2_transform=False to avoid double-transformation.
    The pipeline will still filter near-zero-variance genes regardless.
    """

    _check_file(gz_path)

    _print_banner("TCGA Xena Expression Adapter  –  Loading")

    # ------------------------------------------------------------------
    # 1. Load gzipped TSV
    # ------------------------------------------------------------------
    print(f"[tcga_adapter] Reading {gz_path} ...")
    print("[tcga_adapter]   (this may take 1-3 minutes for the full ~10 GB matrix)")

    try:
        if chunksize is not None:
            # Chunked read – lower peak RAM, slower
            chunks = []
            with gzip.open(gz_path, "rt") as fh:
                reader = pd.read_csv(fh, sep="\t", index_col=0, chunksize=chunksize)
                for i, chunk in enumerate(reader):
                    chunks.append(chunk)
                    if verbose:
                        print(f"  chunk {i+1}  ({sum(c.shape[0] for c in chunks):,} genes so far)",
                              end="\r")
            raw = pd.concat(chunks, axis=0)
        else:
            with gzip.open(gz_path, "rt") as fh:
                raw = pd.read_csv(fh, sep="\t", index_col=0)
    except Exception as e:
        raise RuntimeError(
            f"[tcga_adapter] Failed to read expression matrix: {e}\n"
            "  Ensure the file is not truncated and is valid gzipped TSV."
        ) from e

    print(f"\n[tcga_adapter] Raw matrix loaded: {raw.shape[0]:,} genes x "
          f"{raw.shape[1]:,} samples")

    # ------------------------------------------------------------------
    # 2. Sanity checks on raw layout
    # ------------------------------------------------------------------
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("[tcga_adapter] Matrix is empty after loading.")

    # Heuristic: if there are more columns than rows, the matrix may already
    # be transposed (samples x genes) – warn the user
    if raw.shape[1] > raw.shape[0] * 5:
        warnings.warn(
            "[tcga_adapter] The matrix has many more columns than rows. "
            "Expected genes x samples layout.  Check your file source.",
            UserWarning
        )

    # Confirm index looks like gene IDs (Ensembl or symbol)
    sample_gene_ids = raw.index[:5].tolist()
    print(f"[tcga_adapter] First 5 gene IDs (index): {sample_gene_ids}")
    sample_barcodes = raw.columns[:5].tolist()
    print(f"[tcga_adapter] First 5 sample barcodes (columns): {sample_barcodes}")

    # ------------------------------------------------------------------
    # 3. Transpose  →  samples × genes
    # ------------------------------------------------------------------
    print("[tcga_adapter] Transposing to samples × genes ...")
    df = raw.T  # Now: rows = samples, columns = genes
    df.index.name = "sample_id"
    print(f"[tcga_adapter] Shape after transpose: {df.shape}  "
          f"(samples × genes)")

    sample_ids_raw = df.index.tolist()

    # ------------------------------------------------------------------
    # 4. Deduplicate gene columns (can arise from gene symbol mapping)
    # ------------------------------------------------------------------
    df = _deduplicate_genes(df, strategy=deduplicate_strategy)

    # ------------------------------------------------------------------
    # 5. Coerce to numeric safely
    # ------------------------------------------------------------------
    print("[tcga_adapter] Coercing values to numeric ...")
    df = _safe_numeric(df)

    # ------------------------------------------------------------------
    # 6. Log2(x + 1)  –  applied to stored values
    # ------------------------------------------------------------------
    if log2_transform:
        print("[tcga_adapter] Applying log2(x + 1) transform ...")
        # Guard against negative values (shouldn't exist in TPM but check)
        n_neg = (df < 0).sum().sum()
        if n_neg > 0:
            print(f"[tcga_adapter] WARNING: {n_neg:,} negative values found – "
                  f"clipping to 0 before log transform.")
            df = df.clip(lower=0)
        df = np.log2(df + 1)
    else:
        print("[tcga_adapter] Skipping log2 transform (log2_transform=False).")

    print(f"[tcga_adapter] Value range after transform: "
          f"[{df.values.min():.3f}, {df.values.max():.3f}]")

    # ------------------------------------------------------------------
    # 7. Remove near-zero-variance genes
    # ------------------------------------------------------------------
    print(f"[tcga_adapter] Filtering near-zero-variance genes "
          f"(threshold={variance_threshold}) ...")
    n_before_var = df.shape[1]
    df = _remove_near_zero_variance(df, variance_threshold=variance_threshold)
    print(f"[tcga_adapter] Genes after variance filter: "
          f"{n_before_var:,} → {df.shape[1]:,}")

    if df.shape[1] == 0:
        raise ValueError(
            "[tcga_adapter] All genes removed by variance filter.  "
            "Check that log2_transform is set correctly for your file."
        )

    # ------------------------------------------------------------------
    # 8. Select top N most variable genes
    # ------------------------------------------------------------------
    actual_top_n = min(top_n, df.shape[1])
    if actual_top_n < top_n:
        warnings.warn(
            f"[tcga_adapter] Requested top_n={top_n} but only "
            f"{df.shape[1]:,} genes remain after filtering.  "
            f"Using top_n={actual_top_n}.",
            UserWarning
        )
    df = _select_top_variable_genes(df, top_n=actual_top_n)

    # ------------------------------------------------------------------
    # 9. Optionally standardize features (z-score per gene)
    # ------------------------------------------------------------------
    if standardize:
        print("[tcga_adapter] Standardizing features (z-score per gene) ...")
        scaler = StandardScaler()
        X = scaler.fit_transform(df.values)
    else:
        print("[tcga_adapter] Skipping standardization (standardize=False).")
        X = df.values.astype(np.float64)

    # ------------------------------------------------------------------
    # 10. Extract outputs
    # ------------------------------------------------------------------
    feature_names = df.columns.tolist()
    sample_ids    = df.index.tolist()

    # Final shape sanity check
    assert X.shape[0] == len(sample_ids), "Sample count mismatch after processing"
    assert X.shape[1] == len(feature_names), "Feature count mismatch after processing"

    _print_banner("TCGA Adapter  –  Done")
    print(f"  X shape        : {X.shape}  (samples × genes)")
    print(f"  # features     : {len(feature_names):,}")
    print(f"  # samples      : {len(sample_ids):,}")
    print(f"  X dtype        : {X.dtype}")
    print(f"  X range        : [{X.min():.4f}, {X.max():.4f}]")
    print(f"  Any NaN in X?  : {np.isnan(X).any()}")
    print(f"  Any Inf in X?  : {np.isinf(X).any()}")
    print(f"  Sample IDs (5) : {sample_ids[:5]}")
    print(f"  Features  (5)  : {feature_names[:5]}")
    print("=" * 60)

    return X, feature_names, sample_ids


# =============================================================================
# LABEL HELPERS
# =============================================================================

# TCGA barcode sample-type codes:
#   01–09  →  Tumour  (01 = Primary Solid Tumour, most common)
#   10–19  →  Normal  (10 = Blood Derived Normal, 11 = Solid Tissue Normal)
#   20–29  →  Control / other
# The sample type is encoded in the 4th field of the barcode, first two digits.
# Full barcode format:  TCGA-{TSS}-{PARTICIPANT}-{SAMPLE}{VIAL}-{PORTION}{ANALYTE}-{PLATE}-{CENTER}
# Abbreviated (16-char): TCGA-XX-XXXX-YYZ   where YY = sample type code

_TUMOUR_CODES  = set(f"{i:02d}" for i in range(1, 10))   # "01"–"09"
_NORMAL_CODES  = set(f"{i:02d}" for i in range(10, 20))  # "10"–"19"


def _extract_sample_type_code(barcode: str) -> Optional[str]:
    """
    Extract the two-digit sample-type code from a TCGA barcode.
    Returns None if the barcode doesn't match the expected format.
    """
    parts = barcode.split("-")
    if len(parts) >= 4:
        sample_field = parts[3]          # e.g. "01A", "11B", "10A"
        if len(sample_field) >= 2 and sample_field[:2].isdigit():
            return sample_field[:2]
    return None


def infer_tumour_normal_labels(
    sample_ids: List[str],
    tumour_label: int = 1,
    normal_label: int = 0,
    unknown_label: int = -1,
) -> np.ndarray:
    """
    Infer binary tumour/normal labels from TCGA sample barcodes.

    TCGA barcode convention (4th hyphen-delimited field):
      01–09  →  Tumour  (e.g. 01A = Primary Solid Tumour)
      10–19  →  Normal  (e.g. 11A = Solid Tissue Normal)
      Other  →  Unknown (returned as `unknown_label`, default -1)

    Parameters
    ----------
    sample_ids    : List of TCGA barcodes (from load_tcga_expression)
    tumour_label  : Integer label for tumour samples (default 1)
    normal_label  : Integer label for normal samples (default 0)
    unknown_label : Label for unrecognised barcodes (default -1)

    Returns
    -------
    y : np.ndarray of int, shape (n_samples,)
        Filter out unknown_label before passing to EvoXplain:
            mask = y != -1
            X_labelled = X[mask]
            y_labelled = y[mask]
    """
    y = np.full(len(sample_ids), unknown_label, dtype=int)

    n_tumour  = 0
    n_normal  = 0
    n_unknown = 0

    for i, sid in enumerate(sample_ids):
        code = _extract_sample_type_code(sid)
        if code is None:
            n_unknown += 1
        elif code in _TUMOUR_CODES:
            y[i] = tumour_label
            n_tumour += 1
        elif code in _NORMAL_CODES:
            y[i] = normal_label
            n_normal += 1
        else:
            n_unknown += 1

    print(f"[tcga_adapter] Label inference from barcodes:")
    print(f"  Tumour  (label={tumour_label}) : {n_tumour:,}")
    print(f"  Normal  (label={normal_label}) : {n_normal:,}")
    print(f"  Unknown (label={unknown_label}): {n_unknown:,}")

    if n_tumour == 0 and n_normal == 0:
        warnings.warn(
            "[tcga_adapter] No tumour or normal labels inferred.  "
            "Check that sample_ids are valid TCGA barcodes "
            "(expected format: TCGA-XX-XXXX-YYZ...).",
            UserWarning
        )

    return y


def merge_external_labels(
    sample_ids: List[str],
    label_path: str,
    sample_col: str,
    label_col: str,
    label_map: Optional[dict] = None,
    missing_label: int = -1,
    sep: str = "\t",
) -> np.ndarray:
    """
    Merge labels from an external clinical/phenotype file by sample ID.

    Useful when you have a Xena phenotype file or custom annotation:
        https://toil.xenahubs.net/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz

    Parameters
    ----------
    sample_ids   : List of TCGA barcodes (from load_tcga_expression)
    label_path   : Path to the label file (.tsv, .csv, or .tsv.gz / .csv.gz)
    sample_col   : Column name in label_path that contains sample IDs
    label_col    : Column name in label_path that contains the target label
    label_map    : Optional dict to map raw label values to integers.
                   e.g. {"Tumor": 1, "Normal": 0}
                   If None, the raw values are returned as-is (must be numeric).
    missing_label: Label assigned to samples not found in label_path (default -1)
    sep          : Delimiter for the label file ('\\t' for TSV, ',' for CSV)

    Returns
    -------
    y : np.ndarray of int, shape (n_samples,)
        Filter out missing_label before passing to EvoXplain.

    Example
    -------
        y = merge_external_labels(
            sample_ids = sample_ids,
            label_path = "/data/tcga_toil/TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
            sample_col = "sample",
            label_col  = "_primary_disease",
            label_map  = {"breast invasive carcinoma": 1,
                          "kidney clear cell carcinoma": 0},
        )
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"[tcga_adapter] Label file not found: {label_path}"
        )

    print(f"[tcga_adapter] Loading external labels from: {label_path}")

    # Support both gzipped and plain files
    compression = "gzip" if label_path.endswith(".gz") else None
    label_df = pd.read_csv(
        label_path,
        sep=sep,
        compression=compression,
        usecols=[sample_col, label_col],
        dtype=str
    )

    # Strip whitespace from sample IDs and label values
    label_df[sample_col] = label_df[sample_col].str.strip()
    label_df[label_col]  = label_df[label_col].str.strip()

    # Check for duplicates in the label file
    n_dup = label_df[sample_col].duplicated().sum()
    if n_dup > 0:
        warnings.warn(
            f"[tcga_adapter] {n_dup:,} duplicate sample IDs in label file – "
            "keeping first occurrence.",
            UserWarning
        )
        label_df = label_df.drop_duplicates(subset=[sample_col], keep="first")

    label_lookup = dict(zip(label_df[sample_col], label_df[label_col]))

    print(f"[tcga_adapter] Label file: {len(label_lookup):,} unique samples, "
          f"label column='{label_col}'")
    unique_raw = label_df[label_col].unique()
    print(f"[tcga_adapter] Unique raw label values: {unique_raw[:10].tolist()} "
          f"{'...' if len(unique_raw) > 10 else ''}")

    # Build label array
    y_raw = [label_lookup.get(sid, None) for sid in sample_ids]

    if label_map is not None:
        y = np.array(
            [label_map.get(v, missing_label) if v is not None else missing_label
             for v in y_raw],
            dtype=int
        )
    else:
        # Try direct numeric conversion
        converted = []
        for v in y_raw:
            if v is None:
                converted.append(missing_label)
            else:
                try:
                    converted.append(int(float(v)))
                except (ValueError, TypeError):
                    raise ValueError(
                        f"[tcga_adapter] Cannot convert label '{v}' to int.  "
                        "Provide a label_map dict to map string labels to integers."
                    )
        y = np.array(converted, dtype=int)

    n_found   = np.sum(y != missing_label)
    n_missing = np.sum(y == missing_label)

    print(f"[tcga_adapter] Merge results:")
    print(f"  Matched samples  : {n_found:,} / {len(sample_ids):,}")
    print(f"  Missing (label={missing_label}): {n_missing:,}")

    if label_map is not None:
        for raw_val, mapped_val in label_map.items():
            count = np.sum(y == mapped_val)
            print(f"  '{raw_val}' → {mapped_val}  : {count:,} samples")

    return y


# =============================================================================
# CONVENIENCE WRAPPER: returns EvoXplain-compatible (X, y, feature_names)
# =============================================================================

def load_tcga_for_evoxplain(
    gz_path: str,
    label_source: str = "barcode",
    label_path: Optional[str] = None,
    sample_col: str = "sample",
    label_col: str = "_sample_type",
    label_map: Optional[dict] = None,
    top_n: int = 1000,
    standardize: bool = True,
    variance_threshold: float = 1e-6,
    log2_transform: bool = False,
    drop_unknown: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    One-shot loader that returns (X, y, feature_names) matching the
    EvoXplain core engine's load_dataset() output signature exactly.

    Parameters
    ----------
    gz_path       : Path to tcga_RSEM_gene_tpm.gz
    label_source  : 'barcode' → infer tumour/normal from TCGA barcodes
                    'file'    → merge from an external label file (requires label_path)
    label_path    : Path to external label TSV/CSV (required if label_source='file')
    sample_col    : Sample ID column in label_path
    label_col     : Label column in label_path
    label_map     : Mapping from raw string labels to int (for label_source='file')
    top_n         : Top N most variable genes to keep
    standardize   : Z-score standardise features
    variance_threshold : Near-zero variance removal threshold
    log2_transform: Apply log2(x+1) transform
    drop_unknown  : Remove samples with label == -1 (default True)

    Returns
    -------
    X            : np.ndarray  (n_samples, top_n)
    y            : np.ndarray  (n_samples,)  int labels
    feature_names: list[str]   length top_n

    Example
    -------
        X, y, feature_names = load_tcga_for_evoxplain(
            gz_path      = "/data/tcga_toil/tcga_RSEM_gene_tpm.gz",
            label_source = "barcode",
            top_n        = 1000,
        )
        # Plug directly into EvoXplain core's run_chunk() logic
    """
    # Step 1: Load expression matrix
    X, feature_names, sample_ids = load_tcga_expression(
        gz_path            = gz_path,
        top_n              = top_n,
        standardize        = standardize,
        variance_threshold = variance_threshold,
        log2_transform     = log2_transform,
    )

    # Step 2: Get labels
    if label_source == "barcode":
        y = infer_tumour_normal_labels(sample_ids)

    elif label_source == "file":
        if label_path is None:
            raise ValueError(
                "[tcga_adapter] label_source='file' requires label_path to be set."
            )
        y = merge_external_labels(
            sample_ids  = sample_ids,
            label_path  = label_path,
            sample_col  = sample_col,
            label_col   = label_col,
            label_map   = label_map,
        )
    else:
        raise ValueError(
            f"[tcga_adapter] Unknown label_source='{label_source}'.  "
            "Choose 'barcode' or 'file'."
        )

    # Step 3: Drop unlabelled samples
    if drop_unknown:
        mask = y != -1
        n_dropped = (~mask).sum()
        if n_dropped > 0:
            print(f"[tcga_adapter] Dropping {n_dropped:,} unlabelled samples "
                  f"(y == -1). {mask.sum():,} samples retained.")
        X          = X[mask]
        y          = y[mask]
        sample_ids = [s for s, m in zip(sample_ids, mask) if m]

    # Final sanity check
    _print_banner("EvoXplain-ready output")
    unique, counts = np.unique(y, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"  Label {lbl}: {cnt:,} samples")
    print(f"  X shape       : {X.shape}")
    print(f"  feature_names : {len(feature_names):,}  (first: {feature_names[0]})")
    print("=" * 60)

    return X, y, feature_names


# =============================================================================
# CLI  (quick sanity-check run)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TCGA Xena Adapter – preprocess expression matrix for EvoXplain"
    )
    parser.add_argument("--gz_path", required=True,
                        help="Path to tcga_RSEM_gene_tpm.gz")
    parser.add_argument("--top_n", type=int, default=1000,
                        help="Top N most variable genes (default 1000)")
    parser.add_argument("--standardize", type=int, default=1,
                        help="Z-score standardize (1=yes, 0=no)")
    parser.add_argument("--log2_transform", type=int, default=0,
                        help="Apply log2(x+1) (1=yes, 0=no)")
    parser.add_argument("--label_source", default="barcode",
                        choices=["barcode", "file"],
                        help="How to obtain labels")
    parser.add_argument("--label_path", default=None,
                        help="External label TSV/CSV (required if label_source=file)")
    parser.add_argument("--sample_col", default="sample")
    parser.add_argument("--label_col", default="_sample_type")
    parser.add_argument("--output_npz", default=None,
                        help="Optional: save X, y, feature_names to .npz")
    args = parser.parse_args()

    X, y, feature_names = load_tcga_for_evoxplain(
        gz_path        = args.gz_path,
        label_source   = args.label_source,
        label_path     = args.label_path,
        sample_col     = args.sample_col,
        label_col      = args.label_col,
        top_n          = args.top_n,
        standardize    = bool(args.standardize),
        log2_transform = bool(args.log2_transform),
    )

    if args.output_npz:
        np.savez(
            args.output_npz,
            X             = X,
            y             = y,
            feature_names = np.array(feature_names),
        )
        print(f"[tcga_adapter] Saved preprocessed data to: {args.output_npz}")
    else:
        print("[tcga_adapter] Done. Pass X, y, feature_names to EvoXplain.")
