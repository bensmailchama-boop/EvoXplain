#!/usr/bin/env python3
"""
tissue_composition_check.py

Confounding audit for the two attribution regimes found along the TCGA
regularisation path.

The concern: TCGA adjacent-normal samples are unevenly distributed across
cancer types, and the branch genes are tissue-restricted (keratins ->
squamous epithelium; PCK1/HMGCS2/ACSM -> liver, kidney, gut). If the
regimes simply weight different normal-tissue contrasts, the "fork" would
be a sample-composition artefact rather than model-side divergence.

The test: for each branch gene, compute the tumour-versus-normal effect
size (Cohen's d) WITHIN each tissue source site (TSS). A composition
artefact shows near-zero effects in most tissues and large effects only in
the tissues over-represented among normals. A genuine tumour-versus-normal
marker separates consistently, in one direction, across most tissues.

Reports:
  1. sample-type inventory and TSS composition of the normal class
  2. per-gene, per-tissue effect sizes with a consistency count

Usage:
  python tissue_composition_check.py <tcga_RSEM_gene_tpm.gz> [--min_per_group 8]

Barcode convention: TCGA-<TSS>-<participant>-<sample type>
  sample type 01 = primary tumour, 11 = solid tissue normal
"""

import argparse
import gzip
import sys
from collections import Counter

import numpy as np


# Branch genes. Labels record which regime each gene was associated with.
GENES = {
    'ENSG00000204539': 'CDSN      (B0 kerat)',
    'ENSG00000128422': 'KRT17     (B0 kerat)',
    'ENSG00000205420': 'KRT6A     (B0 kerat)',
    'ENSG00000185479': 'KRT6B     (B0 kerat)',
    'ENSG00000163207': 'IVL       (B0 kerat)',
    'ENSG00000170465': 'KRT6C     (B0 kerat)',
    'ENSG00000124253': 'PCK1      (B1 PPAR)',
    'ENSG00000134240': 'HMGCS2    (B1 PPAR)',
    'ENSG00000163586': 'FABP1     (B1 PPAR)',
    'ENSG00000124205': 'EDN3      (B1 neuro)',
    'ENSG00000006128': 'TAC1      (B1 neuro)',
    'ENSG00000133636': 'NTS       (B1 neuro)',
    'ENSG00000170373': 'CST1      (B1 cystatin)',
    'ENSG00000101441': 'CST4      (B1 cystatin)',
    'ENSG00000162896': 'PIGR?     (B1 cystatin)',
    'ENSG00000066813': 'ACSM2B    (shared)',
    'ENSG00000183747': 'ACSM5     (shared)',
}

TUMOUR_CODE = '01'
NORMAL_CODE = '11'


def parse_barcodes(samples):
    """Return (tss, sample_type) arrays parsed from TCGA barcodes."""
    tss, styp = [], []
    for s in samples:
        p = s.split('-')
        if len(p) > 3:
            tss.append(p[1])
            styp.append(p[3][:2])
        else:
            tss.append('NA')
            styp.append('NA')
    return np.array(tss), np.array(styp)


def cohens_d(a, b):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sd = np.sqrt((a.var() + b.var()) / 2.0)
    return (a.mean() - b.mean()) / sd if sd > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gz_path", help="path to tcga_RSEM_gene_tpm.gz")
    ap.add_argument("--min_per_group", type=int, default=8,
                    help="minimum tumour and normal samples per TSS (default 8)")
    ap.add_argument("--effect_threshold", type=float, default=0.5,
                    help="|d| above which a tissue counts as separating (default 0.5)")
    ap.add_argument("--top_tss", type=int, default=30,
                    help="how many normal-class TSS codes to list (default 30)")
    args = ap.parse_args()

    f = gzip.open(args.gz_path, 'rt')
    header = f.readline().rstrip('\n').split('\t')
    samples = header[1:]
    tss, styp = parse_barcodes(samples)

    # ---- 1. sample inventory ----------------------------------------
    print("total columns: %d" % len(samples))
    print("first 5: %s" % samples[:5])

    print("")
    print("sample type codes:")
    for k, v in Counter(styp).most_common():
        print("  %-6s %d" % (k, v))

    print("")
    print("TSS codes among NORMALS (%s):" % NORMAL_CODE)
    normal_tss = Counter(tss[styp == NORMAL_CODE])
    for k, v in normal_tss.most_common(args.top_tss):
        print("  %-6s %d" % (k, v))

    # ---- 2. per-tissue effect sizes ---------------------------------
    rows = {}
    for line in f:
        gid = line.split('\t', 1)[0]
        base = gid.split('.')[0]
        if base in GENES:
            vals = line.rstrip('\n').split('\t')[1:]
            rows[base] = np.array(
                [float(v) if v not in ('', 'NA') else np.nan for v in vals])
            if len(rows) == len(GENES):
                break
    f.close()

    print("")
    print("genes found: %d/%d" % (len(rows), len(GENES)))
    missing = [g for g in GENES if g not in rows]
    if missing:
        sys.stderr.write("WARN: not found in matrix: %s\n" % ', '.join(missing))

    sites = []
    for t in sorted(set(tss)):
        n_norm = int(((tss == t) & (styp == NORMAL_CODE)).sum())
        n_tum = int(((tss == t) & (styp == TUMOUR_CODE)).sum())
        if n_norm >= args.min_per_group and n_tum >= args.min_per_group:
            sites.append((t, n_tum, n_norm))

    print("sites with >=%d tumour and >=%d normal: %d"
          % (args.min_per_group, args.min_per_group, len(sites)))
    print("  " + ', '.join('%s(T%d/N%d)' % s for s in sites))
    print("")
    print("Effect sizes are Cohen's d, tumour minus normal, computed within")
    print("each tissue source site. up/down count sites with |d| > %.1f."
          % args.effect_threshold)
    print("")

    for base, label in GENES.items():
        if base not in rows:
            continue
        x = rows[base]
        ds = []
        for t, _, _ in sites:
            a = x[(tss == t) & (styp == TUMOUR_CODE)]
            b = x[(tss == t) & (styp == NORMAL_CODE)]
            ds.append((t, cohens_d(a, b)))

        vals = [d for _, d in ds]
        up = sum(1 for v in vals if v > args.effect_threshold)
        down = sum(1 for v in vals if v < -args.effect_threshold)
        ds.sort(key=lambda z: -abs(z[1]))
        top = ', '.join('%s:%+.1f' % z for z in ds[:5])
        print("%-26s up:%2d down:%2d /%d  | strongest: %s"
              % (label, up, down, len(vals), top))


if __name__ == "__main__":
    main()
