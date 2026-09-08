#!/usr/bin/env python3
"""
path_structure_check.py

Tests whether basin membership along the regularisation path is a simple
threshold in C within each split.

Rationale: within a single split every run sees the identical train/test
partition (confirmed by probs_test having one row per run over the same
test samples). If basin membership is determined by C alone, then sample
composition -- tumour type, histology, tumour purity, adjacent-normal
availability -- is held fixed across the basin boundary and cannot be
causing the divergence.

Reports:
  1. how many splits have basin label monotone in C (a single threshold)
  2. the empirical boundary location per split
  3. full run-by-run detail for any split with more than one flip

Usage:
  python path_structure_check.py <aggregates_dir> [--lens lime|shap]

  <aggregates_dir> contains split*/aggregate_split*.npz
"""

import argparse
import glob
import os
import sys

import numpy as np


def load_split(path, lens):
    z = np.load(path, allow_pickle=True)
    lab_key = "cluster_labels_%s" % lens
    if lab_key not in z.files:
        lab_key = "cluster_labels"
    if lab_key not in z.files or "run_C_values" not in z.files:
        return None
    return z["run_C_values"], z[lab_key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aggregates_dir",
                    help="directory containing split*/aggregate_split*.npz")
    ap.add_argument("--lens", default="lime", choices=["lime", "shap"])
    ap.add_argument("--n_examples", type=int, default=5,
                    help="how many per-split boundaries to print (default 5)")
    args = ap.parse_args()

    pattern = os.path.join(args.aggregates_dir, "split*", "aggregate_split*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        pattern = os.path.join(args.aggregates_dir, "aggregate_split*.npz")
        paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit("ERROR: no aggregate_split*.npz found under %s"
                 % args.aggregates_dir)

    total = 0
    non_monotone = []
    examples = []
    boundaries = []

    for p in paths:
        loaded = load_split(p, args.lens)
        if loaded is None:
            sys.stderr.write("WARN: skipping %s (missing keys)\n" % p)
            continue
        C, lab = loaded
        order = np.argsort(C)
        l = lab[order]
        cs = C[order]

        flips = int((np.diff(l) != 0).sum())
        total += 1

        idx = np.where(np.diff(l) != 0)[0]
        bounds = [(float(cs[i]), float(cs[i + 1])) for i in idx]
        for lo, hi in bounds:
            boundaries.append(np.sqrt(lo * hi))   # geometric midpoint

        if flips > 1:
            non_monotone.append((p, flips, cs, l))
        if len(examples) < args.n_examples:
            examples.append((os.path.basename(p), flips, bounds))

    # ---- 1. summary -------------------------------------------------
    print("lens: %s" % args.lens)
    print("splits analysed: %d" % total)
    print("splits where basin is NOT a single threshold in C: %d"
          % len(non_monotone))

    if boundaries:
        b = np.array(boundaries)
        print("boundary C (geometric midpoint of the flanking grid points):")
        print("   min %.4f  median %.4f  max %.4f" % (b.min(), np.median(b), b.max()))

    # ---- 2. example boundaries --------------------------------------
    print("")
    for name, flips, bounds in examples:
        print("%s  flips=%d  boundary_C=%s" % (name, flips, bounds))

    # ---- 3. detail for the exceptions -------------------------------
    if non_monotone:
        print("")
        print("=== splits with more than one flip ===")
        for p, flips, cs, l in non_monotone:
            print("")
            print("%s  flips=%d" % (p, flips))
            for c, b in zip(cs, l):
                print("   C=%.6f  basin=%d" % (c, b))
    else:
        print("")
        print("no splits with more than one flip")


if __name__ == "__main__":
    main()
