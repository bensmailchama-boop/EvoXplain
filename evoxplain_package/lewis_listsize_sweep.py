#!/usr/bin/env python3
"""
lewis_listsize_sweep.py

Sensitivity analysis for the Lewis & Kemp audit: how do gene-list agreement
and pathway enrichment depend on how many genes get reported?

Motivation: at publication-realistic list sizes (50-100 genes) the refits
share almost nothing and most yield no significant enrichment. Apparent
convergence emerges only at several hundred genes. The diagnostic that
distinguishes real convergence from overlap-by-volume is the averaged panel's
appearance rate in individual models: if models were genuinely agreeing, that
rate would rise with list size alongside the Jaccard. If it FALLS while
Jaccard rises, the convergence is an artefact of the lists getting long enough
to overlap by chance and the terms getting broad enough to be unavoidable.

Enrichment uses the measured genes as an explicit background.

Usage:
  python lewis_listsize_sweep.py <refits_dir> [--sizes 50 100 200 500]

  <refits_dir> contains split_*.pickle
"""

import argparse
import glob
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np

try:
    from gprofiler import GProfiler
except ImportError:
    sys.exit("ERROR: gprofiler-official not installed.\n"
             "  pip install gprofiler-official --break-system-packages")


SOURCES = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refits_dir", help="directory containing split_*.pickle")
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500],
                    help="list sizes to sweep (default 50 100 200 500)")
    ap.add_argument("--organism", default="hsapiens")
    ap.add_argument("--user_threshold", type=float, default=0.05)
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.refits_dir, "split_*.pickle")),
                   key=lambda p: int(re.search(r"split_(\d+)", p).group(1)))
    if not paths:
        sys.exit("ERROR: no split_*.pickle found in %s" % args.refits_dir)

    absval, names = [], None
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        absval.append(np.abs(d["shap_values"]).mean(axis=0))
        if names is None:
            names = np.array([f_.split("#")[-1].strip()
                              for f_ in d["feature_names"]])
    absval = np.vstack(absval)
    bg = list(names)
    n_models = len(paths)
    print("models %d | background %d" % (n_models, len(bg)))

    gp = GProfiler(return_dataframe=False)

    def enrich(genes):
        r = gp.profile(organism=args.organism, query=list(genes),
                       sources=args.sources, user_threshold=args.user_threshold,
                       no_evidences=True, background=bg, domain_scope="custom",
                       significance_threshold_method="g_SCS")
        return [t for t in r if t.get("significant", False)]

    iu = np.triu_indices(n_models, 1)
    rows = []

    for TOP in args.sizes:
        tops = [set(names[np.argsort(absval[i])[::-1][:TOP]])
                for i in range(n_models)]
        jac = [len(tops[i] & tops[j]) / len(tops[i] | tops[j])
               for i, j in zip(*iu)]

        counts, terms, tnames = [], Counter(), {}
        for t_set in tops:
            sig = enrich(t_set)
            counts.append(len(sig))
            for t in sig:
                terms[t["native"]] += 1
                tnames[t["native"]] = (t["source"], t["name"])

        panel = names[np.argsort(absval.mean(axis=0))[::-1][:TOP]]
        psig = enrich(panel)
        hits = [sum(g in t for t in tops) for g in panel]

        top_rec = terms.most_common(1)[0][1] if terms else 0
        rows.append((TOP, np.mean(jac), sum(c == 0 for c in counts),
                     top_rec, np.mean(hits)))

        print("")
        print("=== TOP %d ===" % TOP)
        print("  Jaccard mean %.3f | panel gene appearance mean %.1f/%d"
              % (np.mean(jac), np.mean(hits), n_models))
        print("  terms per model: mean %.1f  zero-term models %d/%d  max %d"
              % (np.mean(counts), sum(c == 0 for c in counts), n_models,
                 max(counts) if counts else 0))
        print("  panel terms: %d | union across models: %d"
              % (len(psig), len(terms)))
        if terms:
            print("  top recurrence:")
            for nat, c in terms.most_common(5):
                s, n = tnames[nat]
                print("     %2d/%d  %-6s %s" % (c, n_models, s, n[:55]))

    # ---- summary table ----------------------------------------------
    print("")
    print("=== SUMMARY ===")
    print("  %-6s %-9s %-14s %-12s %s"
          % ("top-N", "Jaccard", "zero-term", "top recur", "panel appearance"))
    for TOP, j, z, tr, pa in rows:
        print("  %-6d %-9.3f %-14s %-12s %.1f/%d"
              % (TOP, j, "%d/%d" % (z, n_models), "%d/%d" % (tr, n_models),
                 pa, n_models))
    print("")
    print("Read the last two columns together: rising recurrence alongside")
    print("FALLING panel appearance indicates overlap by volume, not agreement.")


if __name__ == "__main__":
    main()
