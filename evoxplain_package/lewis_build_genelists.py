#!/usr/bin/env python3
"""
lewis_build_genelists.py

Reproduces the agreement statistics for the Lewis & Kemp retrospective audit
and writes the gene lists used by the enrichment step.

Reads the per-refit pickles produced by lewis_baselearner_faithful.py. Each
holds per-sample SHAP values for that refit's own test draw; the per-model
explanation vector is the mean across that model's test samples.

Two aggregations are computed:
  signed  = mean SHAP           -> direction agreement between models
  absval  = mean |SHAP|         -> importance ranking, used for gene lists

Outputs:
  <out_prefix>_genelists.json    panel, per-model top-N lists, AUROCs
  <out_prefix>_background.json   all measured genes, for enrichment background

Usage:
  python lewis_build_genelists.py <refits_dir> [--out_prefix lewis] [--top 50]

  <refits_dir> contains split_*.pickle
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refits_dir",
                    help="directory containing split_*.pickle")
    ap.add_argument("--out_prefix", default="lewis",
                    help="prefix for output json files (default: lewis)")
    ap.add_argument("--top", type=int, default=50,
                    help="genes per model list (default 50)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.refits_dir, "split_*.pickle")),
                   key=lambda p: int(re.search(r"split_(\d+)", p).group(1)))
    if not paths:
        sys.exit("ERROR: no split_*.pickle found in %s" % args.refits_dir)
    print("models found: %d" % len(paths))

    signed, absval, names, aurocs = [], [], None, []
    for p in paths:
        with open(p, "rb") as f:
            import pickle
            d = pickle.load(f)
        sv = d["shap_values"]
        signed.append(sv.mean(axis=0))
        absval.append(np.abs(sv).mean(axis=0))
        aurocs.append(d["auroc"])
        if names is None:
            names = [f.split("#")[-1].strip() for f in d["feature_names"]]

    signed = np.vstack(signed)
    absval = np.vstack(absval)
    names_arr = np.array(names)

    print("AUROC: %.3f - %.3f (mean %.3f)"
          % (min(aurocs), max(aurocs), np.mean(aurocs)))
    print("genes: %d | gene name examples: %s" % (len(names), names[:3]))

    # ---- signed direction agreement ---------------------------------
    S = signed / np.linalg.norm(signed, axis=1, keepdims=True)
    C = S @ S.T
    iu = np.triu_indices(len(paths), 1)
    cos = C[iu]
    print("")
    print("signed cosine: mean %.3f  median %.3f  frac negative %.2f"
          % (cos.mean(), np.median(cos), (cos < 0).mean()))

    # ---- per-model gene lists ---------------------------------------
    TOP = args.top
    tops = [set(names_arr[np.argsort(absval[i])[::-1][:TOP]])
            for i in range(len(paths))]
    jac = [len(tops[i] & tops[j]) / len(tops[i] | tops[j])
           for i, j in zip(*iu)]
    ov = [len(tops[i] & tops[j]) for i, j in zip(*iu)]
    print("top-%d Jaccard between models: mean %.3f  min %.3f  max %.3f"
          % (TOP, np.mean(jac), min(jac), max(jac)))
    print("top-%d shared genes:           mean %.1f/%d  min %d  max %d"
          % (TOP, np.mean(ov), TOP, min(ov), max(ov)))

    # ---- the averaged "consensus" panel ------------------------------
    panel = list(names_arr[np.argsort(absval.mean(axis=0))[::-1][:TOP]])
    hits = [sum(g in t for t in tops) for g in panel]
    print("")
    print("averaged panel: mean appearances in an individual model top-%d: %.1f/%d"
          % (TOP, np.mean(hits), len(paths)))
    print("  appearing in <=5 of %d models: %d/%d"
          % (len(paths), sum(h <= 5 for h in hits), TOP))
    print("  worst 5: %s"
          % [(g, h) for g, h in sorted(zip(panel, hits), key=lambda z: z[1])[:5]])

    # ---- outputs ------------------------------------------------------
    gl_path = "%s_genelists.json" % args.out_prefix
    bg_path = "%s_background.json" % args.out_prefix
    with open(gl_path, "w") as f:
        json.dump({"panel": panel,
                   "models": [sorted(t) for t in tops],
                   "aurocs": aurocs,
                   "top_n": TOP,
                   "n_models": len(paths)}, f, indent=1)
    with open(bg_path, "w") as f:
        json.dump(names, f)
    print("")
    print("wrote %s and %s" % (gl_path, bg_path))


if __name__ == "__main__":
    main()
