#!/usr/bin/env python3
"""
tcga_basin_recurrence_check.py

Per-split, per-regime pathway recurrence for the TCGA regularisation-path
results.

Each split is read separately -- no averaging of centroids across splits, which
would conflate the very multiplicity being measured. For each split with the
target k*, the top-N genes of each regime centroid are enriched independently
and a recurrence table is built across splits.

Term-gene intersections are retained (no_evidences=False). These matter: several
ontology terms in these results are driven by the same small gene set, so term
counts overstate the evidence unless the intersections are inspected.

Outputs (in --save_dir):
  <lens>_per_split_enrichment_records.json   per split, per regime, with intersections
  <lens>_recurrence_summary.json             term recurrence across splits

Usage:
  python tcga_basin_recurrence_check.py <aggregates_dir> <save_dir> [options]

  <aggregates_dir> contains split<seed>/aggregate_split<seed>.npz
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_split_data(output_dir, seed, lens):
    agg = Path(output_dir) / ("split%d" % seed) / ("aggregate_split%d.npz" % seed)
    if not agg.exists():
        agg = Path(output_dir) / ("aggregate_split%d.npz" % seed)
    if not agg.exists():
        return None
    data = np.load(agg, allow_pickle=True)
    k_key = "k_star_%s" % lens
    c_key = "centroids_normed_%s" % lens
    if k_key not in data.files and "k_star" not in data.files:
        return None
    k = int(data[k_key]) if k_key in data.files else int(data["k_star"])
    centroids = data[c_key] if c_key in data.files else data["centroids_normed"]
    return {"seed": seed, "k": k,
            "centroids": np.asarray(centroids),
            "feature_names": list(data["feature_names"])}


def get_dominant_k(output_dir, split_range, lens):
    counts = Counter()
    for seed in split_range:
        d = load_split_data(output_dir, seed, lens)
        if d is not None:
            counts[d["k"]] += 1
    if not counts:
        raise RuntimeError("No splits found for lens=%s in %s" % (lens, output_dir))
    return counts.most_common(1)[0][0], counts


def get_top_genes(centroid, feature_names, top_n):
    ranked = np.argsort(np.abs(centroid))[::-1][:top_n]
    out = []
    for idx in ranked:
        ensg_v = feature_names[idx]
        out.append((ensg_v.split(".")[0], ensg_v, float(centroid[idx])))
    return out


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aggregates_dir",
                    help="directory containing split<seed>/aggregate_split<seed>.npz")
    ap.add_argument("save_dir", help="destination for the json outputs")
    ap.add_argument("--lens", nargs="+", default=["lime"],
                    choices=["lime", "shap"])
    ap.add_argument("--split_start", type=int, default=800)
    ap.add_argument("--split_stop", type=int, default=900)
    ap.add_argument("--top_n", type=int, default=50,
                    help="genes per regime centroid (default 50)")
    ap.add_argument("--top_terms_per_basin", type=int, default=25,
                    help="terms retained per split-regime (default 25). "
                         "Set this generously: a small value makes terms look "
                         "regime-exclusive when they are merely lower-ranked.")
    ap.add_argument("--target_k", type=int, default=None,
                    help="force a k*; default uses the dominant k* per lens")
    ap.add_argument("--max_splits", type=int, default=None,
                    help="analyse at most this many splits (for a quick pass)")
    ap.add_argument("--sources", nargs="+",
                    default=["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"])
    ap.add_argument("--organism", default="hsapiens")
    ap.add_argument("--user_threshold", type=float, default=0.05)
    args = ap.parse_args()

    try:
        from gprofiler import GProfiler
    except ImportError:
        sys.exit("ERROR: gprofiler-official not installed.\n"
                 "  pip install gprofiler-official --break-system-packages")

    gp = GProfiler(return_dataframe=False)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    split_range = range(args.split_start, args.split_stop)

    def run_enrichment(gene_ids, label):
        r = gp.profile(organism=args.organism, query=gene_ids,
                       sources=args.sources, user_threshold=args.user_threshold,
                       no_evidences=False,
                       significance_threshold_method="g_SCS")
        sig = [t for t in r if t.get("significant", False)]
        print("  [%s] %d significant terms" % (label, len(sig)))
        return sig

    for lens in args.lens:
        dominant_k, k_counts = get_dominant_k(args.aggregates_dir,
                                              split_range, lens)
        target_k = args.target_k if args.target_k is not None else dominant_k

        print("")
        print("#" * 78)
        print("LENS=%s | dominant k*=%d | using target k*=%d"
              % (lens.upper(), dominant_k, target_k))
        print("k* counts: %s" % dict(k_counts))
        print("#" * 78)

        selected = []
        for seed in split_range:
            d = load_split_data(args.aggregates_dir, seed, lens)
            if d is not None and d["k"] == target_k:
                selected.append(d)
        if args.max_splits is not None:
            selected = selected[:args.max_splits]
        if not selected:
            print("No splits with k*=%d for lens=%s" % (target_k, lens))
            continue

        print("Analysing %d splits for %s with k*=%d\n"
              % (len(selected), lens, target_k))

        term_counts_by_basin = defaultdict(Counter)
        split_records = []

        for d in selected:
            seed = d["seed"]
            split_out = {"seed": seed, "lens": lens, "k": int(d["k"]),
                         "basins": []}
            print("--- split %d ---" % seed)
            for b in range(target_k):
                genes = get_top_genes(d["centroids"][b], d["feature_names"],
                                      args.top_n)
                sig = run_enrichment([g[0] for g in genes],
                                     "split %d basin %d" % (seed, b))
                best = sorted(sig, key=lambda x: x["p_value"])[:args.top_terms_per_basin]

                for r in best:
                    term_counts_by_basin[b][(r["source"], r["native"], r["name"])] += 1

                split_out["basins"].append({
                    "basin": b,
                    "top_genes": genes,
                    "n_significant_terms": len(sig),
                    "top_terms": [{"source": r["source"], "native": r["native"],
                                   "name": r["name"], "p_value": r["p_value"],
                                   "term_size": r.get("term_size"),
                                   "intersection_size": r.get("intersection_size"),
                                   "intersections": r.get("intersections")}
                                  for r in best]})
            split_records.append(split_out)

        recurrence = {"lens": lens, "target_k": target_k,
                      "n_splits_analyzed": len(selected),
                      "split_seeds": [d["seed"] for d in selected],
                      "top_n_genes": args.top_n,
                      "top_terms_per_basin": args.top_terms_per_basin,
                      "sources": args.sources,
                      "per_basin_term_recurrence": {}}

        print("")
        print("=" * 78)
        print("RECURRENCE SUMMARY - %s | k*=%d | n=%d splits"
              % (lens.upper(), target_k, len(selected)))
        print("=" * 78)

        for b in range(target_k):
            common = term_counts_by_basin[b].most_common(20)
            recurrence["per_basin_term_recurrence"][str(b)] = [
                {"source": src, "native": native, "name": name,
                 "count": count, "fraction_of_splits": count / len(selected)}
                for (src, native, name), count in common]

            print("")
            print("Basin %d: most recurrent terms" % b)
            if not common:
                print("  (no significant terms)")
                continue
            for (src, native, name), count in common[:10]:
                print("  %3d/%d  (%.2f)  %-6s  %s"
                      % (count, len(selected), count / len(selected), src, name))

        save_json(split_records,
                  save_dir / ("%s_per_split_enrichment_records.json" % lens))
        save_json(recurrence, save_dir / ("%s_recurrence_summary.json" % lens))
        print("")
        print("[Saved] %s" % (save_dir / ("%s_per_split_enrichment_records.json" % lens)))
        print("[Saved] %s" % (save_dir / ("%s_recurrence_summary.json" % lens)))

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
