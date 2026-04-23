#!/usr/bin/env python3
"""
tcga_test13_story_strength.py
===============================
EvoXplain Falsification Test 13 — "Story dominance only after aggregation."
TCGA Tumour vs Normal, LR C-grid, PER LENS.

Version: v2 (standardised, verdict block added)
Last revised: April 2026

Attack
------
The divergent biological stories (extracellular vs metabolism) only
appear when you average across models. Individual models do not
consistently commit to one story — the basin centroids are aggregation
artefacts.

Defence Strategy
----------------
1. PER-MODEL STORY STRENGTH: For each individual LR fit, compute
   attribution mass assigned to story-A and story-B gene sets.
2. PER-BASIN DOMINANCE FRACTION: What fraction of models within each
   basin consistently favour one story over the other?
3. EXCLUSIVE-GENE ANALYSIS: Repeat with overlapping genes removed.

Kill Condition
--------------
  In at least one basin, >= dominance_threshold fraction of individual
  models show storyB > storyA (or vice versa)
  AND in another basin, the opposite dominance holds
  → ATTACK KILLED: story dominance is a per-model property

Data
----
Reads frozen aggregates + external gene-set files. No retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results/final_freeze/tcga_lr_cgrid_shap_lime_100x100")
    p.add_argument("--split-start", type=int, default=800)
    p.add_argument("--split-end", type=int, default=900, help="exclusive end")
    p.add_argument("--lens", choices=["shap", "lime"], default="shap")
    p.add_argument("--target-k", type=int, default=None,
                   help="Only analyze splits with this k*. Default: dominant k* for this lens")
    p.add_argument("--story-a-name", default="extracellular")
    p.add_argument("--story-b-name", default="metabolism")
    p.add_argument("--story-a-file", default=None,
                   help="Plain-text gene file for story A. Can be one gene per line or the gene-list files saved by tcga_basin_enrichment.py")
    p.add_argument("--story-b-file", default=None,
                   help="Plain-text gene file for story B. Can be one gene per line or the gene-list files saved by tcga_basin_enrichment.py")
    p.add_argument("--story-a-inline", nargs="*", default=None,
                   help="Optional inline gene list for story A")
    p.add_argument("--story-b-inline", nargs="*", default=None,
                   help="Optional inline gene list for story B")
    p.add_argument("--top-splits", type=int, default=None,
                   help="Quick mode: only analyze the first N matching splits")
    p.add_argument("--signed", action="store_true",
                   help="Use signed sums instead of absolute sums")
    p.add_argument("--save-dir", default="results/final_freeze/tcga_story_strength")
    p.add_argument("--dominance-threshold", type=float, default=0.90,
                   help="Min fraction of models in a basin that must favour one story "
                        "for that basin to count as 'committed'. Default 0.90.")
    return p.parse_args()


def norm_gene(g: str) -> str:
    return g.strip().split(".")[0].upper()


def load_gene_set(path: str | None, inline: Sequence[str] | None) -> Set[str]:
    genes: Set[str] = set()
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Gene file not found: {p}")
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("ensembl_id_versioned"):
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith("ENSG"):
                    genes.add(norm_gene(parts[1]))
                else:
                    genes.add(norm_gene(parts[0]))
    if inline:
        for g in inline:
            genes.add(norm_gene(g))
    return genes


def pick_key(data: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> str:
    for k in candidates:
        if k in data:
            return k
    raise KeyError(f"None of these keys found: {candidates}. Available keys: {list(data.keys())}")


def get_dominant_k(output_dir: Path, split_range: Iterable[int], lens: str) -> Tuple[int, Dict[int, int]]:
    counts = Counter()
    for seed in split_range:
        agg = output_dir / f"split{seed}" / f"aggregate_split{seed}.npz"
        if not agg.exists():
            continue
        data = np.load(agg, allow_pickle=True)
        k_key = f"k_star_{lens}" if f"k_star_{lens}" in data else ("k_star" if "k_star" in data else None)
        if k_key is None:
            continue
        counts[int(data[k_key])] += 1
    if not counts:
        raise RuntimeError(f"No valid aggregate_split*.npz files found in {output_dir}")
    dominant_k, _ = counts.most_common(1)[0]
    return dominant_k, dict(counts)


def load_split_runs(agg_path: Path, lens: str) -> Dict[str, np.ndarray]:
    data = np.load(agg_path, allow_pickle=True)

    feature_key = pick_key(data, ["feature_names"])
    k_key = pick_key(data, [f"k_star_{lens}", "k_star"])
    labels_key = pick_key(data, [f"cluster_labels_{lens}", "cluster_labels"])
    exp_key = pick_key(data, [f"expvec_normed_{lens}", f"expvec_raw_{lens}", "expvec_normed", "expvec_raw"])

    run_ids_key = None
    for cand in [f"run_ids_{lens}", "run_ids"]:
        if cand in data:
            run_ids_key = cand
            break

    c_values_key = None
    for cand in [f"run_C_values_{lens}", "run_C_values"]:
        if cand in data:
            c_values_key = cand
            break

    l1_values_key = None
    for cand in [f"run_l1_values_{lens}", "run_l1_values"]:
        if cand in data:
            l1_values_key = cand
            break

    out = {
        "feature_names": np.asarray(data[feature_key]),
        "k": int(data[k_key]),
        "cluster_labels": np.asarray(data[labels_key]),
        "expvecs": np.asarray(data[exp_key]),
    }
    if run_ids_key:
        out["run_ids"] = np.asarray(data[run_ids_key])
    if c_values_key:
        out["run_C_values"] = np.asarray(data[c_values_key])
    if l1_values_key:
        out["run_l1_values"] = np.asarray(data[l1_values_key])
    return out


def build_index(feature_names: Sequence[str]) -> Dict[str, int]:
    return {norm_gene(g): i for i, g in enumerate(feature_names)}


def strength(expvec: np.ndarray, idxs: Sequence[int], signed: bool = False) -> float:
    vals = expvec[idxs]
    return float(np.sum(vals) if signed else np.sum(np.abs(vals)))


def summarise(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def fmt_stats(d: Dict[str, float]) -> str:
    if d.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={d['n']} | mean={d['mean']:.4f} | std={d['std']:.4f} | "
        f"min={d['min']:.4f} | q25={d['q25']:.4f} | median={d['median']:.4f} | "
        f"q75={d['q75']:.4f} | max={d['max']:.4f}"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    story_a = load_gene_set(args.story_a_file, args.story_a_inline)
    story_b = load_gene_set(args.story_b_file, args.story_b_inline)

    if not story_a:
        raise ValueError("Story A gene set is empty. Provide --story-a-file or --story-a-inline.")
    if not story_b:
        raise ValueError("Story B gene set is empty. Provide --story-b-file or --story-b-inline.")

    overlap = story_a & story_b
    only_a = story_a - story_b
    only_b = story_b - story_a

    split_range = range(args.split_start, args.split_end)
    dominant_k, k_counts = get_dominant_k(output_dir, split_range, args.lens)
    target_k = args.target_k if args.target_k is not None else dominant_k

    print("=" * 78)
    print(f"EvoXplain — Test 13: Story Strength (TCGA LR-Cgrid)")
    print(f"  'Is story dominance a per-model property or only after aggregation?'")
    print("=" * 78)
    print(f"  Aggregates             : {args.output_dir}")
    print(f"  Lens                   : {args.lens}")
    print(f"  Splits                 : {args.split_start}..{args.split_end}")
    print(f"  Dominance threshold    : {args.dominance_threshold}")
    print(f"  Story A ({args.story_a_name:>12})  : {len(story_a)} genes")
    print(f"  Story B ({args.story_b_name:>12})  : {len(story_b)} genes")
    print(f"  Overlap                : {len(overlap)} genes")
    if overlap:
        print("  WARNING: story sets overlap. Also computing exclusive-only strengths.")
    print("=" * 78)

    rows: List[Dict[str, object]] = []
    matched_splits = 0

    for seed in split_range:
        agg = output_dir / f"split{seed}" / f"aggregate_split{seed}.npz"
        if not agg.exists():
            continue
        d = load_split_runs(agg, args.lens)
        if d["k"] != target_k:
            continue

        feature_names = list(d["feature_names"])
        idx_map = build_index(feature_names)
        idx_a = [idx_map[g] for g in story_a if g in idx_map]
        idx_b = [idx_map[g] for g in story_b if g in idx_map]
        idx_overlap = [idx_map[g] for g in overlap if g in idx_map]
        idx_only_a = [idx_map[g] for g in only_a if g in idx_map]
        idx_only_b = [idx_map[g] for g in only_b if g in idx_map]

        if not idx_a:
            raise RuntimeError(f"No story A genes found in features for split {seed}")
        if not idx_b:
            raise RuntimeError(f"No story B genes found in features for split {seed}")

        expvecs = np.asarray(d["expvecs"])
        labels = np.asarray(d["cluster_labels"]).astype(int)
        n_runs = expvecs.shape[0]

        run_ids = np.asarray(d.get("run_ids", np.arange(n_runs)))
        c_values = np.asarray(d.get("run_C_values", np.full(n_runs, np.nan)))
        l1_values = np.asarray(d.get("run_l1_values", np.full(n_runs, np.nan)))

        if labels.shape[0] != n_runs:
            raise RuntimeError(f"Mismatch in split {seed}: {n_runs} expvecs vs {labels.shape[0]} labels")

        matched_splits += 1
        for i in range(n_runs):
            exp = np.asarray(expvecs[i], dtype=float)
            a_strength = strength(exp, idx_a, signed=args.signed)
            b_strength = strength(exp, idx_b, signed=args.signed)
            overlap_strength = strength(exp, idx_overlap, signed=args.signed) if idx_overlap else 0.0
            only_a_strength = strength(exp, idx_only_a, signed=args.signed) if idx_only_a else 0.0
            only_b_strength = strength(exp, idx_only_b, signed=args.signed) if idx_only_b else 0.0

            ratio_b_over_a = b_strength / (a_strength + EPS)
            ratio_only_b_over_only_a = only_b_strength / (only_a_strength + EPS) if (idx_only_a or idx_only_b) else np.nan
            dominant_story = args.story_b_name if b_strength > a_strength else args.story_a_name
            dominant_exclusive = None
            if idx_only_a or idx_only_b:
                dominant_exclusive = args.story_b_name if only_b_strength > only_a_strength else args.story_a_name

            rows.append({
                "split_seed": seed,
                "lens": args.lens,
                "k_star": d["k"],
                "run_index": i,
                "run_id": str(run_ids[i]),
                "basin": int(labels[i]),
                "C_value": float(c_values[i]) if np.isfinite(c_values[i]) else None,
                "l1_value": float(l1_values[i]) if np.isfinite(l1_values[i]) else None,
                f"{args.story_a_name}_strength": a_strength,
                f"{args.story_b_name}_strength": b_strength,
                "overlap_strength": overlap_strength,
                f"only_{args.story_a_name}_strength": only_a_strength,
                f"only_{args.story_b_name}_strength": only_b_strength,
                f"ratio_{args.story_b_name}_over_{args.story_a_name}": ratio_b_over_a,
                f"ratio_only_{args.story_b_name}_over_only_{args.story_a_name}": ratio_only_b_over_only_a,
                "dominant_story": dominant_story,
                "dominant_story_exclusive_only": dominant_exclusive,
            })

        if args.top_splits is not None and matched_splits >= args.top_splits:
            break

    if not rows:
        raise RuntimeError("No matching runs found. Check --target-k, split range, and file paths.")

    # Save per-run CSV
    csv_path = save_dir / f"{args.lens}_k{target_k}_{args.story_a_name}_vs_{args.story_b_name}_per_model.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summaries by basin
    by_basin: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_basin[int(r["basin"])].append(r)

    summary = {
        "lens": args.lens,
        "target_k": target_k,
        "k_counts": k_counts,
        "n_matching_splits": matched_splits,
        "story_a_name": args.story_a_name,
        "story_b_name": args.story_b_name,
        "story_a_gene_count": len(story_a),
        "story_b_gene_count": len(story_b),
        "overlap_gene_count": len(overlap),
        "overlap_genes": sorted(overlap),
        "per_basin": {},
    }

    print("\n" + "#" * 78)
    print(f"PER-BASIN STORY STRENGTH SUMMARY — {args.lens.upper()} | k*={target_k}")
    print("#" * 78)

    ratio_key = f"ratio_{args.story_b_name}_over_{args.story_a_name}"
    ratio_ex_key = f"ratio_only_{args.story_b_name}_over_only_{args.story_a_name}"
    a_key = f"{args.story_a_name}_strength"
    b_key = f"{args.story_b_name}_strength"
    only_a_key = f"only_{args.story_a_name}_strength"
    only_b_key = f"only_{args.story_b_name}_strength"

    for basin in sorted(by_basin):
        rr = by_basin[basin]
        ratios = [float(r[ratio_key]) for r in rr]
        ratios_ex = [float(r[ratio_ex_key]) for r in rr if r[ratio_ex_key] is not None and not np.isnan(r[ratio_ex_key])]
        a_vals = [float(r[a_key]) for r in rr]
        b_vals = [float(r[b_key]) for r in rr]
        only_a_vals = [float(r[only_a_key]) for r in rr]
        only_b_vals = [float(r[only_b_key]) for r in rr]

        n_b_dominant = sum(float(r[b_key]) > float(r[a_key]) for r in rr)
        n_b_dominant_ex = sum(float(r[only_b_key]) > float(r[only_a_key]) for r in rr) if (only_a or only_b) else None
        top_models = sorted(rr, key=lambda x: float(x[ratio_key]), reverse=True)[:10]

        summary["per_basin"][str(basin)] = {
            "n_models": len(rr),
            f"{args.story_a_name}_strength": summarise(a_vals),
            f"{args.story_b_name}_strength": summarise(b_vals),
            ratio_key: summarise(ratios),
            only_a_key: summarise(only_a_vals),
            only_b_key: summarise(only_b_vals),
            ratio_ex_key: summarise(ratios_ex) if ratios_ex else {"n": 0},
            "n_models_where_storyB_gt_storyA": n_b_dominant,
            "fraction_models_where_storyB_gt_storyA": n_b_dominant / len(rr),
            "n_models_where_storyB_exclusive_gt_storyA_exclusive": n_b_dominant_ex,
            "top_10_models_by_ratio": [
                {
                    "split_seed": int(x["split_seed"]),
                    "run_index": int(x["run_index"]),
                    "run_id": x["run_id"],
                    "C_value": x["C_value"],
                    a_key: float(x[a_key]),
                    b_key: float(x[b_key]),
                    ratio_key: float(x[ratio_key]),
                    only_a_key: float(x[only_a_key]),
                    only_b_key: float(x[only_b_key]),
                    ratio_ex_key: float(x[ratio_ex_key]) if x[ratio_ex_key] is not None and not np.isnan(x[ratio_ex_key]) else None,
                }
                for x in top_models
            ],
        }

        print(f"\nBasin {basin}")
        print(f"  {args.story_a_name:>14}: {fmt_stats(summarise(a_vals))}")
        print(f"  {args.story_b_name:>14}: {fmt_stats(summarise(b_vals))}")
        print(f"  {ratio_key:>14}: {fmt_stats(summarise(ratios))}")
        print(f"  storyB > storyA in {n_b_dominant}/{len(rr)} models ({n_b_dominant/len(rr):.3f})")
        if overlap:
            print(f"  only {args.story_a_name:>9}: {fmt_stats(summarise(only_a_vals))}")
            print(f"  only {args.story_b_name:>9}: {fmt_stats(summarise(only_b_vals))}")
            print(f"  {ratio_ex_key:>14}: {fmt_stats(summarise(ratios_ex)) if ratios_ex else 'n=0'}")
            if n_b_dominant_ex is not None:
                print(f"  storyB_exclusive > storyA_exclusive in {n_b_dominant_ex}/{len(rr)} models ({n_b_dominant_ex/len(rr):.3f})")

        print("  top models by storyB/storyA ratio:")
        for x in top_models[:5]:
            print(
                f"    split={x['split_seed']} run={x['run_index']} basin={x['basin']} "
                f"C={x['C_value']} ratio={float(x[ratio_key]):.4f} "
                f"{args.story_a_name}={float(x[a_key]):.4f} {args.story_b_name}={float(x[b_key]):.4f}"
            )

    json_path = save_dir / f"{args.lens}_k{target_k}_{args.story_a_name}_vs_{args.story_b_name}_summary.json"

    # ==================================================================
    # VERDICT
    # ==================================================================
    print(f"\n{'='*78}")
    print(f"TEST 13 VERDICT — {args.lens.upper()}")
    print("=" * 78)

    # Check: is there at least one basin where storyB dominates and one where storyA dominates?
    b_key_verdict = f"{args.story_b_name}_strength"
    a_key_verdict = f"{args.story_a_name}_strength"
    basins_committed_to_b = []
    basins_committed_to_a = []
    basin_dominance_summary = {}

    for basin_str, basin_data in summary["per_basin"].items():
        frac_b = basin_data["fraction_models_where_storyB_gt_storyA"]
        n_models = basin_data["n_models"]
        basin_dominance_summary[basin_str] = {
            "n_models": n_models,
            "frac_storyB_dominant": round(frac_b, 4),
        }
        if frac_b >= args.dominance_threshold:
            basins_committed_to_b.append(int(basin_str))
            print(f"  Basin {basin_str}: {frac_b*100:.1f}% → story B dominant ({n_models} models)")
        elif (1.0 - frac_b) >= args.dominance_threshold:
            basins_committed_to_a.append(int(basin_str))
            print(f"  Basin {basin_str}: {(1-frac_b)*100:.1f}% → story A dominant ({n_models} models)")
        else:
            print(f"  Basin {basin_str}: {frac_b*100:.1f}% → mixed ({n_models} models)")

    opposite_commitment = bool(basins_committed_to_a and basins_committed_to_b)
    any_commitment = bool(basins_committed_to_a or basins_committed_to_b)

    if opposite_commitment:
        tier = "ATTACK KILLED"
        v = (f"ATTACK KILLED ({args.lens.upper()}) — "
             f"Basins {basins_committed_to_a} commit to story A at "
             f">={args.dominance_threshold*100:.0f}% per-model level; "
             f"basins {basins_committed_to_b} commit to story B. "
             f"Story dominance is a per-model property, not an aggregation artefact.")
    elif any_commitment:
        tier = "PARTIAL"
        v = (f"PARTIAL ({args.lens.upper()}) — "
             f"Some basins show strong commitment but opposite commitment "
             f"is not established across basins.")
    else:
        tier = "INCONCLUSIVE"
        v = (f"INCONCLUSIVE ({args.lens.upper()}) — "
             f"No basin reaches {args.dominance_threshold*100:.0f}% dominance "
             f"for either story.")

    print(f"\n  >>> {v}")
    print("=" * 78)

    # Standardised JSON
    summary["test_id"] = "test13_story_strength"
    summary["version"] = "v2"
    summary["dominance_threshold"] = args.dominance_threshold
    summary["basin_dominance_summary"] = basin_dominance_summary
    summary["verdict_tier"] = tier
    summary["verdict_text"] = v

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {json_path}")


if __name__ == "__main__":
    main()
