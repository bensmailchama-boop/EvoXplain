#!/usr/bin/env python3
"""
tcga_test10_clinical_meaning.py
================================
EvoXplain Falsification Test 10 — "This isn't clinically meaningful."
TCGA Tumour vs Normal, LR C-grid, DUAL LENS (SHAP + LIME).

Version: v2 (standardised)
Last revised: April 2026

Attack
------
So different C values weight genes differently. This has no practical or
clinical consequence. Nobody cares which genes the model uses internally
as long as accuracy is high.

Defence Strategy
----------------
1. PER-SPLIT RANK DISPLACEMENT: Within the exact same dataset split,
   models from different basins learn fundamentally different feature
   importances. Top genes are displaced entirely.
2. REGULATORY / AUDIT RISK: Two equally valid, highly accurate model
   deployments producing macro-level explanations that contradict each
   other creates a severe regulatory and scientific problem.

Kill Condition
--------------
  Max top-K displaced fraction > displacement_threshold
  → ATTACK KILLED (per lens): multiplicity has direct scientific and
    regulatory consequences through divergent narratives

Data
----
Reads frozen aggregates (centroids required). No retraining.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations


# =====================================================================
# RANK DISPLACEMENT
# =====================================================================

def rank_displacement(ranking_a, ranking_b, top_k=10):
    """Compute rank displacement between two feature rankings.

    For each of the top_k features in ranking_a, find its rank in
    ranking_b. Returns mean displacement and fraction of top_k that
    fall outside ranking_b's top_k.
    """
    top_a = set(ranking_a[:top_k])
    top_b = set(ranking_b[:top_k])

    displaced = len(top_a - top_b)
    displaced_frac = displaced / top_k

    rank_b_lookup = {f: r for r, f in enumerate(ranking_b)}
    displacements = []
    for f in ranking_a[:top_k]:
        rank_in_b = rank_b_lookup.get(f, len(ranking_b))
        displacements.append(abs(rank_in_b - ranking_a.tolist().index(f)))

    return {
        "mean_displacement": float(np.mean(displacements)),
        "displaced_frac": float(displaced_frac),
        "displaced_count": int(displaced),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoXplain TCGA Test 10: Clinical Meaningfulness")
    parser.add_argument("--output_dir_agg", type=str,
                        default="results/final_freeze/tcga_lr_cgrid_shap_lime_100x100")
    parser.add_argument("--split_start", type=int, default=800)
    parser.add_argument("--split_end",   type=int, default=900)
    parser.add_argument("--top_k",       type=int, default=10,
                        help="Number of top features to track for displacement.")
    parser.add_argument("--displacement_threshold", type=float, default=0.30,
                        help="Min displaced fraction for 'clinical impact'.")
    parser.add_argument("--output_dir",  type=str, default=".")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EvoXplain — Test 10: Clinical Meaningfulness (TCGA LR-Cgrid)")
    print("  'Does this multiplicity actually matter in practice?'")
    print("=" * 70)
    print(f"  Aggregates             : {args.output_dir_agg}")
    print(f"  Splits                 : {args.split_start}..{args.split_end}")
    print(f"  Top-K                  : {args.top_k}")
    print(f"  Displacement threshold : {args.displacement_threshold}")

    # ==================================================================
    # LOAD
    # ==================================================================
    print(f"\n[Loading] {args.output_dir_agg}")

    feature_names = None
    all_splits_data = []
    n_loaded = 0

    for seed in range(args.split_start, args.split_end):
        agg = Path(args.output_dir_agg) / f"split{seed}" / f"aggregate_split{seed}.npz"
        if not agg.exists():
            continue
        data = np.load(agg, allow_pickle=True)
        if feature_names is None:
            feature_names = list(data["feature_names"])

        split_record = {"seed": seed}
        for lens in ["shap", "lime"]:
            k_key = f"k_star_{lens}"
            c_key = f"centroids_normed_{lens}"
            split_record[f"k_{lens}"] = int(data[k_key]) if k_key in data else int(data["k_star"])
            split_record[f"centroids_{lens}"] = data[c_key] if c_key in data else data["centroids_normed"]

        all_splits_data.append(split_record)
        n_loaded += 1

    print(f"[Loaded] {n_loaded} splits")

    results = {
        "test_id": "test10_clinical_meaning",
        "version": "v2",
        "config": vars(args),
        "n_splits": n_loaded,
    }

    # ==================================================================
    # PER-LENS ANALYSIS
    # ==================================================================
    for lens in ["shap", "lime"]:
        print(f"\n{'='*70}")
        print(f"  {lens.upper()} — RANK DISPLACEMENT ANALYSIS")
        print(f"{'='*70}")

        k_counts = Counter([s[f"k_{lens}"] for s in all_splits_data])
        dom_k = k_counts.most_common(1)[0][0]
        print(f"  Dominant k*={dom_k} ({k_counts[dom_k]}/{n_loaded} splits)")

        if dom_k < 2:
            print(f"  [WARNING] Dominant k* < 2.")
            results[f"per_lens_{lens}"] = {
                "verdict_tier": "INCONCLUSIVE",
                "verdict_text": f"INCONCLUSIVE ({lens.upper()}) — No multiplicity.",
            }
            continue

        displacement_acc = defaultdict(lambda: {
            "mean_disp": [], "frac": [], "count": []
        })
        valid_splits = 0

        for s in all_splits_data:
            k = s[f"k_{lens}"]
            if k != dom_k:
                continue

            centroids = s[f"centroids_{lens}"]
            basin_rankings = {}
            for b in range(dom_k):
                basin_rankings[b] = np.argsort(np.abs(centroids[b]))[::-1]

            for i, j in combinations(range(dom_k), 2):
                pair_key = f"{i}_vs_{j}"
                rd = rank_displacement(basin_rankings[i], basin_rankings[j],
                                       top_k=args.top_k)
                displacement_acc[pair_key]["mean_disp"].append(rd["mean_displacement"])
                displacement_acc[pair_key]["frac"].append(rd["displaced_frac"])
                displacement_acc[pair_key]["count"].append(rd["displaced_count"])

            valid_splits += 1

        if valid_splits == 0:
            print(f"  [ERROR] No valid splits.")
            results[f"per_lens_{lens}"] = {
                "verdict_tier": "INCONCLUSIVE",
                "verdict_text": f"INCONCLUSIVE ({lens.upper()}) — No valid splits.",
            }
            continue

        final_displacement = {}
        for pair_key, metrics in displacement_acc.items():
            final_displacement[pair_key] = {
                "mean_displacement": round(float(np.mean(metrics["mean_disp"])), 1),
                "displaced_frac": round(float(np.mean(metrics["frac"])), 3),
                "displaced_count": round(float(np.mean(metrics["count"])), 1),
            }

        print(f"\n  --- Rank Displacement ({valid_splits} splits) ---")
        max_displaced_frac = 0.0

        for pair_key, rd in final_displacement.items():
            i, j = pair_key.split("_vs_")
            print(f"\n    Basin {i} vs Basin {j} (top-{args.top_k}):")
            print(f"      Mean rank displacement: {rd['mean_displacement']} positions")
            print(f"      Displaced out of top-{args.top_k}: "
                  f"{rd['displaced_count']}/{args.top_k} "
                  f"({rd['displaced_frac']*100:.0f}%)")
            max_displaced_frac = max(max_displaced_frac, rd["displaced_frac"])

        # Verdict
        has_impact = bool(max_displaced_frac > args.displacement_threshold)

        print(f"\n  --- {lens.upper()} VERDICT ---")
        print(f"    Max displacement: {max_displaced_frac*100:.0f}%")
        print(f"    Clinical impact (>{args.displacement_threshold*100:.0f}%): {has_impact}")

        if has_impact:
            tier = "ATTACK KILLED"
            v = (f"ATTACK KILLED ({lens.upper()}) — "
                 f"{max_displaced_frac*100:.0f}% of the top-{args.top_k} most "
                 f"important features are structurally displaced between basins "
                 f"within the exact same dataset split. The biological narrative "
                 f"diverges significantly depending on the hyperparameter choice, "
                 f"creating severe scientific and regulatory risks.")
        else:
            tier = "PARTIAL"
            v = (f"PARTIAL ({lens.upper()}) — Some displacement "
                 f"({max_displaced_frac*100:.0f}%) but below "
                 f"{args.displacement_threshold*100:.0f}% threshold.")

        print(f"    >>> {v}")

        results[f"per_lens_{lens}"] = {
            "verdict_tier": tier,
            "verdict_text": v,
            "metrics": {
                "dom_k": int(dom_k),
                "valid_splits": valid_splits,
                "max_displaced_frac": round(max_displaced_frac, 3),
                "displacement_per_pair": final_displacement,
            },
        }

    # Combined
    print(f"\n{'='*70}")
    lens_tiers = [results.get(f"per_lens_{l}", {}).get("verdict_tier", "INCONCLUSIVE")
                  for l in ["shap", "lime"]]
    if all(t == "ATTACK KILLED" for t in lens_tiers):
        combined_tier = "ATTACK KILLED"
        combined_text = ("BOTH LENSES: ATTACK KILLED — multiplicity creates severe "
                         "scientific and regulatory consequences through divergent "
                         "narratives.")
    else:
        combined_tier = "PARTIAL"
        combined_text = "MIXED — see per-lens verdicts."

    print(f"  COMBINED: {combined_text}")
    print("=" * 70)

    results["verdict_tier"] = combined_tier
    results["verdict_text"] = combined_text

    json_path = Path(args.output_dir) / "tcga_test10_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {json_path}")


if __name__ == "__main__":
    main()
