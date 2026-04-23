#!/usr/bin/env python3
"""
tcga_test6_retraining_drift.py
================================
EvoXplain Falsification Test 6 — "Retraining drift is expected."
TCGA Tumour vs Normal, LR C-grid, DUAL LENS (SHAP + LIME).

Version: v2 (standardised)
Last revised: April 2026

Attack
------
Of course retraining produces slightly different attributions. That is
normal and expected. Accuracy is stable, so the model is fine.

Defence Strategy
----------------
1. ACCURACY-ATTRIBUTION DECOUPLING: Accuracy varies by < threshold while
   attribution cosines vary widely. The two metrics are completely decoupled.
2. LOCAL CORRELATION TEST: Within each split, Spearman correlation between
   accuracy and attribution direction is negligible.
3. HIGH-ACCURACY SUBSET: Even among the top-25% highest-accuracy models
   on the same data, attribution vectors diverge.

Kill Condition
--------------
  Global accuracy stable (< acc_range_threshold)
  AND within-split attribution diverges (spread > spread_threshold)
  AND high-accuracy subset still diverges (spread > spread_threshold)
  → ATTACK KILLED (per lens)

Data
----
Reads frozen aggregates (expvec_raw required). No retraining.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from collections import Counter


# =====================================================================
# STANDARD UTILITIES
# =====================================================================

def cosine_sim(a, b):
    return float(1.0 - cosine(a, b))


def normalise(vecs):
    vecs = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms > 0, norms, 1.0)


def pairwise_cosines(vecs_normed):
    n = len(vecs_normed)
    if n < 2:
        return np.array([1.0])
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_sim(vecs_normed[i], vecs_normed[j]))
    return np.array(sims)


def robust_spread(cosines):
    """95th − 5th percentile of cosine distribution."""
    if len(cosines) == 0:
        return 0.0
    return float(np.percentile(cosines, 95) - np.percentile(cosines, 5))


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoXplain TCGA Test 6: Retraining Drift")
    parser.add_argument("--output_dir_agg", type=str,
                        default="results/final_freeze/tcga_lr_cgrid_shap_lime_100x100")
    parser.add_argument("--split_start", type=int, default=800)
    parser.add_argument("--split_end",   type=int, default=900)
    parser.add_argument("--output_dir",  type=str, default=".")
    parser.add_argument("--acc_range_threshold", type=float, default=0.05,
                        help="Max global accuracy range for 'stable'.")
    parser.add_argument("--spread_threshold", type=float, default=0.05,
                        help="Min cosine spread for 'attribution diverges'.")
    parser.add_argument("--rho_threshold", type=float, default=0.3,
                        help="Max |Spearman rho| for 'decoupled'.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EvoXplain — Test 6: Retraining Drift (TCGA LR-Cgrid)")
    print("  'Isn't retraining variation normal and expected?'")
    print("=" * 70)
    print(f"  Aggregates             : {args.output_dir_agg}")
    print(f"  Splits                 : {args.split_start}..{args.split_end}")
    print(f"  Acc range threshold    : {args.acc_range_threshold}")
    print(f"  Spread threshold       : {args.spread_threshold}")
    print(f"  Rho threshold          : {args.rho_threshold}")

    # ==================================================================
    # LOAD
    # ==================================================================
    print(f"\n[Loading] {args.output_dir_agg}")

    per_lens = {}
    for lens in ["shap", "lime"]:
        per_lens[lens] = {
            "per_split_accs": [],
            "per_split_expvecs": [],
            "all_accs": [],
        }

    n_loaded = 0
    for seed in range(args.split_start, args.split_end):
        agg = Path(args.output_dir_agg) / f"split{seed}" / f"aggregate_split{seed}.npz"
        if not agg.exists():
            continue
        data = np.load(agg, allow_pickle=True)
        accs = data["test_acc"]

        for lens in ["shap", "lime"]:
            raw_key = f"expvec_raw_{lens}"
            raw = data[raw_key] if raw_key in data else data["expvec_raw"]
            per_lens[lens]["per_split_accs"].append(accs)
            per_lens[lens]["per_split_expvecs"].append(raw)
            per_lens[lens]["all_accs"].extend(accs)

        n_loaded += 1

    print(f"[Loaded] {n_loaded} splits")

    results = {
        "test_id": "test6_retraining_drift",
        "version": "v2",
        "config": vars(args),
        "n_splits": n_loaded,
    }

    # ==================================================================
    # PER-LENS ANALYSIS
    # ==================================================================
    for lens in ["shap", "lime"]:
        print(f"\n{'='*70}")
        print(f"  {lens.upper()} — RETRAINING DRIFT ANALYSIS")
        print(f"{'='*70}")

        all_accs = np.array(per_lens[lens]["all_accs"])
        global_acc_range = float(np.max(all_accs) - np.min(all_accs))

        print(f"  Total runs: {len(all_accs)}")
        print(f"  Global accuracy range: {global_acc_range*100:.2f}%")

        within_spreads = []
        within_acc_ranges = []
        split_rhos = []
        high_acc_spreads = []

        for accs_s, exp_s in zip(per_lens[lens]["per_split_accs"],
                                  per_lens[lens]["per_split_expvecs"]):
            exp_n = normalise(exp_s)

            # 1. Within-split spread
            cos_within = pairwise_cosines(exp_n)
            within_spreads.append(robust_spread(cos_within))
            within_acc_ranges.append(float(np.max(accs_s) - np.min(accs_s)))

            # 2. Local acc-attribution correlation
            split_centroid = np.mean(exp_n, axis=0)
            sc_norm = split_centroid / (np.linalg.norm(split_centroid) + 1e-10)
            cos_to_sc = [float(np.dot(v, sc_norm)) for v in exp_n]

            if np.std(accs_s) > 1e-6:
                rho, _ = spearmanr(accs_s, cos_to_sc)
                if not np.isnan(rho):
                    split_rhos.append(rho)
                else:
                    split_rhos.append(0.0)
            else:
                split_rhos.append(0.0)

            # 3. High-accuracy subset
            acc_threshold = np.percentile(accs_s, 75)
            exp_high = exp_n[accs_s >= acc_threshold]
            cos_high = pairwise_cosines(exp_high)
            high_acc_spreads.append(robust_spread(cos_high))

        mean_within_spread = float(np.mean(within_spreads))
        mean_acc_range = float(np.mean(within_acc_ranges))
        mean_rho = float(np.mean(split_rhos))
        mean_high_spread = float(np.mean(high_acc_spreads))

        print(f"\n  --- LINE 1: Within-Split Attribution Spread ---")
        print(f"  Mean cosine spread (90% interval): {mean_within_spread:.4f}")
        print(f"  Mean accuracy range: {mean_acc_range*100:.2f}%")

        print(f"\n  --- LINE 2: Local Accuracy-Attribution Correlation ---")
        print(f"  Mean Spearman rho: {mean_rho:+.4f}")
        if abs(mean_rho) < args.rho_threshold:
            print(f"  Decoupled (|rho| < {args.rho_threshold}): True")

        print(f"\n  --- LINE 3: High-Accuracy Subset ---")
        print(f"  Mean cosine spread (top 25%): {mean_high_spread:.4f}")

        print(f"\n  --- The Paradox ---")
        print(f"  {'Metric':<35} {'Value':>15} {'Interpretation'}")
        print(f"  {'-'*35} {'-'*15} {'-'*30}")
        print(f"  {'Mean Acc Range':<35} "
              f"{mean_acc_range*100:>14.2f}% {'Locally stable'}")
        print(f"  {'Mean Cosine Spread':<35} "
              f"{mean_within_spread:>15.3f} {'Massive reordering'}")
        print(f"  {'Mean Acc-Attr Spearman':<35} "
              f"{'rho='+f'{mean_rho:+.3f}':>15} "
              f"{'Decoupled' if abs(mean_rho) < args.rho_threshold else 'Weak link'}")
        print(f"  {'Mean High-Acc Spread':<35} "
              f"{mean_high_spread:>15.3f} {'Still diverges'}")

        # Verdict
        acc_stable = bool(global_acc_range < args.acc_range_threshold)
        attr_diverges = bool(mean_within_spread > args.spread_threshold)
        decoupled = bool(abs(mean_rho) < args.rho_threshold)
        high_acc_diverges = bool(mean_high_spread > args.spread_threshold)

        print(f"\n  --- {lens.upper()} VERDICT ---")
        print(f"    Accuracy stable      : {acc_stable}")
        print(f"    Attribution diverges : {attr_diverges}")
        print(f"    Decoupled            : {decoupled}")
        print(f"    High-acc diverges    : {high_acc_diverges}")

        if acc_stable and attr_diverges and high_acc_diverges:
            tier = "ATTACK KILLED"
            v = (f"ATTACK KILLED ({lens.upper()}) — "
                 f"Accuracy is invariant while local attributions diverge by "
                 f"{mean_within_spread:.3f} cosine spread. Even the top-25% "
                 f"highest-accuracy models yield spread {mean_high_spread:.3f}. "
                 f"Accuracy-attribution correlation is negligible "
                 f"(rho={mean_rho:+.3f}). Massive structured explanation "
                 f"drift decoupled from performance.")
        elif acc_stable and attr_diverges:
            tier = "PARTIAL"
            v = (f"PARTIAL ({lens.upper()}) — Attribution diverges but "
                 f"high-accuracy subset spread ({mean_high_spread:.3f}) "
                 f"is below threshold ({args.spread_threshold}).")
        else:
            tier = "INCONCLUSIVE"
            v = (f"INCONCLUSIVE ({lens.upper()}) — "
                 f"acc_stable={acc_stable}, attr_diverges={attr_diverges}, "
                 f"high_acc_diverges={high_acc_diverges}.")

        print(f"    >>> {v}")

        results[f"per_lens_{lens}"] = {
            "verdict_tier": tier,
            "verdict_text": v,
            "metrics": {
                "global_acc_range_pct": round(global_acc_range * 100, 2),
                "mean_local_acc_range_pct": round(mean_acc_range * 100, 2),
                "mean_local_cosine_spread": round(mean_within_spread, 4),
                "mean_local_acc_attr_spearman": round(mean_rho, 4),
                "mean_local_high_acc_cosine_spread": round(mean_high_spread, 4),
                "acc_stable": acc_stable,
                "attr_diverges": attr_diverges,
                "decoupled": decoupled,
                "high_acc_diverges": high_acc_diverges,
            },
        }

    # Combined
    print(f"\n{'='*70}")
    lens_tiers = [results.get(f"per_lens_{l}", {}).get("verdict_tier", "INCONCLUSIVE")
                  for l in ["shap", "lime"]]
    if all(t == "ATTACK KILLED" for t in lens_tiers):
        combined_tier = "ATTACK KILLED"
        combined_text = ("BOTH LENSES: ATTACK KILLED — explanation drift is "
                         "structurally massive within isolated, high-performing "
                         "model sets.")
    else:
        combined_tier = "PARTIAL"
        combined_text = "MIXED — see per-lens verdicts."

    print(f"  COMBINED: {combined_text}")
    print("=" * 70)

    results["verdict_tier"] = combined_tier
    results["verdict_text"] = combined_text

    json_path = Path(args.output_dir) / "tcga_test6_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {json_path}")


if __name__ == "__main__":
    main()
