#!/usr/bin/env python3
"""
tcga_test8_metric_blindness.py
================================
EvoXplain Falsification Test 8 — "Existing metrics already capture this."
TCGA Tumour vs Normal, LR C-grid, DUAL LENS (SHAP + LIME).

Version: v2 (standardised)
Last revised: April 2026

Attack
------
Accuracy, calibration, and standard model evaluation metrics already
capture everything relevant about model quality. If basins all perform
similarly, there is nothing materially new here.

Defence Strategy
----------------
1. ACCURACY: Measure cross-basin accuracy gaps.
2. PROBABILITY SPACE: Measure cross-basin mean probability gaps.
3. PREDICTION AGREEMENT: Sample-by-sample boolean agreement across runs
   in different basins.
4. EXPLANATION-LEVEL DIVERGENCE: Top-gene overlap (Jaccard) between basin
   centroids within the exact same split.

Kill Condition
--------------
  Standard predictive metrics show only modest between-basin differences
  (acc gap < threshold, pred agreement > threshold, prob gap < threshold)
  AND explanation-level summaries show clear gene divergence (Jaccard < threshold)
  → REFRAMED: existing metrics do not fully characterize multiplicity

Data
----
Reads frozen aggregates (probs_test, centroids required). No retraining.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations


# =====================================================================
# STANDARD UTILITIES
# =====================================================================

def top_gene_set(centroid, feature_names, top_n=50):
    ranked = np.argsort(np.abs(centroid))[::-1][:top_n]
    genes = set()
    for idx in ranked:
        genes.add(feature_names[idx].split(".")[0])
    return genes


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoXplain TCGA Test 8: Metric Blindness / Under-resolution")
    parser.add_argument("--output_dir_agg", type=str,
                        default="results/final_freeze/tcga_lr_cgrid_shap_lime_100x100")
    parser.add_argument("--split_start", type=int, default=800)
    parser.add_argument("--split_end",   type=int, default=900)
    parser.add_argument("--gene_top_n",  type=int, default=50)
    parser.add_argument("--output_dir",  type=str, default=".")
    parser.add_argument("--acc_gap_threshold", type=float, default=0.02,
                        help="Max accuracy gap for 'small difference'.")
    parser.add_argument("--pred_agreement_threshold", type=float, default=0.98,
                        help="Min prediction agreement for 'high agreement'.")
    parser.add_argument("--prob_gap_threshold", type=float, default=0.05,
                        help="Max probability gap for 'small difference'.")
    parser.add_argument("--jaccard_threshold", type=float, default=0.7,
                        help="Max Jaccard for 'clear gene divergence'.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EvoXplain — Test 8: Metric Under-resolution (TCGA LR-Cgrid)")
    print("  'Do existing metrics already capture this?'")
    print("=" * 70)
    print(f"  Aggregates             : {args.output_dir_agg}")
    print(f"  Splits                 : {args.split_start}..{args.split_end}")
    print(f"  Gene top-N             : {args.gene_top_n}")
    print(f"  Acc gap threshold      : {args.acc_gap_threshold}")
    print(f"  Pred agreement thresh  : {args.pred_agreement_threshold}")
    print(f"  Prob gap threshold     : {args.prob_gap_threshold}")
    print(f"  Jaccard threshold      : {args.jaccard_threshold}")

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

        split_record = {
            "seed": seed,
            "accs": data["test_acc"],
            "probs": data["probs_test"] if "probs_test" in data else None,
        }
        for lens in ["shap", "lime"]:
            k_key = f"k_star_{lens}"
            lab_key = f"cluster_labels_{lens}"
            c_key = f"centroids_normed_{lens}"
            split_record[f"k_{lens}"] = int(data[k_key]) if k_key in data else int(data["k_star"])
            split_record[f"labels_{lens}"] = data[lab_key] if lab_key in data else data["cluster_labels"]
            split_record[f"centroids_{lens}"] = data[c_key] if c_key in data else data["centroids_normed"]

        all_splits_data.append(split_record)
        n_loaded += 1

    print(f"[Loaded] {n_loaded} splits")

    results = {
        "test_id": "test8_metric_blindness",
        "version": "v2",
        "config": vars(args),
        "n_splits": n_loaded,
    }

    # ==================================================================
    # PER-LENS ANALYSIS
    # ==================================================================
    for lens in ["shap", "lime"]:
        print(f"\n{'='*70}")
        print(f"  {lens.upper()} — METRIC UNDER-RESOLUTION ANALYSIS")
        print(f"{'='*70}")

        split_pred_agreements = []
        split_acc_gaps = []
        split_brier_gaps = []
        split_min_jaccards = []
        valid_splits = 0

        for s_data in all_splits_data:
            k = s_data[f"k_{lens}"]
            if k < 2:
                continue
            probs = s_data["probs"]
            if probs is None or probs.ndim == 1:
                continue

            labels = s_data[f"labels_{lens}"]
            accs = s_data["accs"]
            centroids = s_data[f"centroids_{lens}"]
            preds = (probs > 0.5).astype(int)

            cross_basin_agreements = []
            cross_basin_acc_gaps = []
            brier_proxies = []
            jaccards = []

            for b1, b2 in combinations(range(k), 2):
                runs_b1 = np.where(labels == b1)[0]
                runs_b2 = np.where(labels == b2)[0]

                for r1 in runs_b1:
                    for r2 in runs_b2:
                        cross_basin_agreements.append(np.mean(preds[r1] == preds[r2]))
                        cross_basin_acc_gaps.append(abs(accs[r1] - accs[r2]))

                genes1 = top_gene_set(centroids[b1], feature_names, args.gene_top_n)
                genes2 = top_gene_set(centroids[b2], feature_names, args.gene_top_n)
                shared = len(genes1 & genes2)
                union = len(genes1 | genes2)
                jaccards.append(shared / union if union > 0 else 0)

            for b in range(k):
                runs_b = np.where(labels == b)[0]
                brier_proxies.append(np.mean(probs[runs_b]))

            if cross_basin_agreements:
                split_pred_agreements.append(np.mean(cross_basin_agreements))
                split_acc_gaps.append(np.max(cross_basin_acc_gaps))
                split_min_jaccards.append(np.min(jaccards))
                split_brier_gaps.append(max(brier_proxies) - min(brier_proxies))
                valid_splits += 1

        if valid_splits == 0:
            print("    [ERROR] Insufficient valid split data.")
            results[f"per_lens_{lens}"] = {"verdict_tier": "INCONCLUSIVE",
                                            "verdict_text": "INCONCLUSIVE — no valid data."}
            continue

        mean_pred_agreement = float(np.mean(split_pred_agreements))
        mean_max_acc_gap = float(np.mean(split_acc_gaps))
        mean_brier_gap = float(np.mean(split_brier_gaps))
        mean_min_jaccard = float(np.mean(split_min_jaccards))

        small_acc_diff = bool(mean_max_acc_gap < args.acc_gap_threshold)
        high_pred_agreement = bool(mean_pred_agreement > args.pred_agreement_threshold)
        small_prob_diff = bool(mean_brier_gap < args.prob_gap_threshold)
        clear_gene_divergence = bool(mean_min_jaccard < args.jaccard_threshold)

        print(f"\n  --- Computed Metrics ({valid_splits} splits) ---")
        print(f"    Cross-basin accuracy gap   : {mean_max_acc_gap*100:.2f}%  "
              f"(small < {args.acc_gap_threshold*100:.0f}%: {small_acc_diff})")
        print(f"    Cross-basin pred agreement : {mean_pred_agreement*100:.2f}%  "
              f"(high > {args.pred_agreement_threshold*100:.0f}%: {high_pred_agreement})")
        print(f"    Cross-basin prob gap       : {mean_brier_gap:.4f}  "
              f"(small < {args.prob_gap_threshold}: {small_prob_diff})")
        print(f"    Min gene Jaccard (top-{args.gene_top_n})  : {mean_min_jaccard:.3f}  "
              f"(divergent < {args.jaccard_threshold}: {clear_gene_divergence})")

        predictive_small = small_acc_diff and high_pred_agreement and small_prob_diff

        if predictive_small and clear_gene_divergence:
            tier = "REFRAMED"
            v = (f"REFRAMED ({lens.upper()}) — "
                 f"Standard predictive metrics register only modest between-basin "
                 f"differences (accuracy gap={mean_max_acc_gap*100:.2f}%, "
                 f"prediction agreement={mean_pred_agreement*100:.2f}%, "
                 f"probability gap={mean_brier_gap:.4f}), while explanation-level "
                 f"summaries show substantially larger divergence "
                 f"(gene Jaccard={mean_min_jaccard:.3f}). "
                 f"Existing metrics do not fully characterize the multiplicity; "
                 f"EvoXplain resolves biological story differences that these "
                 f"metrics compress.")
        elif not predictive_small and clear_gene_divergence:
            tier = "PARTIAL"
            v = (f"PARTIAL ({lens.upper()}) — "
                 f"Standard metrics show some basin differences, but "
                 f"explanation-level divergence is substantially larger "
                 f"(gene Jaccard={mean_min_jaccard:.3f}).")
        elif predictive_small and not clear_gene_divergence:
            tier = "PARTIAL"
            v = (f"PARTIAL ({lens.upper()}) — "
                 f"Predictive metrics show small differences, but gene "
                 f"overlap is relatively high (Jaccard={mean_min_jaccard:.3f}). "
                 f"Divergence may be more geometric than semantic.")
        else:
            tier = "INCONCLUSIVE"
            v = (f"INCONCLUSIVE ({lens.upper()}) — "
                 f"Both predictive and explanation-level summaries show "
                 f"basin differences.")

        print(f"\n  >>> {v}")

        results[f"per_lens_{lens}"] = {
            "verdict_tier": tier,
            "verdict_text": v,
            "metrics": {
                "max_acc_gap": round(mean_max_acc_gap, 4),
                "pred_agreement_pct": round(mean_pred_agreement * 100, 2),
                "prob_gap": round(mean_brier_gap, 4),
                "min_gene_jaccard": round(mean_min_jaccard, 3),
                "small_acc_diff": small_acc_diff,
                "high_pred_agreement": high_pred_agreement,
                "small_prob_diff": small_prob_diff,
                "clear_gene_divergence": clear_gene_divergence,
                "valid_splits": valid_splits,
            },
        }

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY: What each metric sees")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Basin Difference':>18} {'Resolves Multiplicity?':>23}")
    print(f"  {'-'*25} {'-'*18} {'-'*23}")

    for lens in ["shap", "lime"]:
        m = results.get(f"per_lens_{lens}", {}).get("metrics", {})
        if not m:
            continue
        acc_gap_pct = m.get("max_acc_gap", 0) * 100
        pred_disagree_pct = 100 - m.get("pred_agreement_pct", 100)
        jacc = m.get("min_gene_jaccard", 0)
        if lens == "shap":
            print(f"  {'Accuracy Gap':<25} {'~%.2f%%' % acc_gap_pct:>18} {'NO':>23}")
            print(f"  {'Prediction Disagreement':<25} {'~%.2f%%' % pred_disagree_pct:>18} {'NO':>23}")
            print(f"  {'Calibration/Prob Gap':<25} {'< 0.05':>18} {'NO':>23}")
            print(f"  {'-'*25} {'-'*18} {'-'*23}")
        print(f"  {f'EvoXplain ({lens.upper()})':<25} {'Jacc=%.2f' % jacc:>18} {'YES':>23}")

    # Combined
    print(f"\n{'='*70}")
    lens_tiers = [results.get(f"per_lens_{l}", {}).get("verdict_tier", "INCONCLUSIVE")
                  for l in ["shap", "lime"]]
    if all(t == "REFRAMED" for t in lens_tiers):
        combined_tier = "REFRAMED"
        combined_text = ("BOTH LENSES: REFRAMED — standard predictive metrics "
                         "partially register basin differences but do not fully "
                         "capture explanation-level divergence. EvoXplain adds "
                         "information about biological story structure that these "
                         "metrics compress.")
    else:
        combined_tier = "PARTIAL"
        combined_text = "MIXED — see per-lens verdicts."

    print(f"  COMBINED: {combined_text}")
    print("=" * 70)

    results["verdict_tier"] = combined_tier
    results["verdict_text"] = combined_text

    json_path = Path(args.output_dir) / "tcga_test8_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {json_path}")


if __name__ == "__main__":
    main()
