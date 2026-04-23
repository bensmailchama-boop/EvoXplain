#!/usr/bin/env python3
"""
tcga_test5_ranking_functional.py
=================================
EvoXplain Falsification Test 5 — "Multiplicity is only in the attribution
layer, not in the actual model predictions."
TCGA Tumour vs Normal, LR C-grid, DUAL LENS (SHAP + LIME).

Version: v2 (standardised)
Last revised: April 2026

Attack
------
The basin structure only appears because SHAP/LIME transform the model
output in some post-hoc way. The model's raw predictions (probabilities,
logits) are actually the same across basins.

Defence Strategy
----------------
1. PREDICTION DISAGREEMENT: Fraction of test samples where hard predictions
   differ between runs in different attribution basins vs same basin.
2. BOUNDARY MAD: Mean absolute probability difference on uncertain samples
   (probability in [0.3, 0.7] for either run).
3. SPEARMAN RANK CORRELATION: Of probability vectors between runs.
4. PROBABILITY-SPACE CLUSTERING: Cluster probability vectors and compare
   to attribution basin labels via Adjusted Rand Index.

Kill Condition
--------------
  Cross-basin disagreement ratio >= threshold AND Mann-Whitney p < 0.01
  AND probability-space clustering recovers attribution basins (ARI > threshold)
  → ATTACK KILLED
  Disagreement confirmed but ARI fails → PARTIAL
  Neither passes → INCONCLUSIVE

Data
----
Reads frozen aggregates (probs_test required). No retraining.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from collections import Counter
from itertools import combinations


# =====================================================================
# STANDARD UTILITIES
# =====================================================================

def cosine_sim(a, b):
    return float(1.0 - cosine(a, b))


def normalise(vecs):
    vecs = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms > 0, norms, 1.0)


def pick_k_kmeans(vecs_normed, k_max=5, seed=0,
                  silhouette_threshold=0.15,
                  cosine_collapse_threshold=0.99):
    """Standard KMeans with silhouette threshold and cosine-collapse guard."""
    n = len(vecs_normed)
    if n < 4:
        return 1, np.zeros(n, dtype=int), vecs_normed.mean(axis=0, keepdims=True)

    sims = []
    for i in range(min(n, 200)):
        for j in range(i + 1, min(n, 200)):
            sims.append(cosine_sim(vecs_normed[i], vecs_normed[j]))
    if sims and np.min(sims) >= cosine_collapse_threshold:
        return 1, np.zeros(n, dtype=int), vecs_normed.mean(axis=0, keepdims=True)

    best_k, best_score = 1, 0.0
    best_labels = np.zeros(n, dtype=int)
    best_centers = vecs_normed.mean(axis=0, keepdims=True)

    for k in range(2, min(k_max + 1, n)):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(vecs_normed)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(vecs_normed, labels)
        if score > best_score and score >= silhouette_threshold:
            best_score = score
            best_k = k
            best_labels = labels
            best_centers = np.array([
                vecs_normed[labels == i].mean(axis=0) for i in range(k)
            ])
    return best_k, best_labels, best_centers


# =====================================================================
# PROBABILITY DIVERGENCE METRICS
# =====================================================================

def prediction_disagreement_rate(p_i, p_j):
    """Fraction of test samples where hard predictions differ."""
    return float(np.mean((p_i >= 0.5) != (p_j >= 0.5)))


def boundary_mad(p_i, p_j, low=0.3, high=0.7):
    """Mean absolute probability difference on uncertain samples."""
    mask = ((p_i >= low) & (p_i <= high)) | ((p_j >= low) & (p_j <= high))
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs(p_i[mask] - p_j[mask])))


def prob_spearman(p_i, p_j):
    """Spearman rank correlation of probability vectors."""
    rho, _ = spearmanr(p_i, p_j)
    return float(rho) if not np.isnan(rho) else 1.0


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoXplain TCGA Test 5: Ranking Functional Culprit")
    parser.add_argument("--output_dir_agg", type=str,
                        default="results/final_freeze/tcga_lr_cgrid_shap_lime_100x100")
    parser.add_argument("--split_start", type=int, default=800)
    parser.add_argument("--split_end",   type=int, default=900)
    parser.add_argument("--output_dir",  type=str, default=".")
    parser.add_argument("--ari_threshold", type=float, default=0.3,
                        help="Min ARI for prob-space clustering to 'match'.")
    parser.add_argument("--disagreement_ratio_threshold", type=float, default=2.0,
                        help="Min ratio cross/within disagreement rate.")
    parser.add_argument("--mann_whitney_p_threshold", type=float, default=0.01,
                        help="Max p-value for Mann-Whitney U test.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EvoXplain — Test 5: Ranking Functional Culprit (TCGA LR-Cgrid)")
    print("  'Is multiplicity in the model or just in the attribution layer?'")
    print("=" * 70)
    print(f"  Aggregates             : {args.output_dir_agg}")
    print(f"  Splits                 : {args.split_start}..{args.split_end}")
    print(f"  ARI threshold          : {args.ari_threshold}")
    print(f"  Disagreement ratio     : {args.disagreement_ratio_threshold}")
    print(f"  Mann-Whitney p         : {args.mann_whitney_p_threshold}")

    # ==================================================================
    # LOAD
    # ==================================================================
    print(f"\n[Loading] {args.output_dir_agg}")

    split_records = []
    n_loaded = 0

    for seed in range(args.split_start, args.split_end):
        agg = (Path(args.output_dir_agg) / f"split{seed}"
               / f"aggregate_split{seed}.npz")
        if not agg.exists():
            continue
        data = np.load(agg, allow_pickle=True)

        probs = data["probs_test"] if "probs_test" in data else None
        accs = data["test_acc"]
        c_vals = data["run_C_values"] if "run_C_values" in data else None

        rec = {"seed": seed, "probs": probs, "accs": accs, "c_vals": c_vals}
        for lens in ["shap", "lime"]:
            k_key = f"k_star_{lens}"
            lab_key = f"cluster_labels_{lens}"
            rec[f"{lens}_k"] = (int(data[k_key]) if k_key in data
                                else int(data["k_star"]))
            rec[f"{lens}_labels"] = (data[lab_key] if lab_key in data
                                     else data["cluster_labels"])

        split_records.append(rec)
        n_loaded += 1

    print(f"[Loaded] {n_loaded} splits")

    has_probs = any(r["probs"] is not None and r["probs"].ndim == 2
                    for r in split_records)
    if not has_probs:
        print("[ERROR] No 2D probability arrays found. Cannot proceed.")
        sys.exit(1)

    results = {
        "test_id": "test5_ranking_functional",
        "version": "v2",
        "config": vars(args),
        "n_splits": n_loaded,
    }

    # ==================================================================
    # LINE 1: MATHEMATICAL CONTEXT
    # ==================================================================
    print(f"\n{'='*70}")
    print("LINE 1: Mathematical Context (LR + SHAP LinearExplainer)")
    print("=" * 70)
    print("  For Logistic Regression:")
    print("    P(y=1|x) = sigmoid(w·x + b)")
    print("    SHAP attribution: phi_j = w_j × (x_j − mean(X_train)_j)")
    print()
    print("  Attribution IS a linear transform of the coefficient vector.")
    print("  If attributions diverge → coefficients diverge → probabilities")
    print("  diverge. This test provides EMPIRICAL CONFIRMATION.")

    # ==================================================================
    # LINE 2: PROBABILITY DIVERGENCE (per lens)
    # ==================================================================
    print(f"\n{'='*70}")
    print("LINE 2: Probability Divergence Between Basins")
    print("=" * 70)

    per_lens_results = {}

    for lens in ["shap", "lime"]:
        print(f"\n  --- {lens.upper()} ---")

        within_disagree = []
        cross_disagree = []
        within_boundary_mad = []
        cross_boundary_mad = []
        within_spearman = []
        cross_spearman = []
        ari_scores = []
        prob_k_stars = []
        valid_splits = 0

        for rec in split_records:
            probs = rec["probs"]
            if probs is None or probs.ndim != 2:
                continue
            labels = np.asarray(rec[f"{lens}_labels"], dtype=int)
            k = rec[f"{lens}_k"]
            if k < 2:
                continue

            n_runs = probs.shape[0]
            valid_splits += 1

            if n_runs > 50:
                rng = np.random.RandomState(rec["seed"])
                idx = rng.choice(n_runs, 50, replace=False)
            else:
                idx = np.arange(n_runs)

            for a_pos in range(len(idx)):
                for b_pos in range(a_pos + 1, len(idx)):
                    i, j = idx[a_pos], idx[b_pos]
                    dr = prediction_disagreement_rate(probs[i], probs[j])
                    bm = boundary_mad(probs[i], probs[j])
                    sr = prob_spearman(probs[i], probs[j])

                    if labels[i] == labels[j]:
                        within_disagree.append(dr)
                        if bm is not None:
                            within_boundary_mad.append(bm)
                        within_spearman.append(sr)
                    else:
                        cross_disagree.append(dr)
                        if bm is not None:
                            cross_boundary_mad.append(bm)
                        cross_spearman.append(sr)

            probs_n = normalise(probs)
            prob_k, prob_labels, _ = pick_k_kmeans(probs_n, k_max=5, seed=0)
            ari = adjusted_rand_score(labels, prob_labels)
            ari_scores.append(ari)
            prob_k_stars.append(prob_k)

        if valid_splits == 0:
            print(f"    No valid splits with k>=2 for {lens.upper()}")
            per_lens_results[lens] = {"has_valid_data": False}
            continue

        w_dr = float(np.mean(within_disagree)) if within_disagree else 0.0
        c_dr = float(np.mean(cross_disagree)) if cross_disagree else 0.0
        w_bm = float(np.mean(within_boundary_mad)) if within_boundary_mad else 0.0
        c_bm = float(np.mean(cross_boundary_mad)) if cross_boundary_mad else 0.0
        w_sr = float(np.mean(within_spearman)) if within_spearman else 1.0
        c_sr = float(np.mean(cross_spearman)) if cross_spearman else 1.0
        mean_ari = float(np.mean(ari_scores))

        dr_ratio = (c_dr / w_dr) if w_dr > 1e-9 else (float("inf") if c_dr > 0 else 1.0)

        if len(within_disagree) > 10 and len(cross_disagree) > 10:
            u_stat, u_p = mannwhitneyu(
                cross_disagree, within_disagree, alternative="greater")
        else:
            u_stat, u_p = float("nan"), 1.0

        print(f"\n    Valid splits processed: {valid_splits}")
        print(f"    Pairs: {len(within_disagree)} within-basin, "
              f"{len(cross_disagree)} cross-basin")

        print(f"\n    Prediction disagreement rate:")
        print(f"      Within-basin mean : {w_dr:.6f}")
        print(f"      Cross-basin mean  : {c_dr:.6f}")
        print(f"      Ratio cross/within: {dr_ratio:.2f}x")
        print(f"      Mann-Whitney U: U={u_stat:.0f}  p={u_p:.2e}")

        print(f"\n    Boundary MAD (uncertain samples):")
        print(f"      Within-basin mean : {w_bm:.6f}")
        print(f"      Cross-basin mean  : {c_bm:.6f}")

        print(f"\n    Spearman rank correlation:")
        print(f"      Within-basin mean : {w_sr:.6f}")
        print(f"      Cross-basin mean  : {c_sr:.6f}")

        print(f"\n    Probability-space clustering:")
        print(f"      Prob-space k* distribution: {dict(Counter(prob_k_stars))}")
        print(f"      Mean ARI: {mean_ari:.4f}")
        print(f"      ARI > {args.ari_threshold}: {mean_ari > args.ari_threshold}")

        diverges = bool(
            dr_ratio >= args.disagreement_ratio_threshold
            and u_p < args.mann_whitney_p_threshold
        )
        clusters_match = bool(mean_ari > args.ari_threshold)

        per_lens_results[lens] = {
            "has_valid_data":         True,
            "valid_splits":           valid_splits,
            "n_within_pairs":         len(within_disagree),
            "n_cross_pairs":          len(cross_disagree),
            "within_disagree_mean":   w_dr,
            "cross_disagree_mean":    c_dr,
            "disagreement_ratio":     dr_ratio,
            "mann_whitney_u":         float(u_stat) if not np.isnan(u_stat) else None,
            "mann_whitney_p":         float(u_p),
            "within_boundary_mad":    w_bm,
            "cross_boundary_mad":     c_bm,
            "within_spearman_mean":   w_sr,
            "cross_spearman_mean":    c_sr,
            "mean_ari":               mean_ari,
            "prob_k_star_dist":       dict(Counter(prob_k_stars)),
            "diverges":               diverges,
            "clusters_match":         clusters_match,
        }

    # ==================================================================
    # VERDICT
    # ==================================================================
    print(f"\n{'='*70}")
    print("TEST 5 VERDICT")
    print("=" * 70)

    for lens in ["shap", "lime"]:
        plr = per_lens_results[lens]
        print(f"\n  [{lens.upper()}]")

        if not plr.get("has_valid_data", False):
            tier = "INCONCLUSIVE"
            v = (f"INCONCLUSIVE ({lens.upper()}) — No valid splits with k>=2. "
                 f"Cannot test probability divergence.")
        else:
            print(f"    Prob diverges (ratio>={args.disagreement_ratio_threshold} "
                  f"& p<{args.mann_whitney_p_threshold})     : {plr['diverges']}")
            print(f"    Prob-space clustering matches (ARI>{args.ari_threshold}) : "
                  f"{plr['clusters_match']}")

            if plr["diverges"] and plr["clusters_match"]:
                tier = "ATTACK KILLED"
                v = (
                    f"ATTACK KILLED ({lens.upper()}) — "
                    f"Cross-basin prediction disagreement rate "
                    f"({plr['cross_disagree_mean']:.6f}) is "
                    f"{plr['disagreement_ratio']:.1f}x higher than within-basin "
                    f"({plr['within_disagree_mean']:.6f}), "
                    f"p={plr['mann_whitney_p']:.2e}. "
                    f"Boundary MAD: cross={plr['cross_boundary_mad']:.6f} vs "
                    f"within={plr['within_boundary_mad']:.6f}. "
                    f"Prob-space clustering recovers attribution basins "
                    f"(ARI={plr['mean_ari']:.3f}). "
                    f"Multiplicity is in the MODEL COEFFICIENTS, "
                    f"not in the attribution layer."
                )
            elif plr["diverges"] and not plr["clusters_match"]:
                tier = "PARTIAL"
                v = (
                    f"PARTIAL ({lens.upper()}) — "
                    f"Probability divergence confirmed "
                    f"(ratio={plr['disagreement_ratio']:.1f}x, "
                    f"p={plr['mann_whitney_p']:.2e}), "
                    f"but prob-space clustering does not recover "
                    f"attribution basins (ARI={plr['mean_ari']:.3f} "
                    f"< {args.ari_threshold}). Models differ, but "
                    f"the structure is not isomorphic to attribution basins."
                )
            elif not plr["diverges"] and plr["clusters_match"]:
                tier = "PARTIAL"
                v = (
                    f"PARTIAL ({lens.upper()}) — "
                    f"Prob-space clustering recovers attribution basins "
                    f"(ARI={plr['mean_ari']:.3f}), but prediction "
                    f"disagreement ratio ({plr['disagreement_ratio']:.1f}x) "
                    f"does not reach the {args.disagreement_ratio_threshold}x "
                    f"threshold or Mann-Whitney p is not significant."
                )
            else:
                tier = "INCONCLUSIVE"
                v = (
                    f"INCONCLUSIVE ({lens.upper()}) — "
                    f"Prediction disagreement ratio "
                    f"({plr['disagreement_ratio']:.1f}x) and "
                    f"ARI ({plr['mean_ari']:.3f}) both below thresholds."
                )

        print(f"    >>> {v}")
        results[f"per_lens_{lens}"] = {"verdict_tier": tier, "verdict_text": v,
                                        "metrics": plr}

    # Combined
    print(f"\n{'='*70}")
    lens_tiers = [results.get(f"per_lens_{l}", {}).get("verdict_tier", "INCONCLUSIVE")
                  for l in ["shap", "lime"]]
    if all(t == "ATTACK KILLED" for t in lens_tiers):
        combined_tier = "ATTACK KILLED"
        combined_text = ("BOTH LENSES: ATTACK KILLED — multiplicity is empirically "
                         "confirmed in the model predictions, not just the "
                         "attribution layer.")
    elif any(t == "ATTACK KILLED" for t in lens_tiers):
        combined_tier = "STRONG"
        combined_text = "MIXED — one lens fully killed; see per-lens verdicts."
    elif any(t == "PARTIAL" for t in lens_tiers):
        combined_tier = "PARTIAL"
        combined_text = "PARTIAL — divergence confirmed but clustering limited."
    else:
        combined_tier = "INCONCLUSIVE"
        combined_text = "INCONCLUSIVE — see per-lens verdicts."

    print(f"  COMBINED: {combined_text}")
    print("=" * 70)

    results["verdict_tier"] = combined_tier
    results["verdict_text"] = combined_text

    json_path = Path(args.output_dir) / "tcga_test5_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {json_path}")


if __name__ == "__main__":
    main()
