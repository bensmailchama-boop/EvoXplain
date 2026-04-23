#!/usr/bin/env python3
"""
tcga_test3_model_instability.py
================================
EvoXplain Falsification Test 3 — "Model instability / optimisation noise."
TCGA Tumour vs Normal, LR C-grid, DUAL LENS (SHAP + LIME).

Version: v3 (standardised, ρ-normalised)
Last revised: April 2026

Attack
------
The basin structure is just hyperparameter sensitivity or optimisation
noise. Different C values produce slightly different solutions, which
is normal for any regularised model.

Defence Strategy
----------------
1. PERFORMANCE EQUIVALENCE: All basins achieve near-identical accuracy.
2. NUMERICAL STABILITY: Weight-vector CV at same C is below threshold
   (if model_weights available in aggregates).
3. C-MONOTONICITY: Basin assignment is monotonic in C within each split,
   using normalised Spearman rho (rho_norm = |rho|/rho_max) to correct
   for the mathematical ceiling on rho for different k/balance.
4. WITHIN-C STABILITY: Same C → same basin within a split.
5. CROSS-BASIN CENTROIDS: Descriptive centroid cosines (not gated).

Kill Condition
--------------
  Performance-equivalent AND C-monotonic (weak) AND within-C stable
  AND (numerically stable OR not tested)
  → ATTACK KILLED (per lens)

Data
----
Reads frozen aggregates. No retraining.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import Counter, defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, ttest_ind


def cosine_sim(a, b):
    return float(1.0 - cosine(a, b))


# ------------------------------------------------------------------
# Monotonicity helpers (split-wise)
# ------------------------------------------------------------------

def relabel_by_c_order(labels, c_vals, dom_k):
    """
    Relabel basin IDs within a split by ascending median C.

    Returns
    -------
    new_labels : np.ndarray
        Labels remapped so that low-C basin -> 0, next -> 1, ...
    mapping : dict
        old_label -> new_label
    basin_stats : dict
        Stats per new label in the remapped space.
    """
    labels = np.asarray(labels)
    c_vals = np.asarray(c_vals, dtype=float)

    old_stats = {}
    for b in range(dom_k):
        mask = labels == b
        if not np.any(mask):
            continue
        bc = c_vals[mask]
        old_stats[b] = {
            "min": float(np.min(bc)),
            "max": float(np.max(bc)),
            "median": float(np.median(bc)),
            "n": int(np.sum(mask)),
        }

    ordered_old = sorted(old_stats.keys(), key=lambda b: old_stats[b]["median"])
    mapping = {old_b: new_b for new_b, old_b in enumerate(ordered_old)}

    new_labels = np.array([mapping[int(x)] for x in labels], dtype=int)

    basin_stats = {}
    for old_b, new_b in mapping.items():
        basin_stats[new_b] = dict(old_stats[old_b])
        basin_stats[new_b]["old_label"] = int(old_b)
        basin_stats[new_b]["new_label"] = int(new_b)

    return new_labels, mapping, basin_stats


def _adjacent_overlap_stats_log10(c_vals, labels, dom_k):
    """
    Compute adjacent-basin overlaps in log10(C) space after labels have
    already been ordered low-C -> high-C.

    Returns a list of adjacent pair stats.
    """
    log_c = np.log10(np.asarray(c_vals, dtype=float))
    labels = np.asarray(labels, dtype=int)

    basin_ranges = {}
    for b in range(dom_k):
        bc = log_c[labels == b]
        if len(bc) == 0:
            continue
        basin_ranges[b] = {
            "min": float(np.min(bc)),
            "max": float(np.max(bc)),
            "median": float(np.median(bc)),
            "n": int(len(bc)),
        }

    pairs = []
    for b in range(dom_k - 1):
        if b not in basin_ranges or (b + 1) not in basin_ranges:
            continue
        lo = basin_ranges[b]
        hi = basin_ranges[b + 1]
        overlap = max(0.0, lo["max"] - hi["min"])
        union_span = max(hi["max"] - lo["min"], 1e-12)
        overlap_fraction = float(overlap / union_span)
        gap = float(hi["min"] - lo["max"])
        pairs.append({
            "basins": (int(b), int(b + 1)),
            "lo_max_log10": float(lo["max"]),
            "hi_min_log10": float(hi["min"]),
            "overlap_log10": float(overlap),
            "overlap_fraction": overlap_fraction,
            "gap_log10": gap,
        })
    return pairs, basin_ranges


def compute_rho_max(ordered_labels, c_vals, dom_k):
    """Compute the maximum achievable Spearman rho for a given label balance.

    For k ordinal labels with proportions (n_0, n_1, ..., n_{k-1}), the
    maximum Spearman rho between a continuous variable and the labels is
    achieved when all label-0 items have the lowest continuous values,
    all label-1 items have the next lowest, etc.

    This ceiling depends on k and the per-label counts. For k=2 at 40/60
    balance, rho_max ≈ 0.851 — which is exactly where LIME was hitting.

    We compute this empirically by constructing the perfect assignment
    and running spearmanr, which is general for any k and any balance.
    """
    labels = np.asarray(ordered_labels, dtype=int)
    c_vals = np.asarray(c_vals, dtype=float)
    n = len(labels)

    # Count per label (in the ordered label space: 0, 1, ..., dom_k-1)
    counts = [int(np.sum(labels == b)) for b in range(dom_k)]

    # Construct perfect labels: first counts[0] get label 0, etc.
    perfect_labels = np.zeros(n, dtype=int)
    pos = 0
    for b in range(dom_k):
        perfect_labels[pos:pos + counts[b]] = b
        pos += counts[b]

    # Sort C values and compute Spearman against perfect assignment
    sorted_c = np.sort(c_vals)
    rho_max, _ = spearmanr(np.log10(sorted_c), perfect_labels)

    # Guard against degenerate cases
    if np.isnan(rho_max) or rho_max <= 0:
        rho_max = 1.0  # conservative fallback

    return float(rho_max)


def check_split_monotonicity(c_vals, labels, dom_k,
                             rho_threshold=0.9,
                             weak_overlap_fraction_threshold=0.10):
    """
    Evaluate monotonicity for a *single split* after relabelling labels by
    ascending median C.

    Uses NORMALISED rho: rho_norm = |rho| / rho_max, where rho_max is the
    maximum achievable Spearman rho for the given label balance and k.
    This corrects for the mathematical ceiling that makes raw rho
    unreachable for certain k/balance combinations (e.g., k=2 at 40/60
    balance has rho_max ≈ 0.851, so raw threshold of 0.9 is impossible).

    STRICT monotonicity:
      - rho_norm >= rho_threshold
      - zero adjacent overlap in log10(C) space

    WEAK monotonicity:
      - same rho_norm condition
      - maximum adjacent overlap fraction <= weak threshold

    Returns (details_dict).
    """
    c_vals = np.asarray(c_vals, dtype=float)
    labels = np.asarray(labels, dtype=int)

    ordered_labels, mapping, basin_stats = relabel_by_c_order(labels, c_vals, dom_k)
    log_c = np.log10(c_vals)
    rho, p_rho = spearmanr(log_c, ordered_labels)

    # Compute rho_max for this label balance and normalise
    rho_max = compute_rho_max(ordered_labels, c_vals, dom_k)
    rho_norm = abs(rho) / rho_max if rho_max > 0 else abs(rho)
    rho_ok = bool(rho_norm >= rho_threshold)

    adj_pairs, basin_ranges_log10 = _adjacent_overlap_stats_log10(
        c_vals, ordered_labels, dom_k
    )

    strict_nonoverlap = all(p["overlap_log10"] <= 1e-12 for p in adj_pairs)
    max_overlap_fraction = max([p["overlap_fraction"] for p in adj_pairs], default=0.0)
    weak_overlap_ok = bool(max_overlap_fraction <= weak_overlap_fraction_threshold)

    strict_pass = bool(rho_ok and strict_nonoverlap)
    weak_pass = bool(rho_ok and weak_overlap_ok)

    sorted_idx = np.argsort(c_vals)
    ordered_along_c = ordered_labels[sorted_idx]
    transitions = int(np.sum(ordered_along_c[1:] != ordered_along_c[:-1]))

    return {
        "ordered_labels": ordered_labels,
        "old_to_new_mapping": {int(k): int(v) for k, v in mapping.items()},
        "spearman_rho": float(rho),
        "spearman_rho_max": float(rho_max),
        "spearman_rho_norm": float(rho_norm),
        "spearman_p": float(p_rho),
        "rho_threshold": float(rho_threshold),
        "rho_passed": rho_ok,
        "strict_nonoverlap": strict_nonoverlap,
        "weak_overlap_fraction_threshold": float(weak_overlap_fraction_threshold),
        "weak_overlap_ok": weak_overlap_ok,
        "max_adjacent_overlap_fraction": float(max_overlap_fraction),
        "adjacent_overlap_stats_log10": adj_pairs,
        "basin_c_stats_raw": {int(b): basin_stats[b] for b in sorted(basin_stats)},
        "basin_c_stats_log10": {int(b): basin_ranges_log10[b] for b in sorted(basin_ranges_log10)},
        "strict_pass": strict_pass,
        "weak_pass": weak_pass,
        "transitions": transitions,
        "min_transitions": dom_k - 1,
    }


# ------------------------------------------------------------------
# Within-C stability: check that same C -> same basin within a split
# ------------------------------------------------------------------

def check_within_c_stability(dom_splits, tol_digits=6):
    """
    For each split, group runs by rounded C value. If any C-group
    contains more than one basin label, within-C stability fails.

    Note: this check uses the split-wise C-ordered labels, not raw labels.

    Returns (stable, details_dict).
    """
    n_violations = 0
    n_groups_checked = 0
    violation_examples = []

    for s in dom_splits:
        if s["c_vals"] is None or "labels_ordered" not in s:
            continue

        c_rounded = np.round(s["c_vals"], decimals=tol_digits)
        unique_c = np.unique(c_rounded)

        for cv in unique_c:
            mask = c_rounded == cv
            labels_at_c = s["labels_ordered"][mask]
            n_groups_checked += 1

            if len(np.unique(labels_at_c)) > 1:
                n_violations += 1
                if len(violation_examples) < 5:
                    violation_examples.append({
                        "seed": int(s["seed"]),
                        "C": float(cv),
                        "labels": [int(x) for x in labels_at_c],
                    })

    stable = (n_violations == 0) and (n_groups_checked > 0)

    details = {
        "stable": stable,
        "n_groups_checked": n_groups_checked,
        "n_violations": n_violations,
        "violation_examples": violation_examples,
    }
    return stable, details


# ------------------------------------------------------------------
# Numerical stability: measure weight-vector variance at same C
# ------------------------------------------------------------------

def check_numerical_stability(dom_splits, cv_threshold=0.01):
    """
    If the frozen aggregates contain raw weight vectors per run
    (key: 'model_weights'), measure the coefficient of variation
    across runs at each unique C value.

    If weight data is unavailable, returns (None, details) so the
    verdict can note the test was not run rather than hardcoding True.

    Returns (stable_or_None, details_dict).
    """
    has_weights = any("model_weights" in s for s in dom_splits)

    if not has_weights:
        return None, {
            "tested": False,
            "reason": "model_weights not available in frozen aggregates. "
                      "Re-run aggregation with --save_weights to enable "
                      "this check. Convexity is assumed from LR theory "
                      "but NOT empirically verified.",
        }

    max_cv = 0.0
    n_c_groups = 0
    high_cv_examples = []

    for s in dom_splits:
        if s["c_vals"] is None or "model_weights" not in s:
            continue

        weights = np.array(s["model_weights"])
        c_rounded = np.round(s["c_vals"], decimals=6)
        unique_c = np.unique(c_rounded)

        for cv_val in unique_c:
            mask = c_rounded == cv_val
            if np.sum(mask) < 2:
                continue

            w_group = weights[mask]
            w_mean = np.mean(w_group, axis=0)
            w_std = np.std(w_group, axis=0)

            safe_mean = np.where(np.abs(w_mean) > 1e-12, w_mean, 1e-12)
            cv_per_feature = np.abs(w_std / safe_mean)
            this_max_cv = float(np.max(cv_per_feature))

            max_cv = max(max_cv, this_max_cv)
            n_c_groups += 1

            if this_max_cv > cv_threshold and len(high_cv_examples) < 5:
                high_cv_examples.append({
                    "seed": int(s["seed"]),
                    "C": float(cv_val),
                    "max_feature_cv": this_max_cv,
                })

    stable = bool(max_cv < cv_threshold) and (n_c_groups > 0)

    return stable, {
        "tested": True,
        "max_cv_across_all": max_cv,
        "cv_threshold": cv_threshold,
        "n_c_groups_tested": n_c_groups,
        "stable": stable,
        "high_cv_examples": high_cv_examples,
    }


# ------------------------------------------------------------------
# Aggregate split-wise monotonicity
# ------------------------------------------------------------------

def summarize_splitwise_monotonicity(dom_splits, dom_k,
                                     rho_threshold=0.9,
                                     weak_overlap_fraction_threshold=0.10):
    per_split = []

    for s in dom_splits:
        if s["c_vals"] is None:
            continue

        details = check_split_monotonicity(
            s["c_vals"], s["labels"], dom_k,
            rho_threshold=rho_threshold,
            weak_overlap_fraction_threshold=weak_overlap_fraction_threshold,
        )
        s["labels_ordered"] = details["ordered_labels"]
        s["old_to_new_mapping"] = details["old_to_new_mapping"]

        per_split.append({
            "seed": int(s["seed"]),
            "spearman_rho": float(details["spearman_rho"]),
            "spearman_rho_max": float(details["spearman_rho_max"]),
            "spearman_rho_norm": float(details["spearman_rho_norm"]),
            "spearman_p": float(details["spearman_p"]),
            "rho_passed": bool(details["rho_passed"]),
            "strict_nonoverlap": bool(details["strict_nonoverlap"]),
            "weak_overlap_ok": bool(details["weak_overlap_ok"]),
            "max_adjacent_overlap_fraction": float(details["max_adjacent_overlap_fraction"]),
            "strict_pass": bool(details["strict_pass"]),
            "weak_pass": bool(details["weak_pass"]),
            "transitions": int(details["transitions"]),
            "min_transitions": int(details["min_transitions"]),
            "old_to_new_mapping": details["old_to_new_mapping"],
            "adjacent_overlap_stats_log10": details["adjacent_overlap_stats_log10"],
            "basin_c_stats_raw": details["basin_c_stats_raw"],
        })

    n = len(per_split)
    if n == 0:
        return False, {
            "tested": False,
            "reason": "no split-wise C information available",
            "n_splits_tested": 0,
            "per_split": [],
        }

    rhos = np.array([x["spearman_rho"] for x in per_split], dtype=float)
    rho_norms = np.array([x["spearman_rho_norm"] for x in per_split], dtype=float)
    rho_maxes = np.array([x["spearman_rho_max"] for x in per_split], dtype=float)
    strict_passes = [x for x in per_split if x["strict_pass"]]
    weak_passes = [x for x in per_split if x["weak_pass"]]

    weak_monotonic = bool(len(weak_passes) == n)
    strict_monotonic = bool(len(strict_passes) == n)

    weak_fail_examples = [
        {
            "seed": x["seed"],
            "rho": x["spearman_rho"],
            "rho_norm": x["spearman_rho_norm"],
            "rho_max": x["spearman_rho_max"],
            "rho_passed": x["rho_passed"],
            "max_adjacent_overlap_fraction": x["max_adjacent_overlap_fraction"],
            "strict_nonoverlap": x["strict_nonoverlap"],
        }
        for x in per_split if not x["weak_pass"]
    ][:5]

    strict_fail_examples = [
        {
            "seed": x["seed"],
            "rho": x["spearman_rho"],
            "rho_norm": x["spearman_rho_norm"],
            "rho_max": x["spearman_rho_max"],
            "max_adjacent_overlap_fraction": x["max_adjacent_overlap_fraction"],
            "strict_nonoverlap": x["strict_nonoverlap"],
        }
        for x in per_split if not x["strict_pass"]
    ][:5]

    details = {
        "tested": True,
        "n_splits_tested": n,
        "rho_threshold": float(rho_threshold),
        "weak_overlap_fraction_threshold": float(weak_overlap_fraction_threshold),
        "median_spearman_rho": float(np.median(rhos)),
        "mean_spearman_rho": float(np.mean(rhos)),
        "min_spearman_rho": float(np.min(rhos)),
        "max_spearman_rho": float(np.max(rhos)),
        "median_spearman_rho_norm": float(np.median(rho_norms)),
        "mean_spearman_rho_norm": float(np.mean(rho_norms)),
        "min_spearman_rho_norm": float(np.min(rho_norms)),
        "max_spearman_rho_norm": float(np.max(rho_norms)),
        "median_rho_max": float(np.median(rho_maxes)),
        "mean_rho_max": float(np.mean(rho_maxes)),
        "strict_pass_count": int(len(strict_passes)),
        "weak_pass_count": int(len(weak_passes)),
        "strict_pass_rate": float(len(strict_passes) / n),
        "weak_pass_rate": float(len(weak_passes) / n),
        "strict_monotonic": strict_monotonic,
        "weak_monotonic": weak_monotonic,
        "weak_fail_examples": weak_fail_examples,
        "strict_fail_examples": strict_fail_examples,
        "per_split": per_split,
    }
    return weak_monotonic, details


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EvoXplain TCGA Falsification Test 3 v3 — "
                    "Model instability check (patched split-wise monotonicity)"
    )
    parser.add_argument("--output_dir_agg", type=str,
                        default="results/final_freeze/tcga_lr_cgrid_shap_lime_100x100")
    parser.add_argument("--split_start", type=int, default=800)
    parser.add_argument("--split_end",   type=int, default=900)
    parser.add_argument("--output_dir",  type=str, default=".")
    parser.add_argument("--rho_threshold", type=float, default=0.9,
                        help="Minimum |Spearman rho| for monotonicity.")
    parser.add_argument("--cv_threshold", type=float, default=0.01,
                        help="Max coefficient of variation for weight stability.")
    parser.add_argument("--perf_gap_threshold", type=float, default=0.01,
                        help="Max accuracy gap for performance equivalence.")
    parser.add_argument("--weak_overlap_fraction_threshold", type=float, default=0.10,
                        help="Max allowed adjacent overlap fraction in log10(C) space for weak monotonicity.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    split_range = range(args.split_start, args.split_end)

    print("=" * 70)
    print("EvoXplain — Test 3: Model Instability (TCGA LR-Cgrid)")
    print("  'Is this just hyperparameter sensitivity / optimisation noise?'")
    print("=" * 70)
    print(f"  Aggregates             : {args.output_dir_agg}")
    print(f"  Splits                 : {args.split_start}..{args.split_end}")
    print(f"  Rho threshold (norm)   : {args.rho_threshold}")
    print(f"  CV threshold           : {args.cv_threshold}")
    print(f"  Perf gap threshold     : {args.perf_gap_threshold}")
    print(f"  Weak overlap threshold : {args.weak_overlap_fraction_threshold}")

    print(f"\n[Loading] {args.output_dir_agg}")

    per_split = {"shap": [], "lime": []}
    feature_names = None
    n_loaded = 0

    for seed in split_range:
        agg = (Path(args.output_dir_agg) / f"split{seed}" /
               f"aggregate_split{seed}.npz")
        if not agg.exists():
            continue
        data = np.load(agg, allow_pickle=True)
        if feature_names is None:
            feature_names = list(data["feature_names"])

        accs = data["test_acc"]
        c_vals = data["run_C_values"] if "run_C_values" in data else None
        weights = data["model_weights"] if "model_weights" in data else None

        for lens in ["shap", "lime"]:
            k_key = f"k_star_{lens}"
            lab_key = f"cluster_labels_{lens}"
            c_key = f"centroids_normed_{lens}"

            k = int(data[k_key]) if k_key in data else int(data["k_star"])
            labels = (data[lab_key] if lab_key in data
                      else data["cluster_labels"])
            centroids = (data[c_key] if c_key in data
                         else data["centroids_normed"])

            entry = {
                "seed": seed, "k_star": k,
                "labels": np.asarray(labels, dtype=int),
                "centroids": np.asarray(centroids),
                "accs": np.asarray(accs, dtype=float),
                "c_vals": None if c_vals is None else np.asarray(c_vals, dtype=float),
            }
            if weights is not None:
                entry["model_weights"] = weights

            per_split[lens].append(entry)
        n_loaded += 1

    print(f"[Loaded] {n_loaded} splits")

    results = {
        "test_id": "test3_model_instability",
        "version": "v3",
        "config": vars(args),
        "n_splits": n_loaded,
    }

    for lens in ["shap", "lime"]:
        print(f"\n{'='*70}")
        print(f"  {lens.upper()} ANALYSIS")
        print(f"{'='*70}")

        dom_k = Counter(s["k_star"] for s in per_split[lens]).most_common(1)[0][0]
        dom_splits = [s for s in per_split[lens] if s["k_star"] == dom_k]
        n_dom = len(dom_splits)
        print(f"  Dominant k*={dom_k} ({n_dom}/{n_loaded} splits)")

        # Precompute ordered labels per split so all downstream analyses use
        # consistent low-C -> high-C basin indexing.
        monotonic, mono_details = summarize_splitwise_monotonicity(
            dom_splits, dom_k,
            rho_threshold=args.rho_threshold,
            weak_overlap_fraction_threshold=args.weak_overlap_fraction_threshold,
        )

        # ==========================================================
        # LINE 1: PERFORMANCE EQUIVALENCE
        # ==========================================================
        print(f"\n  --- LINE 1: Performance Equivalence ---")

        all_basin_accs = defaultdict(list)
        all_basin_c = defaultdict(list)
        global_accs = []

        for s in dom_splits:
            labels_use = s.get("labels_ordered", s["labels"])
            for b in range(dom_k):
                mask = labels_use == b
                if np.sum(mask) == 0:
                    continue
                b_accs = s["accs"][mask]
                all_basin_accs[b].extend(b_accs)
                if s["c_vals"] is not None:
                    all_basin_c[b].extend(s["c_vals"][mask])
            global_accs.extend(s["accs"])

        global_accs = np.array(global_accs)
        print(f"  Global accuracy: {np.mean(global_accs):.4f} +/- {np.std(global_accs):.4f}")

        max_gap = 0.0
        for b in range(dom_k):
            ba = np.array(all_basin_accs[b])
            bc = np.array(all_basin_c[b]) if all_basin_c[b] else None
            c_str = ""
            if bc is not None and len(bc) > 0:
                c_str = (f"  C: mean={np.mean(bc):.4f} "
                         f"[{np.min(bc):.4f}, {np.max(bc):.4f}]")
            print(f"    Basin {b}: acc={np.mean(ba):.4f} +/- {np.std(ba):.4f}"
                  f"  n={len(ba)}{c_str}")

        print(f"\n  Pairwise accuracy comparisons:")
        for i, j in combinations(range(dom_k), 2):
            ai = np.array(all_basin_accs[i])
            aj = np.array(all_basin_accs[j])
            gap = abs(np.mean(ai) - np.mean(aj))
            t, p = ttest_ind(ai, aj, equal_var=False)
            max_gap = max(max_gap, gap)
            print(f"    Basin {i} vs {j}: gap={gap:.4f} (t={t:.2f}, p={p:.2e})")

        perf_equivalent = bool(max_gap < args.perf_gap_threshold)
        print(f"\n  Max accuracy gap: {max_gap:.4f}")
        print(f"  Performance-equivalent (gap < {args.perf_gap_threshold}): {perf_equivalent}")

        # ==========================================================
        # LINE 2: NUMERICAL STABILITY
        # ==========================================================
        print(f"\n  --- LINE 2: Numerical Stability ---")

        num_stable, num_details = check_numerical_stability(
            dom_splits, cv_threshold=args.cv_threshold
        )

        if num_details.get("tested"):
            print(f"  Weight-vector CV test: max_cv = {num_details['max_cv_across_all']:.6f} "
                  f"(threshold = {args.cv_threshold})")
            print(f"  C-groups tested: {num_details['n_c_groups_tested']}")
            print(f"  Numerically stable: {num_stable}")
            if num_details["high_cv_examples"]:
                print("  High-CV examples:")
                for ex in num_details["high_cv_examples"]:
                    print(f"    seed={ex['seed']} C={ex['C']:.4g} max_cv={ex['max_feature_cv']:.4f}")
        else:
            print(f"  WARNING: {num_details['reason']}")
            print("  Numerical stability: NOT TESTED (treated as unknown)")

        # ==========================================================
        # LINE 3: C-MONOTONICITY (patched: split-wise, relabelled)
        # ==========================================================
        print(f"\n  --- LINE 3: C-Monotonicity (split-wise, relabelled, ρ-normalised) ---")
        if mono_details.get("tested"):
            print(f"  Splits tested: {mono_details['n_splits_tested']}")
            print(f"  Median Spearman ρ (raw)       : {mono_details['median_spearman_rho']:+.4f}")
            print(f"  Median ρ_max (label-balance)  : {mono_details['median_rho_max']:+.4f}")
            print(f"  Median ρ_norm (= |ρ|/ρ_max)   : {mono_details['median_spearman_rho_norm']:+.4f}")
            print(f"  Mean   ρ_norm                 : {mono_details['mean_spearman_rho_norm']:+.4f}")
            print(f"  Min/Max ρ_norm: [{mono_details['min_spearman_rho_norm']:+.4f}, {mono_details['max_spearman_rho_norm']:+.4f}]")
            print(f"  Strict pass rate (ρ_norm + zero overlap): {mono_details['strict_pass_count']}/{mono_details['n_splits_tested']} = {mono_details['strict_pass_rate']:.3f}")
            print(f"  Weak   pass rate (ρ_norm + overlap <= {args.weak_overlap_fraction_threshold:.2f}): {mono_details['weak_pass_count']}/{mono_details['n_splits_tested']} = {mono_details['weak_pass_rate']:.3f}")
            print(f"  Monotonic (weak, used for verdict): {monotonic}")
            print(f"  Monotonic (strict, reported only): {mono_details['strict_monotonic']}")
            if mono_details["weak_fail_examples"]:
                print("  Example weak failures:")
                for ex in mono_details["weak_fail_examples"]:
                    print(f"    seed={ex['seed']} ρ={ex['rho']:+.4f} ρ_norm={ex['rho_norm']:+.4f} "
                          f"ρ_max={ex['rho_max']:.4f} rho_passed={ex['rho_passed']} "
                          f"max_adj_overlap_frac={ex['max_adjacent_overlap_fraction']:.4f} "
                          f"strict_nonoverlap={ex['strict_nonoverlap']}")
            elif mono_details["strict_fail_examples"]:
                print("  Example strict-only failures:")
                for ex in mono_details["strict_fail_examples"]:
                    print(f"    seed={ex['seed']} ρ={ex['rho']:+.4f} ρ_norm={ex['rho_norm']:+.4f} "
                          f"ρ_max={ex['rho_max']:.4f} "
                          f"max_adj_overlap_frac={ex['max_adjacent_overlap_fraction']:.4f} "
                          f"strict_nonoverlap={ex['strict_nonoverlap']}")

            print("  Pooled C-range per ordered basin:")
            for b in range(dom_k):
                bc = np.array(all_basin_c[b]) if all_basin_c[b] else None
                if bc is None or len(bc) == 0:
                    continue
                print(f"    Basin {b}: C=[{np.min(bc):.4f}, {np.max(bc):.4f}]  median={np.median(bc):.4f}  n={len(bc)}")
        else:
            monotonic = False
            print(f"  {mono_details.get('reason', 'No C values available')}")

        # ==========================================================
        # LINE 4: WITHIN-C STABILITY
        # ==========================================================
        print(f"\n  --- LINE 4: Within-C Stability ---")

        within_c_stable, wc_details = check_within_c_stability(dom_splits)
        print(f"  Groups checked: {wc_details['n_groups_checked']}")
        print(f"  Violations: {wc_details['n_violations']}")
        if wc_details["violation_examples"]:
            for ex in wc_details["violation_examples"]:
                print(f"    seed={ex['seed']} C={ex['C']:.4g} labels={ex['labels']}")
        print(f"  Within-C stable: {within_c_stable}")

        # ==========================================================
        # LINE 5: CROSS-BASIN CENTROID ANALYSIS
        # ==========================================================
        print(f"\n  --- LINE 5: Cross-Basin Centroid Cosines ---")

        aligned_centroids = []
        for s in dom_splits:
            mapping = s.get("old_to_new_mapping")
            if mapping is None:
                aligned_centroids.append(s["centroids"])
                continue
            reordered = np.zeros_like(s["centroids"])
            for old_b, new_b in mapping.items():
                reordered[new_b] = s["centroids"][old_b]
            aligned_centroids.append(reordered)

        stacked = np.array(aligned_centroids)
        avg_centroids = np.mean(stacked, axis=0)

        centroid_cosines = {}
        for i, j in combinations(range(dom_k), 2):
            c = cosine_sim(avg_centroids[i], avg_centroids[j])
            centroid_cosines[f"{i}_vs_{j}"] = round(c, 4)
            print(f"    Basin {i} vs {j}: cosine = {c:+.4f}")

        # ==========================================================
        # VERDICT
        # ==========================================================
        print(f"\n  --- {lens.upper()} VERDICT ---")

        num_tested = num_details.get("tested", False)
        num_passed = (num_stable is True)

        print(f"    Performance-equivalent   : {perf_equivalent}")
        print(f"    Numerically stable       : {'NOT TESTED' if not num_tested else num_stable}")
        print(f"    C-monotonic (weak main)  : {monotonic}")
        print(f"    C-monotonic (strict aux) : {mono_details.get('strict_monotonic', False)}")
        print(f"    Within-C stable          : {within_c_stable}")

        if (perf_equivalent and monotonic and within_c_stable
                and (num_passed or not num_tested)):
            if num_tested and num_passed:
                stability_clause = (
                    "Solver weight vectors are empirically stable "
                    f"(max CV < {args.cv_threshold})."
                )
            else:
                stability_clause = (
                    "Solver stability not directly measured (no weight data in aggregates); "
                    "convexity assumed from LR theory. Re-run with --save_weights for empirical confirmation."
                )

            v = (
                f"ATTACK KILLED ({lens.upper()}) — "
                f"All basins are performance-equivalent (max gap={max_gap:.4f}). "
                f"Basin assignment is split-wise monotonic in C after relabelling by median C "
                f"(weak pass rate={mono_details['weak_pass_rate']:.3f}, "
                f"median ρ_norm={mono_details['median_spearman_rho_norm']:+.3f} "
                f"[raw ρ={mono_details['median_spearman_rho']:+.3f}, "
                f"ρ_max={mono_details['median_rho_max']:.3f}]; "
                f"strict pass rate={mono_details['strict_pass_rate']:.3f}). "
                f"Within-C runs are stable. {stability_clause} "
                f"This is structured mechanistic multiplicity: a performance-irrelevant hyperparameter "
                f"determines which biological narrative emerges."
            )
        elif perf_equivalent and (monotonic or within_c_stable):
            missing = []
            if not monotonic:
                missing.append("split-wise weak monotonicity failed")
            if not within_c_stable:
                missing.append("within-C stability failed")
            if num_tested and not num_passed:
                missing.append("numerical stability failed")
            v = (
                f"PARTIALLY KILLED ({lens.upper()}) — "
                f"Performance-equivalent (gap={max_gap:.4f}), but: {'; '.join(missing)}. "
                f"Strict monotonicity pass rate={mono_details.get('strict_pass_rate', 0.0):.3f}; "
                f"weak pass rate={mono_details.get('weak_pass_rate', 0.0):.3f}."
            )
        else:
            v = (
                f"INCONCLUSIVE ({lens.upper()}) — "
                f"perf_equiv={perf_equivalent}, "
                f"weak_monotonic={monotonic}, "
                f"within_c_stable={within_c_stable}, "
                f"num_stable={'NOT TESTED' if not num_tested else num_stable}."
            )

        print(f"    >>> {v}")

        # Determine tier
        if "ATTACK KILLED" in v:
            tier = "ATTACK KILLED"
        elif "PARTIALLY KILLED" in v:
            tier = "PARTIAL"
        else:
            tier = "INCONCLUSIVE"

        results[f"per_lens_{lens}"] = {
            "verdict_tier": tier,
            "verdict_text": v,
            "metrics": {
                "perf_equivalent": perf_equivalent,
                "max_acc_gap": round(max_gap, 4),
                "perf_gap_threshold": args.perf_gap_threshold,
                "numerically_stable": num_stable,
                "numerical_stability_tested": num_tested,
                "monotonic_weak": monotonic,
                "monotonic_strict": mono_details.get("strict_monotonic", False),
                "within_c_stable": within_c_stable,
            },
            "basin_accs": {
                str(b): {
                    "mean": round(float(np.mean(all_basin_accs[b])), 4),
                    "std": round(float(np.std(all_basin_accs[b])), 4),
                    "n": len(all_basin_accs[b]),
                }
                for b in range(dom_k)
            },
            "monotonicity_details": mono_details,
            "within_c_details": wc_details,
            "numerical_stability_details": num_details,
            "centroid_cosines": centroid_cosines,
        }
        if all_basin_c[0]:
            results[f"per_lens_{lens}"]["basin_c_ranges"] = {
                str(b): {
                    "min": round(float(np.min(all_basin_c[b])), 4),
                    "max": round(float(np.max(all_basin_c[b])), 4),
                    "mean": round(float(np.mean(all_basin_c[b])), 4),
                    "median": round(float(np.median(all_basin_c[b])), 4),
                }
                for b in range(dom_k) if all_basin_c[b]
            }

    print(f"\n{'='*70}")
    lens_tiers = [results.get(f"per_lens_{l}", {}).get("verdict_tier", "INCONCLUSIVE")
                  for l in ["shap", "lime"]]
    if all(t == "ATTACK KILLED" for t in lens_tiers):
        combined_tier = "ATTACK KILLED"
        combined_text = "BOTH LENSES: ATTACK KILLED — model instability ruled out."
    elif any(t == "ATTACK KILLED" for t in lens_tiers):
        combined_tier = "STRONG"
        combined_text = "MIXED — one lens killed; see per-lens verdicts."
    else:
        combined_tier = "INCONCLUSIVE"
        combined_text = "INCONCLUSIVE — see per-lens verdicts."

    print(f"  COMBINED: {combined_text}")
    print("=" * 70)
    results["verdict_tier"] = combined_tier
    results["verdict_text"] = combined_text

    json_path = Path(args.output_dir) / "tcga_test3_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {json_path}")

    txt_path = Path(args.output_dir) / "tcga_test3_summary.txt"
    with open(txt_path, "w") as f:
        f.write("EvoXplain — TCGA Falsification Test 3\n")
        f.write("Model Instability (LR C-grid, SHAP+LIME, rho-normalised)\n\n")
        for lens in ["shap", "lime"]:
            pl = results.get(f"per_lens_{lens}", {})
            fl = pl.get("metrics", {})
            md = pl.get("monotonicity_details", {})
            f.write(f"--- {lens.upper()} ---\n")
            f.write(f"  Perf-equivalent: {fl.get('perf_equivalent')} (max gap={fl.get('max_acc_gap')})\n")
            f.write(f"  Numerically stable: {'NOT TESTED' if not fl.get('numerical_stability_tested') else fl.get('numerically_stable')}\n")
            f.write(f"  Monotonic weak (main): {fl.get('monotonic_weak')}\n")
            f.write(f"  Monotonic strict: {fl.get('monotonic_strict')}\n")
            if md:
                f.write(f"  Weak pass rate: {md.get('weak_pass_rate')}\n")
                f.write(f"  Strict pass rate: {md.get('strict_pass_rate')}\n")
                f.write(f"  Median rho_norm: {md.get('median_spearman_rho_norm')}\n")
                f.write(f"  Median rho_max: {md.get('median_rho_max')}\n")
            f.write(f"  Within-C stable: {fl.get('within_c_stable')}\n")
            f.write(f"  Verdict: {pl.get('verdict_text')}\n\n")
        f.write(f"Combined: {combined_text}\n")
    print(f"[Saved] {txt_path}")


if __name__ == "__main__":
    main()
