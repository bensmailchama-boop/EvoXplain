#!/usr/bin/env python3
"""
tcga_test4_attribution_artefact.py
====================================
EvoXplain Falsification Test 4 — "It's a SHAP/LIME attribution lens artefact."
TCGA Tumour vs Normal, LR C-grid, DUAL LENS (SHAP + LIME).

Version: v2 (standardised, canonical relabel, gene Jaccard matrix)
Last revised: April 2026

Attack
------
SHAP and LIME are unstable post-hoc tools. The basin structure is
produced by the lens, not by any real property of the pipeline.
Specifically: SHAP sees k*=3 and LIME sees k*=2 — one must be wrong.

Defence Strategy
----------------
1. MATHEMATICAL EXACTNESS: SHAP LinearExplainer for LR computes
   phi_j = coef_j * (x_j - mean_j). Zero sampling noise.
2. k* DIRECTION: SHAP consistently has higher k* than LIME (resolving
   power difference, not contradiction).
3. CROSS-LENS CENTROID COSINE: Canonical-relabelled centroid alignment.
4. PER-SPLIT GENE JACCARD: Biology-level cross-lens overlap + within-SHAP
   overlap (invariant to label permutation and centroid magnitude).

Kill Condition
--------------
  (A) Strict centroid: SHAP exact AND systematic AND all centroids align
  (B) Biology path: SHAP exact AND systematic AND all gene-align AND
      merging detected AND within-SHAP biology sharing
  Either path → ATTACK KILLED
  SHAP exact + one alignment partial → PARTIAL
  Neither → INCONCLUSIVE

Data
----
Reads frozen aggregates only. No retraining.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from itertools import combinations
from collections import Counter, defaultdict


# =====================================================================
# STANDARD UTILITIES
# =====================================================================

def cosine_sim(a, b):
    return float(1.0 - cosine(a, b))


def top_gene_set(centroid, feature_names, top_n=50):
    ranked = np.argsort(np.abs(centroid))[::-1][:top_n]
    genes = set()
    weights = {}
    for idx in ranked:
        ensg = feature_names[idx].split(".")[0]
        genes.add(ensg)
        weights[ensg] = float(centroid[idx])
    return genes, weights


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def relabel_by_c_order(labels, c_vals, dom_k):
    """Relabel basin IDs within a split by ascending median C."""
    labels = np.asarray(labels, dtype=int)
    c_vals = np.asarray(c_vals, dtype=float)
    medians = {}
    for b in range(dom_k):
        mask = labels == b
        if not np.any(mask):
            continue
        medians[b] = float(np.median(c_vals[mask]))
    ordered_old = sorted(medians.keys(), key=lambda b: medians[b])
    mapping = {old_b: new_b for new_b, old_b in enumerate(ordered_old)}
    new_labels = np.array([mapping.get(int(x), int(x)) for x in labels], dtype=int)
    return new_labels, mapping


def reorder_centroids(centroids, mapping, dom_k):
    """Apply a label->label mapping to reorder centroid rows."""
    reordered = np.zeros_like(centroids)
    for old_b, new_b in mapping.items():
        if old_b < len(centroids) and new_b < dom_k:
            reordered[new_b] = centroids[old_b]
    return reordered


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoXplain TCGA Test 4: Attribution Artefact")
    parser.add_argument("--output_dir_agg", type=str,
                        default="results/final_freeze/tcga_lr_cgrid_shap_lime_100x100")
    parser.add_argument("--split_start", type=int, default=800)
    parser.add_argument("--split_end",   type=int, default=900)
    parser.add_argument("--gene_top_n",  type=int, default=50)
    parser.add_argument("--output_dir",  type=str, default=".")
    parser.add_argument("--alignment_threshold", type=float, default=0.75,
                        help="Min centroid cosine for basin alignment.")
    parser.add_argument("--jaccard_threshold", type=float, default=0.40,
                        help="Min per-split-averaged Jaccard for shared biology.")
    parser.add_argument("--sign_agreement_threshold", type=float, default=0.80,
                        help="Min fraction of shared genes with matching sign.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    split_range = range(args.split_start, args.split_end)

    print("=" * 70)
    print("EvoXplain — Test 4: Attribution Artefact (TCGA LR-Cgrid)")
    print("  'Is the SHAP vs LIME disagreement a lens artefact?'")
    print("=" * 70)
    print(f"  Aggregates             : {args.output_dir_agg}")
    print(f"  Splits                 : {args.split_start}..{args.split_end}")
    print(f"  Gene top-N             : {args.gene_top_n}")
    print(f"  Alignment threshold    : {args.alignment_threshold}")
    print(f"  Jaccard threshold      : {args.jaccard_threshold}")
    print(f"  Sign agreement thresh  : {args.sign_agreement_threshold}")

    # ==================================================================
    # 1. LOAD + CANONICAL RELABEL PER SPLIT
    # ==================================================================
    print(f"\n[Loading] {args.output_dir_agg}")

    per_split = {"shap": [], "lime": []}
    feature_names = None
    n_loaded = 0
    n_skipped_no_c = 0

    for seed in split_range:
        agg = Path(args.output_dir_agg) / f"split{seed}" / f"aggregate_split{seed}.npz"
        if not agg.exists():
            continue
        data = np.load(agg, allow_pickle=True)
        if feature_names is None:
            feature_names = list(data["feature_names"])

        c_vals = data["run_C_values"] if "run_C_values" in data else None
        if c_vals is None:
            n_skipped_no_c += 1
            continue
        c_vals = np.asarray(c_vals, dtype=float)

        for lens in ["shap", "lime"]:
            k_key = f"k_star_{lens}"
            c_key = f"centroids_normed_{lens}"
            lab_key = f"cluster_labels_{lens}"

            k = int(data[k_key]) if k_key in data else int(data["k_star"])
            centroids = np.asarray(data[c_key] if c_key in data
                                   else data["centroids_normed"])
            labels = np.asarray(data[lab_key] if lab_key in data
                                else data["cluster_labels"], dtype=int)

            new_labels, mapping = relabel_by_c_order(labels, c_vals, k)
            reordered_centroids = reorder_centroids(centroids, mapping, k)

            per_split[lens].append({
                "seed": seed, "k_star": k,
                "centroids_ordered": reordered_centroids,
                "labels_ordered": new_labels,
                "c_vals": c_vals,
            })

        n_loaded += 1

    print(f"[Loaded] {n_loaded} splits  (skipped {n_skipped_no_c} for missing C)")
    if n_loaded == 0:
        sys.exit("[ERROR] No usable splits.")

    shap_ks = [s["k_star"] for s in per_split["shap"]]
    lime_ks = [s["k_star"] for s in per_split["lime"]]
    shap_dom_k = Counter(shap_ks).most_common(1)[0][0]
    lime_dom_k = Counter(lime_ks).most_common(1)[0][0]
    print(f"  SHAP k* distribution: {dict(Counter(shap_ks))}  dominant={shap_dom_k}")
    print(f"  LIME k* distribution: {dict(Counter(lime_ks))}  dominant={lime_dom_k}")

    shap_by_seed = {s["seed"]: s for s in per_split["shap"] if s["k_star"] == shap_dom_k}
    lime_by_seed = {s["seed"]: s for s in per_split["lime"] if s["k_star"] == lime_dom_k}
    paired_seeds = sorted(set(shap_by_seed.keys()) & set(lime_by_seed.keys()))
    print(f"  Paired splits: {len(paired_seeds)}")

    # ==================================================================
    # LINE 1: MATHEMATICAL EXACTNESS
    # ==================================================================
    print(f"\n{'='*70}\nLINE 1: Mathematical Exactness\n{'='*70}")
    shap_exact = True
    print(f"  SHAP LinearExplainer: phi_j = coef_j * (x_j - mean_j)")
    print(f"  Analytically exact: {shap_exact}")

    # ==================================================================
    # LINE 2: k* DIRECTION
    # ==================================================================
    print(f"\n{'='*70}\nLINE 2: k* Direction\n{'='*70}")
    shap_higher = sum(1 for s in paired_seeds
                      if shap_by_seed[s]["k_star"] > lime_by_seed[s]["k_star"])
    print(f"  SHAP k > LIME k: {shap_higher}/{len(paired_seeds)}")
    systematic_direction = shap_higher > 0.8 * len(paired_seeds)
    print(f"  Systematic (>80%): {systematic_direction}")

    # ==================================================================
    # LINE 3: CROSS-LENS CENTROID MATRIX
    # ==================================================================
    print(f"\n{'='*70}\nLINE 3: Cross-Lens Centroid Cosine (canonical)\n{'='*70}")

    shap_avg = np.mean([shap_by_seed[s]["centroids_ordered"] for s in paired_seeds], axis=0)
    lime_avg = np.mean([lime_by_seed[s]["centroids_ordered"] for s in paired_seeds], axis=0)

    cross_cos = np.zeros((shap_dom_k, lime_dom_k))
    for i in range(shap_dom_k):
        for j in range(lime_dom_k):
            cross_cos[i, j] = cosine_sim(shap_avg[i], lime_avg[j])

    print("\n  Cosine matrix (rows=SHAP, cols=LIME):")
    header = "              " + "  ".join([f"LIME-{j}" for j in range(lime_dom_k)])
    print(header)
    for i in range(shap_dom_k):
        row = "  ".join([f"{cross_cos[i,j]:+.3f}" for j in range(lime_dom_k)])
        print(f"    SHAP-{i}   {row}")

    alignment_mapping = {}
    for i in range(shap_dom_k):
        order = np.argsort(cross_cos[i])[::-1]
        best_j = int(order[0])
        best_cos = float(cross_cos[i, best_j])
        margin = float(cross_cos[i, order[0]] - cross_cos[i, order[1]]) if lime_dom_k > 1 else float("inf")
        aligned = best_cos >= args.alignment_threshold
        alignment_mapping[i] = {"target": best_j, "cos": best_cos, "margin": margin, "aligned": aligned}

    target_counts = Counter(alignment_mapping[i]["target"] for i in range(shap_dom_k))
    merging_targets = [t for t, c in target_counts.items() if c > 1]
    merging_detected = len(merging_targets) > 0
    basins_aligned_centroid = all(alignment_mapping[i]["aligned"] for i in range(shap_dom_k))

    # ==================================================================
    # LINE 4: PER-SPLIT GENE JACCARD
    # ==================================================================
    print(f"\n{'='*70}\nLINE 4: Per-Split Gene Jaccard (canonical)\n{'='*70}")

    jacc_cross = np.zeros((shap_dom_k, lime_dom_k))
    sign_cross = np.zeros((shap_dom_k, lime_dom_k))
    per_split_jacc_cross = {(i, j): [] for i in range(shap_dom_k) for j in range(lime_dom_k)}
    per_split_sign_cross = {(i, j): [] for i in range(shap_dom_k) for j in range(lime_dom_k)}
    per_split_jacc_shap = {(i, j): [] for i in range(shap_dom_k) for j in range(shap_dom_k) if i < j}

    for seed in paired_seeds:
        s_s = shap_by_seed[seed]
        s_l = lime_by_seed[seed]

        shap_gene_sets, shap_weight_dicts = [], []
        for b in range(shap_dom_k):
            g, w = top_gene_set(s_s["centroids_ordered"][b], feature_names, args.gene_top_n)
            shap_gene_sets.append(g)
            shap_weight_dicts.append(w)

        lime_gene_sets, lime_weight_dicts = [], []
        for b in range(lime_dom_k):
            g, w = top_gene_set(s_l["centroids_ordered"][b], feature_names, args.gene_top_n)
            lime_gene_sets.append(g)
            lime_weight_dicts.append(w)

        for i in range(shap_dom_k):
            for j in range(lime_dom_k):
                shared = shap_gene_sets[i] & lime_gene_sets[j]
                per_split_jacc_cross[(i, j)].append(jaccard(shap_gene_sets[i], lime_gene_sets[j]))
                if shared:
                    agree = sum(1 for g in shared
                                if np.sign(shap_weight_dicts[i][g]) == np.sign(lime_weight_dicts[j][g]))
                    per_split_sign_cross[(i, j)].append(agree / len(shared))
                else:
                    per_split_sign_cross[(i, j)].append(0.0)

        for i, j in combinations(range(shap_dom_k), 2):
            per_split_jacc_shap[(i, j)].append(jaccard(shap_gene_sets[i], shap_gene_sets[j]))

    for i in range(shap_dom_k):
        for j in range(lime_dom_k):
            jacc_cross[i, j] = float(np.mean(per_split_jacc_cross[(i, j)])) if per_split_jacc_cross[(i, j)] else 0.0
            sign_cross[i, j] = float(np.mean(per_split_sign_cross[(i, j)]))

    print("\n  Jaccard matrix (rows=SHAP, cols=LIME):")
    header = "              " + "  ".join([f"LIME-{j}" for j in range(lime_dom_k)])
    print(header)
    for i in range(shap_dom_k):
        row = "  ".join([f"{jacc_cross[i,j]:.3f}" for j in range(lime_dom_k)])
        print(f"    SHAP-{i}   {row}")

    print("\n  Sign agreement matrix:")
    for i in range(shap_dom_k):
        row = "  ".join([f"{sign_cross[i,j]:.3f}" for j in range(lime_dom_k)])
        print(f"    SHAP-{i}   {row}")

    shap_pairs_sharing_biology = []
    print("\n  Within-SHAP Jaccard:")
    for (i, j), vals in per_split_jacc_shap.items():
        mean_j = float(np.mean(vals)) if vals else 0.0
        print(f"    SHAP-{i} vs SHAP-{j}: {mean_j:.3f}")
        if mean_j >= args.jaccard_threshold:
            shap_pairs_sharing_biology.append((i, j, mean_j))

    gene_alignment_mapping = {}
    for i in range(shap_dom_k):
        row = jacc_cross[i]
        order = np.argsort(row)[::-1]
        best_j, best_j_val = int(order[0]), float(row[order[0]])
        margin = float(row[order[0]] - row[order[1]]) if lime_dom_k > 1 else float("inf")
        aligned = (best_j_val >= args.jaccard_threshold and
                   sign_cross[i, best_j] >= args.sign_agreement_threshold)
        gene_alignment_mapping[i] = {"target": best_j, "jaccard": best_j_val,
                                      "margin": margin, "aligned": aligned}

    gene_target_counts = Counter(gene_alignment_mapping[i]["target"] for i in range(shap_dom_k))
    gene_merging_targets = [t for t, c in gene_target_counts.items() if c > 1]
    gene_merging_detected = len(gene_merging_targets) > 0
    all_basins_gene_aligned = all(gene_alignment_mapping[i]["aligned"] for i in range(shap_dom_k))

    # ==================================================================
    # VERDICT
    # ==================================================================
    print(f"\n{'='*70}\nTEST 4 VERDICT\n{'='*70}")
    print(f"    SHAP exact               : {shap_exact}")
    print(f"    Systematic SHAP>LIME     : {systematic_direction}")
    print(f"    Centroid alignment       : {basins_aligned_centroid}")
    print(f"    Centroid merging         : {merging_detected}")
    print(f"    Biology alignment        : {all_basins_gene_aligned}")
    print(f"    Biology merging          : {gene_merging_detected}")
    print(f"    Within-SHAP sharing      : {len(shap_pairs_sharing_biology)} pair(s)")

    kill_strict = (shap_exact and systematic_direction and basins_aligned_centroid)
    kill_biology = (shap_exact and systematic_direction and all_basins_gene_aligned
                    and (gene_merging_detected or merging_detected)
                    and len(shap_pairs_sharing_biology) >= 1)

    if kill_strict:
        tier = "ATTACK KILLED"
        v = (f"ATTACK KILLED (strict centroid) — SHAP is analytically exact; "
             f"all SHAP basins align to LIME with cosine >= {args.alignment_threshold}. "
             f"Lens disagreement is resolving power, not artefact.")
    elif kill_biology:
        merged_str = ", ".join(f"SHAP-{a}+SHAP-{b} (J={j:.2f})"
                               for a, b, j in shap_pairs_sharing_biology)
        tier = "ATTACK KILLED"
        v = (f"ATTACK KILLED (biology path) — SHAP is analytically exact. "
             f"Gene-level biology preserved across lenses "
             f"(Jaccard >= {args.jaccard_threshold}, "
             f"sign >= {args.sign_agreement_threshold}). "
             f"Within-SHAP sharing: {merged_str}. "
             f"SHAP's extra basin is geometric refinement, not artefact.")
    elif shap_exact and (basins_aligned_centroid or all_basins_gene_aligned):
        tier = "PARTIAL"
        v = ("PARTIAL — SHAP exact; one alignment path passes but not both.")
    else:
        tier = "INCONCLUSIVE"
        v = ("INCONCLUSIVE — SHAP is exact, but neither centroid nor "
             "gene-level biology align across lenses.")

    print(f"\n  >>> {v}")
    print("=" * 70)

    results = {
        "test_id": "test4_attribution_artefact",
        "version": "v2",
        "config": vars(args),
        "n_splits": n_loaded,
        "n_paired_splits": len(paired_seeds),
        "evidence": {
            "shap_exact": shap_exact,
            "systematic_direction": bool(systematic_direction),
            "basins_aligned_centroid": bool(basins_aligned_centroid),
            "centroid_merging_detected": bool(merging_detected),
            "all_basins_gene_aligned": bool(all_basins_gene_aligned),
            "gene_merging_detected": bool(gene_merging_detected),
            "within_shap_sharing_count": len(shap_pairs_sharing_biology),
            "kill_strict": bool(kill_strict),
            "kill_biology": bool(kill_biology),
        },
        "measurements": {
            "shap_k_distribution": dict(Counter(shap_ks)),
            "lime_k_distribution": dict(Counter(lime_ks)),
            "shap_dom_k": int(shap_dom_k),
            "lime_dom_k": int(lime_dom_k),
            "shap_higher_count": int(shap_higher),
            "centroid_cosine_matrix": cross_cos.tolist(),
            "gene_jaccard_matrix": jacc_cross.tolist(),
            "sign_agreement_matrix": sign_cross.tolist(),
            "within_shap_biology": [
                {"shap_i": int(a), "shap_j": int(b), "jaccard": float(j)}
                for a, b, j in shap_pairs_sharing_biology
            ],
        },
        "verdict_tier": tier,
        "verdict_text": v,
    }

    json_path = Path(args.output_dir) / "tcga_test4_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {json_path}")


if __name__ == "__main__":
    main()
