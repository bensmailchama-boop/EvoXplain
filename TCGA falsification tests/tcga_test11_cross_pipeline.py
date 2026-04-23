#!/usr/bin/env python3
"""
tcga_test11_cross_pipeline.py
===============================
EvoXplain Falsification Test 11 — "This is pipeline-specific, not general."
TCGA Tumour vs Normal.

Version: v2 (standardised, variable-shadowing bug fixed)
Last revised: April 2026

Attack
------
Multiplicity is an LR-specific artefact. Other model classes (DNN, XGB)
do not show it. Therefore it is not a general phenomenon.

Defence Strategy
----------------
1. PER-SPLIT k* COMPARISON: Do both LR and DNN find k*>=2 on the
   same dataset splits?
2. CROSS-PIPELINE CENTROID ALIGNMENT: Per-split best-match cosine.
3. CROSS-PIPELINE GENE OVERLAP: Per-split best-match gene Jaccard.

Kill Condition
--------------
  Both pipelines show multiplicity in majority of matching splits
  AND some cross-pipeline overlap (alignment or gene pass rate > 0.25)
  AND evidence level is strong (>= min_splits_strong matching splits)
  → ATTACK SUBSTANTIALLY WEAKENED

Data
----
Reads frozen aggregates from two directories. No retraining.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from collections import Counter
from itertools import combinations


# =====================================================================
# STANDARD UTILITIES
# =====================================================================

def cosine_sim(a, b):
    return float(1.0 - cosine(a, b))


def top_gene_set(centroid, feature_names, top_n=50):
    ranked = np.argsort(np.abs(centroid))[::-1][:top_n]
    return set(str(feature_names[idx]).split(".")[0] for idx in ranked)


def load_split_npz(agg_path, lens="shap"):
    if not agg_path.exists():
        return None
    data = np.load(agg_path, allow_pickle=True)

    k_key = f"k_star_{lens}"
    c_key = f"centroids_normed_{lens}"

    k_star = int(data[k_key]) if k_key in data else (int(data["k_star"]) if "k_star" in data else None)
    if k_star is None:
        return None

    centroids = (np.array(data[c_key], dtype=float) if c_key in data
                 else (np.array(data["centroids_normed"], dtype=float) if "centroids_normed" in data else None))
    if centroids is None:
        return None

    feature_names = list(data["feature_names"]) if "feature_names" in data else None

    return {"k_star": k_star, "centroids": centroids, "feature_names": feature_names}


def get_split_ids(root_dir):
    ids = []
    for p in sorted(Path(root_dir).glob("split*/aggregate_split*.npz")):
        try:
            ids.append(int(p.stem.replace("aggregate_split", "")))
        except ValueError:
            continue
    return sorted(set(ids))


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoXplain TCGA Test 11: Cross-Pipeline Consistency")
    parser.add_argument("--lr_cgrid_dir", type=str,
                        default="results/final_freeze/tcga_lr_cgrid_shap_lime_100x100")
    parser.add_argument("--dnn_shap_dir", type=str,
                        default="results/final_freeze/tcga_dnn_shap_lime_100x100")
    parser.add_argument("--gene_top_n", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--min_splits_strong", type=int, default=20,
                        help="Min matching splits for 'strong' evidence.")
    parser.add_argument("--min_splits_prelim", type=int, default=5,
                        help="Min matching splits for 'preliminary' evidence.")
    parser.add_argument("--alignment_threshold", type=float, default=0.2,
                        help="Min max|cross-cosine| for alignment pass.")
    parser.add_argument("--gene_jaccard_threshold", type=float, default=0.1,
                        help="Min max gene Jaccard for gene pass.")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EvoXplain — Test 11: Cross-Pipeline Consistency (TCGA)")
    print("  'Is multiplicity limited to LR, or does DNN show it too?'")
    print("=" * 70)
    print(f"  LR-Cgrid dir           : {args.lr_cgrid_dir}")
    print(f"  DNN-SHAP dir           : {args.dnn_shap_dir}")
    print(f"  Gene top-N             : {args.gene_top_n}")
    print(f"  Min splits (strong)    : {args.min_splits_strong}")
    print(f"  Alignment threshold    : {args.alignment_threshold}")
    print(f"  Gene Jaccard threshold : {args.gene_jaccard_threshold}")

    lr_ids = get_split_ids(args.lr_cgrid_dir)
    dnn_ids = get_split_ids(args.dnn_shap_dir)
    common_ids = sorted(set(lr_ids) & set(dnn_ids))

    print(f"\n  LR splits: {len(lr_ids)}, DNN splits: {len(dnn_ids)}, "
          f"Common: {len(common_ids)}")

    if not common_ids:
        sys.exit("[ERROR] No matching split IDs.")

    split_records = []
    lr_k_list, dnn_k_list = [], []
    feature_names_ref = None

    print(f"\n{'='*70}\nPER-SPLIT COMPARISON\n{'='*70}")

    for split_id in common_ids:
        lr_path = Path(args.lr_cgrid_dir) / f"split{split_id}" / f"aggregate_split{split_id}.npz"
        dnn_path = Path(args.dnn_shap_dir) / f"split{split_id}" / f"aggregate_split{split_id}.npz"

        lr = load_split_npz(lr_path, lens="shap")
        dnn = load_split_npz(dnn_path, lens="shap")
        if lr is None or dnn is None:
            continue

        feature_names = lr["feature_names"] or dnn["feature_names"]
        if feature_names_ref is None:
            feature_names_ref = feature_names

        lr_k, dnn_k = lr["k_star"], dnn["k_star"]
        lr_k_list.append(lr_k)
        dnn_k_list.append(dnn_k)

        lr_centroids = np.array(lr["centroids"], dtype=float)
        dnn_centroids = np.array(dnn["centroids"], dtype=float)

        # Cross-pipeline centroid cosines
        cross_mat = np.zeros((lr_centroids.shape[0], dnn_centroids.shape[0]))
        for i in range(lr_centroids.shape[0]):
            for j_idx in range(dnn_centroids.shape[0]):
                cross_mat[i, j_idx] = cosine_sim(lr_centroids[i], dnn_centroids[j_idx])

        max_abs_cross = float(np.max(np.abs(cross_mat))) if cross_mat.size else 0.0
        mean_abs_cross = float(np.mean(np.abs(cross_mat))) if cross_mat.size else 0.0

        # Cross-pipeline gene Jaccard (bug-fixed: no variable shadowing)
        lr_gene_sets = [top_gene_set(c, feature_names, args.gene_top_n) for c in lr_centroids]
        dnn_gene_sets = [top_gene_set(c, feature_names, args.gene_top_n) for c in dnn_centroids]

        jaccards = []
        best_jacc = -1.0
        best_pair = None
        for i, lg in enumerate(lr_gene_sets):
            for j_idx, dg in enumerate(dnn_gene_sets):
                inter = len(lg & dg)
                union = len(lg | dg)
                jacc_val = inter / union if union > 0 else 0.0
                jaccards.append(jacc_val)
                if jacc_val > best_jacc:
                    best_jacc = jacc_val
                    best_pair = (i, j_idx)

        record = {
            "split": split_id,
            "lr_k": lr_k, "dnn_k": dnn_k,
            "lr_has_mult": bool(lr_k >= 2),
            "dnn_has_mult": bool(dnn_k >= 2),
            "max_abs_cross_cosine": round(max_abs_cross, 4),
            "mean_abs_cross_cosine": round(mean_abs_cross, 4),
            "max_gene_jaccard": round(float(max(jaccards)) if jaccards else 0.0, 4),
            "mean_gene_jaccard": round(float(np.mean(jaccards)) if jaccards else 0.0, 4),
        }
        split_records.append(record)

        print(f"  Split {split_id}: LR k*={lr_k}, DNN k*={dnn_k}, "
              f"max|cos|={max_abs_cross:.4f}, max J={record['max_gene_jaccard']:.4f}")

    if not split_records:
        sys.exit("[ERROR] No valid matching splits loaded.")

    n_common = len(split_records)
    lr_mult_rate = float(np.mean([r["lr_has_mult"] for r in split_records]))
    dnn_mult_rate = float(np.mean([r["dnn_has_mult"] for r in split_records]))
    max_cross_vals = np.array([r["max_abs_cross_cosine"] for r in split_records])
    max_jacc_vals = np.array([r["max_gene_jaccard"] for r in split_records])
    alignment_pass_rate = float(np.mean(max_cross_vals > args.alignment_threshold))
    gene_pass_rate = float(np.mean(max_jacc_vals > args.gene_jaccard_threshold))

    # Evidence level
    if n_common < args.min_splits_prelim:
        evidence_level = "exploratory"
    elif n_common < args.min_splits_strong:
        evidence_level = "preliminary"
    else:
        evidence_level = "strong"

    both_show_mult = bool(lr_mult_rate > 0.5 and dnn_mult_rate > 0.5)
    some_overlap = bool(alignment_pass_rate > 0.25 or gene_pass_rate > 0.25)

    print(f"\n{'='*70}\nAGGREGATE SUMMARY\n{'='*70}")
    print(f"  Matching splits         : {n_common}")
    print(f"  LR k* distribution      : {dict(Counter(lr_k_list))}")
    print(f"  DNN k* distribution     : {dict(Counter(dnn_k_list))}")
    print(f"  LR mult rate            : {lr_mult_rate:.3f}")
    print(f"  DNN mult rate           : {dnn_mult_rate:.3f}")
    print(f"  Alignment pass rate     : {alignment_pass_rate:.3f}")
    print(f"  Gene pass rate          : {gene_pass_rate:.3f}")
    print(f"  Evidence level          : {evidence_level}")

    # Verdict
    print(f"\n{'='*70}\nTEST 11 VERDICT\n{'='*70}")

    if evidence_level == "exploratory":
        tier = "INCONCLUSIVE"
        v = (f"INCONCLUSIVE — Only {n_common} matching splits. "
             f"Too small for a general claim.")
    elif both_show_mult and some_overlap and evidence_level == "strong":
        tier = "STRONG"
        v = (f"ATTACK SUBSTANTIALLY WEAKENED — Both LR-CGRID-SHAP and "
             f"DNN-SHAP show multiplicity across {n_common} matching splits "
             f"(LR mult rate={lr_mult_rate:.3f}, DNN={dnn_mult_rate:.3f}). "
             f"Cross-pipeline alignment pass rate={alignment_pass_rate:.3f}, "
             f"gene pass rate={gene_pass_rate:.3f}. Multiplicity is not "
             f"exclusive to LR.")
    elif both_show_mult:
        tier = "PARTIAL"
        v = (f"PARTIAL — Both pipelines show multiplicity but "
             f"evidence is {evidence_level} ({n_common} splits).")
    elif lr_mult_rate > 0.5 and dnn_mult_rate <= 0.5:
        tier = "PARTIAL"
        v = (f"PARTIAL — LR shows robust multiplicity but DNN does not "
             f"consistently (DNN mult rate={dnn_mult_rate:.3f}).")
    else:
        tier = "INCONCLUSIVE"
        v = "INCONCLUSIVE — no clear cross-pipeline conclusion."

    print(f"\n  >>> {v}")
    print("=" * 70)

    results = {
        "test_id": "test11_cross_pipeline",
        "version": "v2",
        "config": vars(args),
        "n_matching_splits": n_common,
        "evidence": {
            "evidence_level": evidence_level,
            "both_show_mult": both_show_mult,
            "some_overlap": some_overlap,
        },
        "measurements": {
            "lr_k_distribution": dict(Counter(lr_k_list)),
            "dnn_k_distribution": dict(Counter(dnn_k_list)),
            "lr_mult_rate": round(lr_mult_rate, 4),
            "dnn_mult_rate": round(dnn_mult_rate, 4),
            "alignment_pass_rate": round(alignment_pass_rate, 4),
            "gene_pass_rate": round(gene_pass_rate, 4),
            "median_max_cross_cosine": round(float(np.median(max_cross_vals)), 4),
            "mean_max_cross_cosine": round(float(np.mean(max_cross_vals)), 4),
            "median_max_gene_jaccard": round(float(np.median(max_jacc_vals)), 4),
            "mean_max_gene_jaccard": round(float(np.mean(max_jacc_vals)), 4),
        },
        "per_split": split_records,
        "verdict_tier": tier,
        "verdict_text": v,
    }

    json_path = outdir / "tcga_test11_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {json_path}")


if __name__ == "__main__":
    main()
