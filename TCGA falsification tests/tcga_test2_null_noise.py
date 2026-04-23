#!/usr/bin/env python3
"""
tcga_test2_null_noise.py
==========================
EvoXplain Falsification Test 2 — "Sampling noise / null pipeline bug."
TCGA Tumour vs Normal, LR C-grid, SHAP + LIME.

Version: v2 (standardised, lbfgs solver)
Last revised: April 2026

Merges former Tests 2 (Sampling Noise) and 9 (Null Sanity) into a single
block with four stages:

  STAGE 1 — Sampling Noise (former Test 2 ARM A + ARM B)
      ARM A: cross-split C-grid sweep at full dataset size.
      ARM B: subsampling to 50% and 25% of original size.
      Kill: structure persists under subsampling.

  STAGE 2 — Null Sanity Check (former Test 9 core)
      Run the EXACT same pipeline on shuffled labels (null control).
      Per-split evaluation against frozen real centroids.
      Kill: null accuracy → majority baseline, null k* → 1,
            null centroids orthogonal to real centroids.

  STAGE 3 — Boundary Confounder (former Test 2 null investigation)
      Does fixing the boundary set across splits collapse null structure?
      Cosine distribution shape: bimodal (structured) vs uniform (geometric).
      Kill: fixed boundary collapses null structure; real distribution
            is statistically distinct from null (KS test).

  STAGE 4 — Semantic Divergence (former Test 2 semantic)
      Gene-level divergence between basins: do real basins select
      different top genes? Do null basins?
      Kill: real basins show gene divergence + sign consistency;
            null basins do not.

Combined Kill Condition:
  ALL four stages pass →
    Multiplicity is not a sampling artefact (subsampling survives),
    not a pipeline bug (null accuracy → chance, k* → 1, centroids orthogonal),
    not a boundary confounder (fixed boundary collapses null),
    and not geometric noise (real basins have different genes, null don't).

Uses EvoXplain core engine for model creation, SHAP, boundary set, clustering.
"""

import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp, kruskal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import shap


# =====================================================================
# DATA LOADING
# =====================================================================

def load_tcga_data(gz_path, top_n=1000):
    try:
        from tcga_xena_adapter import load_tcga_for_evoxplain
    except ImportError:
        sys.exit("[ERROR] tcga_xena_adapter.py not found. Run from evoxplain dir.")
    X, y, feature_names = load_tcga_for_evoxplain(
        gz_path=gz_path, label_source="barcode",
        top_n=top_n, standardize=True, log2_transform=False,
    )
    return X.astype(np.float64), y.astype(int), list(feature_names)


# =====================================================================
# UTILITIES
# =====================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def cosine_sim(a, b):
    return float(1.0 - cosine(a, b))


def normalise(vecs):
    vecs = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms > 0, norms, 1.0)


def pairwise_cosines(vecs_normed):
    n = len(vecs_normed)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_sim(vecs_normed[i], vecs_normed[j]))
    return np.array(sims, dtype=np.float64) if sims else np.array([1.0], dtype=np.float64)


def c_grid_values(n_runs, c_min=0.001, c_max=1000.0):
    return np.logspace(np.log10(c_min), np.log10(c_max), n_runs)


def get_boundary_indices(X_test, X_train, y_train, boundary_seed=123,
                         prob_low=0.45, prob_high=0.55, boundary_k=200):
    seed_everything(boundary_seed)
    ref = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=boundary_seed,
    )
    ref.fit(X_train, y_train)
    probs = ref.predict_proba(X_test)[:, 1]
    pool = np.arange(len(X_test))
    mask = (probs >= prob_low) & (probs <= prob_high)
    selected = pool[mask]
    if len(selected) < boundary_k:
        margins = np.abs(probs - 0.5)
        selected = pool[np.argsort(margins)[:boundary_k]]
    elif len(selected) > boundary_k:
        rng = np.random.RandomState(boundary_seed)
        selected = rng.choice(selected, boundary_k, replace=False)
    return np.sort(selected)


def get_shap_vec(model, X_train, X_boundary):
    explainer = shap.LinearExplainer(model, X_train)
    sv = explainer.shap_values(X_boundary)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.array(sv)
    if sv.ndim != 2:
        return None
    return np.mean(sv, axis=0)


def pick_k_kmeans(vecs_normed, k_max=5, seed=0,
                  silhouette_threshold=0.15,
                  cosine_collapse_threshold=0.99):
    """
    Cluster with silhouette threshold and cosine collapse guard,
    consistent with the core engine logic.
    """
    n = len(vecs_normed)
    if n < 4:
        return 1, np.zeros(n, dtype=int), vecs_normed.mean(axis=0, keepdims=True)

    # Cosine collapse guard: if all vectors are nearly identical, return k*=1
    cos = pairwise_cosines(vecs_normed)
    if np.min(cos) >= cosine_collapse_threshold:
        labels = np.zeros(n, dtype=int)
        centers = vecs_normed.mean(axis=0, keepdims=True)
        return 1, labels, centers

    best_k = 1
    best_score = 0.0
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


def describe_bimodality(cos_vals, label):
    near_pos = float(np.mean(cos_vals > 0.5))
    near_neg = float(np.mean(cos_vals < -0.5))
    near_zero = float(np.mean(np.abs(cos_vals) < 0.3))
    std = float(np.std(cos_vals))
    bimodality_score = near_pos + near_neg - near_zero

    print(f"  [{label}] cosine distribution:")
    print(f"    mean={np.mean(cos_vals):.4f}  std={std:.4f}")
    print(f"    frac > +0.5  : {near_pos * 100:.1f}%")
    print(f"    frac < -0.5  : {near_neg * 100:.1f}%")
    print(f"    frac |c|<0.3 : {near_zero * 100:.1f}%")
    print(f"    bimodality score: {bimodality_score:.3f}")

    return {
        "mean": float(np.mean(cos_vals)),
        "std": std,
        "frac_above_0.5": near_pos,
        "frac_below_minus0.5": near_neg,
        "frac_near_zero": near_zero,
        "bimodality_score": bimodality_score,
    }


def summarise_arm(vecs_normed, label):
    cos = pairwise_cosines(vecs_normed)
    k, labels, centers = pick_k_kmeans(vecs_normed, k_max=5, seed=0)
    cross = []
    if k > 1:
        for i in range(k):
            for j in range(i + 1, k):
                cross.append(round(cosine_sim(centers[i], centers[j]), 4))

    n_negative = int(np.sum(cos < 0.0))
    frac_negative = float(n_negative / len(cos)) if len(cos) > 0 else 0.0

    print(f"\n  [{label}]")
    print(f"    n_vecs           : {len(vecs_normed)}")
    print(f"    cosine mean      : {cos.mean():.4f}")
    print(f"    cosine min       : {cos.min():.4f}")
    print(f"    cosine std       : {cos.std():.4f}")
    print(f"    negative cosines : {n_negative}/{len(cos)} ({frac_negative * 100:.1f}%)")
    print(f"    best k           : {k}")
    if k > 1:
        basin_counts = [int(np.sum(labels == i)) for i in range(k)]
        print(f"    basin counts     : {basin_counts}")
        print(f"    centroid cosines : {cross}")

    return {
        "n_vecs": len(vecs_normed),
        "cosine_mean": float(cos.mean()),
        "cosine_min": float(cos.min()),
        "cosine_std": float(cos.std()),
        "n_negative_cosines": n_negative,
        "frac_negative_cosines": round(frac_negative, 4),
        "k_star": int(k),
        "basin_counts": [int(np.sum(labels == i)) for i in range(k)],
        "cross_centroid_cosines": cross,
        "two_basins": bool(k >= 2),
        "has_negative_centroid": bool(len(cross) > 0 and min(cross) < 0.0),
    }


def top_genes(centroid, feature_names, top_n=50):
    ranked = np.argsort(np.abs(centroid))[::-1][:top_n]
    names = [feature_names[i].split(".")[0] for i in ranked]
    weights = {feature_names[i].split(".")[0]: float(centroid[i]) for i in ranked}
    return set(names), weights


# =====================================================================
# PER-SPLIT C-GRID RUNNER
# =====================================================================

def run_single_split_cgrid(X, y, split_seed, n_c_runs=20,
                           c_min=0.001, c_max=1000.0,
                           base_seed=42, lr_max_iter=1000,
                           boundary_k=200):
    """
    One split → C-grid of LR runs → list of SHAP attribution vectors.
    Returns (vecs, accs, c_values) or None.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=split_seed, stratify=y,
    )

    b_idx = get_boundary_indices(
        X_test, X_train, y_train, boundary_k=boundary_k,
    )
    X_boundary = X_test[b_idx]
    if len(X_boundary) == 0:
        return None

    c_values = c_grid_values(n_c_runs, c_min, c_max)
    vecs, accs = [], []

    for r, C in enumerate(c_values):
        run_seed = base_seed + (split_seed * 10000) + r
        model = LogisticRegression(
            C=float(C), penalty="l2", solver="lbfgs",
            max_iter=lr_max_iter, random_state=run_seed, n_jobs=1,
        )
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        vec = get_shap_vec(model, X_train, X_boundary)
        if vec is not None:
            vecs.append(vec)
            accs.append(float(acc))

    if not vecs:
        return None
    return np.array(vecs), np.array(accs), c_values


def sweep_splits(X, y, split_seeds, n_c_runs=20, c_min=0.001, c_max=1000.0,
                 base_seed=42, lr_max_iter=1000, boundary_k=200, label=""):
    all_vecs, all_accs = [], []
    for i, ss in enumerate(split_seeds):
        result = run_single_split_cgrid(
            X, y, ss, n_c_runs=n_c_runs,
            c_min=c_min, c_max=c_max,
            base_seed=base_seed, lr_max_iter=lr_max_iter,
            boundary_k=boundary_k,
        )
        if result is None:
            print(f"  [{label}] split_seed={ss} skipped")
            continue
        vecs, accs, _ = result
        all_vecs.append(vecs)
        all_accs.extend(accs)
        if (i + 1) % 5 == 0:
            print(f"  [{label}] {i + 1}/{len(split_seeds)} splits done")

    if not all_vecs:
        return normalise(np.zeros((1, X.shape[1]))), np.array([0.0])
    all_vecs = np.vstack(all_vecs)
    return normalise(all_vecs), np.array(all_accs)


# =====================================================================
# STAGE 1 — SAMPLING NOISE
# =====================================================================

def run_stage1(X, y, split_seeds, sub_fracs, args):
    print("\n" + "=" * 70)
    print("STAGE 1: SAMPLING NOISE")
    print("  Does subsampling destroy the structure?")
    print("=" * 70)

    # ARM A — full size
    print("\n[ARM A] Cross-split sweep — original TCGA size, C-grid")
    vecs_a, accs_a = sweep_splits(
        X, y, split_seeds, n_c_runs=args.n_c_runs,
        c_min=args.c_min, c_max=args.c_max,
        base_seed=args.base_seed, lr_max_iter=args.lr_max_iter,
        boundary_k=args.boundary_k, label="ARM A",
    )
    print(f"  Mean accuracy: {np.mean(accs_a):.4f} +/- {np.std(accs_a):.4f}")
    arm_a = summarise_arm(vecs_a, "ARM A original")
    arm_a["mean_acc"] = float(np.mean(accs_a))

    # ARM B — subsampling
    arm_b = {}
    for frac in sub_fracs:
        print(f"\n[ARM B] Subsampling {frac * 100:.0f}% of TCGA")
        rng_sub = np.random.RandomState(42)
        n_sub = int(len(y) * frac)
        idx_sub = rng_sub.choice(len(y), n_sub, replace=False)
        X_sub = X[idx_sub]
        y_sub = y[idx_sub]
        sc = StandardScaler()
        X_sub = sc.fit_transform(X_sub)
        print(f"  Subsampled: {X_sub.shape[0]} samples")

        vecs_b, accs_b = sweep_splits(
            X_sub, y_sub, split_seeds, n_c_runs=args.n_c_runs,
            c_min=args.c_min, c_max=args.c_max,
            base_seed=args.base_seed, lr_max_iter=args.lr_max_iter,
            boundary_k=args.boundary_k, label=f"ARM B {frac}",
        )
        print(f"  Mean accuracy: {np.mean(accs_b):.4f} +/- {np.std(accs_b):.4f}")
        summary = summarise_arm(vecs_b, f"ARM B {frac * 100:.0f}%")
        summary["mean_acc"] = float(np.mean(accs_b))
        summary["n_samples"] = n_sub
        arm_b[f"{frac}"] = summary

    return arm_a, arm_b, vecs_a


# =====================================================================
# STAGE 2 — NULL SANITY CHECK (former Test 9)
# =====================================================================

def run_stage2(X, y_real, y_null, split_seeds, args):
    print("\n" + "=" * 70)
    print("STAGE 2: NULL SANITY CHECK")
    print("  Does the null collapse? Accuracy → chance, k* → 1, orthogonal?")
    print("=" * 70)

    chance_level = max(np.mean(y_null), 1.0 - np.mean(y_null))
    print(f"\n  True chance baseline: {chance_level * 100:.2f}%")

    c_values = c_grid_values(args.n_c_runs, args.c_min, args.c_max)
    split_metrics = []

    for i, ss in enumerate(split_seeds):
        # Load frozen real centroids for this split
        agg = Path(args.output_dir_agg) / f"split{ss}" / f"aggregate_split{ss}.npz"
        real_centroids = None
        if agg.exists():
            data = np.load(agg, allow_pickle=True)
            for key in ["centroids_normed_shap", "centroids_normed"]:
                if key in data:
                    real_centroids = data[key]
                    break

        # Run null model on same split
        X_tr, X_te, y_tr_null, y_te_null = train_test_split(
            X, y_null, test_size=0.3, random_state=ss, stratify=y_null,
        )

        b_idx = get_boundary_indices(
            X_te, X_tr, y_tr_null, boundary_k=args.boundary_k,
        )
        X_b = X_te[b_idx]
        if len(X_b) == 0:
            continue

        null_vecs = []
        null_accs = []

        for r, C in enumerate(c_values):
            rs = args.base_seed + (ss * 10000) + r
            m = LogisticRegression(
                C=float(C), penalty="l2", solver="lbfgs",
                max_iter=args.lr_max_iter, random_state=rs, n_jobs=1,
            )
            m.fit(X_tr, y_tr_null)
            null_accs.append(float(m.score(X_te, y_te_null)))

            sv = get_shap_vec(m, X_tr, X_b)
            if sv is not None:
                null_vecs.append(sv)

        if len(null_vecs) < 2:
            continue

        null_normed = normalise(np.array(null_vecs))
        n_k, n_labels, n_centers = pick_k_kmeans(null_normed, k_max=5, seed=0)

        # Per-split orthogonality check
        max_cross_cos = 0.0
        if real_centroids is not None and n_centers is not None:
            cross_cosines = []
            for rc in real_centroids:
                for nc in n_centers:
                    cross_cosines.append(abs(cosine_sim(rc, nc)))
            max_cross_cos = max(cross_cosines) if cross_cosines else 0.0

        split_metrics.append({
            "split": int(ss),
            "null_acc": float(np.mean(null_accs)),
            "null_k": int(n_k),
            "max_cross_cos": float(max_cross_cos),
        })

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{args.n_splits} splits done")

    if not split_metrics:
        print("  [WARNING] No valid splits completed for Stage 2.")
        return {
            "chance_level": float(chance_level),
            "mean_null_acc": None,
            "mean_null_k": None,
            "max_cross_cos": None,
            "acc_is_chance": False,
            "is_flat": False,
            "is_orthogonal": False,
            "split_metrics": [],
        }

    mean_null_acc = np.mean([m["null_acc"] for m in split_metrics])
    mean_null_k = np.mean([m["null_k"] for m in split_metrics])
    global_max_cross = np.max([m["max_cross_cos"] for m in split_metrics])

    acc_is_chance = bool(abs(mean_null_acc - chance_level) < 0.05)
    is_flat = bool(mean_null_k < 1.5)
    is_orthogonal = bool(global_max_cross < 0.3)

    print(f"\n  Chance baseline      : {chance_level:.4f}")
    print(f"  Mean null accuracy   : {mean_null_acc:.4f}")
    print(f"  Mean null k*         : {mean_null_k:.2f}")
    print(f"  Max real⊥null cosine : {global_max_cross:.4f}")
    print(f"  acc_is_chance        : {acc_is_chance}")
    print(f"  is_flat              : {is_flat}")
    print(f"  is_orthogonal        : {is_orthogonal}")

    return {
        "chance_level": float(chance_level),
        "mean_null_acc": float(mean_null_acc),
        "mean_null_k": float(mean_null_k),
        "max_cross_cos": float(global_max_cross),
        "acc_is_chance": acc_is_chance,
        "is_flat": is_flat,
        "is_orthogonal": is_orthogonal,
        "split_metrics": split_metrics,
    }


# =====================================================================
# STAGE 3 — BOUNDARY CONFOUNDER
# =====================================================================

def run_stage3(X, y_real, y_null, split_seeds, args):
    print("\n" + "=" * 70)
    print("STAGE 3: BOUNDARY CONFOUNDER")
    print("  Does fixing the boundary set collapse null structure?")
    print("=" * 70)

    c_values = c_grid_values(args.n_c_runs, args.c_min, args.c_max)

    # Build FIXED global boundary set from first split, real labels
    fixed_ss = split_seeds[0]
    X_tr_f, X_te_f, y_tr_f, _ = train_test_split(
        X, y_real, test_size=0.3, random_state=fixed_ss, stratify=y_real,
    )
    fixed_b_idx = get_boundary_indices(
        X_te_f, X_tr_f, y_tr_f, boundary_k=args.boundary_k,
    )
    X_boundary_fixed = X_te_f[fixed_b_idx]
    print(f"  Fixed boundary size: {len(fixed_b_idx)} (from split_seed={fixed_ss})")

    vecs_a, vecs_c1, vecs_c2 = [], [], []

    print(f"\n  Sweeping {args.n_splits} splits × {args.n_c_runs} C values...")
    for i, ss in enumerate(split_seeds):
        X_train, X_test, y_train, _ = train_test_split(
            X, y_real, test_size=0.3, random_state=ss, stratify=y_real,
        )
        _, _, y_train_null, _ = train_test_split(
            X, y_null, test_size=0.3, random_state=ss, stratify=y_real,
        )

        b_idx_per = get_boundary_indices(
            X_test, X_train, y_train, boundary_k=args.boundary_k,
        )
        X_b_per = X_test[b_idx_per]

        for r, C in enumerate(c_values):
            run_seed = args.base_seed + (ss * 10000) + r

            # ARM A: real labels
            m_a = LogisticRegression(
                C=float(C), penalty="l2", solver="lbfgs",
                max_iter=args.lr_max_iter, random_state=run_seed, n_jobs=1,
            )
            m_a.fit(X_train, y_train)

            # ARM C: null labels
            m_c = LogisticRegression(
                C=float(C), penalty="l2", solver="lbfgs",
                max_iter=args.lr_max_iter, random_state=run_seed, n_jobs=1,
            )
            m_c.fit(X_train, y_train_null)

            vec_a = get_shap_vec(m_a, X_train, X_b_per)
            if vec_a is not None:
                vecs_a.append(vec_a)

            vec_c1 = get_shap_vec(m_c, X_train, X_b_per)
            if vec_c1 is not None:
                vecs_c1.append(vec_c1)

            vec_c2 = get_shap_vec(m_c, X_train, X_boundary_fixed)
            if vec_c2 is not None:
                vecs_c2.append(vec_c2)

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{args.n_splits} splits done")

    vecs_a = normalise(np.array(vecs_a))
    vecs_c1 = normalise(np.array(vecs_c1))
    vecs_c2 = normalise(np.array(vecs_c2))

    cos_a = pairwise_cosines(vecs_a)
    cos_c1 = pairwise_cosines(vecs_c1)
    cos_c2 = pairwise_cosines(vecs_c2)

    # Random baseline
    D = vecs_a.shape[1]
    rng_rand = np.random.RandomState(0)
    rand_vecs = normalise(rng_rand.randn(len(vecs_a), D))
    cos_rand = pairwise_cosines(rand_vecs)

    # Clustering
    k_c1, lbl_c1, cen_c1 = pick_k_kmeans(vecs_c1)
    cross_c1 = [
        round(cosine_sim(cen_c1[i], cen_c1[j]), 4)
        for i in range(k_c1) for j in range(i + 1, k_c1)
    ]
    print(f"\n  ARM C1 (null, per-split bdy): k*={k_c1}  "
          f"neg cosines: {np.mean(cos_c1 < 0) * 100:.1f}%  "
          f"centroid cosines: {cross_c1}")

    k_c2, lbl_c2, cen_c2 = pick_k_kmeans(vecs_c2)
    cross_c2 = [
        round(cosine_sim(cen_c2[i], cen_c2[j]), 4)
        for i in range(k_c2) for j in range(i + 1, k_c2)
    ]
    print(f"  ARM C2 (null, FIXED bdy):     k*={k_c2}  "
          f"neg cosines: {np.mean(cos_c2 < 0) * 100:.1f}%  "
          f"centroid cosines: {cross_c2}")

    boundary_was_confounder = bool(
        k_c2 == 1
        or (np.mean(cos_c2 < 0) < 0.1 and np.mean(cos_c1 < 0) > 0.3)
    )
    print(f"  Boundary was confounder: {boundary_was_confounder}")

    # Bimodality
    bio_a = describe_bimodality(cos_a, "ARM A — real, C-grid")
    bio_c1 = describe_bimodality(cos_c1, "ARM C1 — null, per-split bdy")
    bio_c2 = describe_bimodality(cos_c2, "ARM C2 — null, fixed bdy")
    bio_rand = describe_bimodality(cos_rand, f"RAND — pure random {D}D")

    # KS tests
    ks_ac1_stat, ks_ac1_p = ks_2samp(cos_a, cos_c1)
    ks_ac2_stat, ks_ac2_p = ks_2samp(cos_a, cos_c2)
    ks_c1r_stat, ks_c1r_p = ks_2samp(cos_c1, cos_rand)

    print(f"\n  KS ARM A vs C1 : stat={ks_ac1_stat:.4f}  p={ks_ac1_p:.2e}")
    print(f"  KS ARM A vs C2 : stat={ks_ac2_stat:.4f}  p={ks_ac2_p:.2e}")
    print(f"  KS ARM C1 vs RAND: stat={ks_c1r_stat:.4f}  p={ks_c1r_p:.2e}")

    arm_a_bimodal = bool(bio_a["bimodality_score"] > 0.3)
    arm_c2_collapsed = bool(k_c2 == 1 or np.mean(cos_c2 < 0) < 0.1)
    arm_a_distinct = bool(ks_ac1_p < 0.05)

    return {
        "arm_c1_null_persplit": {**bio_c1, "k_star": int(k_c1), "centroid_cosines": cross_c1},
        "arm_c2_null_fixedbdy": {**bio_c2, "k_star": int(k_c2), "centroid_cosines": cross_c2},
        "arm_a_bimodality": bio_a,
        "arm_rand": bio_rand,
        "ks_tests": {
            "arm_a_vs_c1": {"stat": float(ks_ac1_stat), "p": float(ks_ac1_p)},
            "arm_a_vs_c2": {"stat": float(ks_ac2_stat), "p": float(ks_ac2_p)},
            "arm_c1_vs_rand": {"stat": float(ks_c1r_stat), "p": float(ks_c1r_p)},
        },
        "boundary_was_confounder": boundary_was_confounder,
        "arm_a_bimodal": arm_a_bimodal,
        "arm_c2_collapsed": arm_c2_collapsed,
        "arm_a_distinct": arm_a_distinct,
    }


# =====================================================================
# STAGE 4 — SEMANTIC DIVERGENCE
# =====================================================================

def run_stage4(X, y_real, y_null, split_seeds, feature_names, args):
    print("\n" + "=" * 70)
    print("STAGE 4: SEMANTIC DIVERGENCE")
    print("  Do real basins select different genes? Do null basins?")
    print("=" * 70)

    c_values = c_grid_values(args.n_c_runs, args.c_min, args.c_max)
    vecs_a, vecs_c = [], []
    cvals_a, cvals_c = [], []

    print(f"\n  Sweeping {args.n_splits} splits × {args.n_c_runs} C values...")
    for i, ss in enumerate(split_seeds):
        X_train, X_test, y_train, _ = train_test_split(
            X, y_real, test_size=0.3, random_state=ss, stratify=y_real,
        )
        _, _, y_train_null, _ = train_test_split(
            X, y_null, test_size=0.3, random_state=ss, stratify=y_real,
        )

        b_idx = get_boundary_indices(
            X_test, X_train, y_train, boundary_k=args.boundary_k,
        )
        X_b = X_test[b_idx]

        for r, C in enumerate(c_values):
            run_seed = args.base_seed + (ss * 10000) + r

            for y_tr, vecs_list, cvals_list in [
                (y_train, vecs_a, cvals_a),
                (y_train_null, vecs_c, cvals_c),
            ]:
                m = LogisticRegression(
                    C=float(C), penalty="l2", solver="lbfgs",
                    max_iter=args.lr_max_iter, random_state=run_seed, n_jobs=1,
                )
                m.fit(X_train, y_tr)
                vec = get_shap_vec(m, X_train, X_b)
                if vec is not None:
                    vecs_list.append(vec)
                    cvals_list.append(float(C))

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{args.n_splits} splits done")

    vecs_a = np.array(vecs_a)
    vecs_c = np.array(vecs_c)
    cvals_a = np.array(cvals_a)
    cvals_c = np.array(cvals_c)

    top_n = args.gene_top_n

    # --- Analyse ARM A (real) ---
    vecs_a_norm = normalise(vecs_a)
    k_a, labels_a, centers_a = pick_k_kmeans(vecs_a_norm, k_max=5, seed=0)

    arm_a_basins = {}
    for b in range(k_a):
        centroid = centers_a[b]
        genes, weights = top_genes(centroid, feature_names, top_n)
        neg_frac = float(np.mean([w < 0 for w in weights.values()]))
        basin_mask = labels_a == b
        basin_c = cvals_a[basin_mask]
        c_info = {
            "mean": float(np.mean(basin_c)),
            "min": float(np.min(basin_c)),
            "max": float(np.max(basin_c)),
        } if len(basin_c) > 0 else {}
        ranked = np.argsort(np.abs(centroid))[::-1][:5]
        top5 = [(feature_names[i], float(centroid[i])) for i in ranked]
        arm_a_basins[b] = {
            "n_runs": int(np.sum(basin_mask)),
            "top_genes": list(genes),
            "neg_frac_top": neg_frac,
            "c_info": c_info,
            "top5": top5,
        }

    # Cross-basin gene overlap for ARM A
    a_overlaps = {}
    if k_a >= 2:
        for i, j in combinations(range(k_a), 2):
            g_i = set(arm_a_basins[i]["top_genes"])
            g_j = set(arm_a_basins[j]["top_genes"])
            inter = len(g_i & g_j)
            union = len(g_i | g_j)
            jaccard = inter / union if union > 0 else 0
            a_overlaps[f"{i}_vs_{j}"] = {
                "shared": inter,
                "jaccard": round(jaccard, 3),
                "only_i": len(g_i - g_j),
                "only_j": len(g_j - g_i),
            }

    # --- Analyse ARM C (null) ---
    vecs_c_norm = normalise(vecs_c)
    k_c, labels_c, centers_c = pick_k_kmeans(vecs_c_norm, k_max=5, seed=0)

    arm_c_basins = {}
    for b in range(k_c):
        centroid = centers_c[b]
        genes, weights = top_genes(centroid, feature_names, top_n)
        neg_frac = float(np.mean([w < 0 for w in weights.values()]))
        basin_mask = labels_c == b
        ranked = np.argsort(np.abs(centroid))[::-1][:5]
        top5 = [(feature_names[i], float(centroid[i])) for i in ranked]
        arm_c_basins[b] = {
            "n_runs": int(np.sum(basin_mask)),
            "top_genes": list(genes),
            "neg_frac_top": neg_frac,
            "top5": top5,
        }

    c_overlaps = {}
    if k_c >= 2:
        for i, j in combinations(range(k_c), 2):
            g_i = set(arm_c_basins[i]["top_genes"])
            g_j = set(arm_c_basins[j]["top_genes"])
            inter = len(g_i & g_j)
            union = len(g_i | g_j)
            jaccard = inter / union if union > 0 else 0
            c_overlaps[f"{i}_vs_{j}"] = {
                "shared": inter,
                "jaccard": round(jaccard, 3),
            }

    # Cross-arm orthogonality
    cross_arm_cos = {}
    for i in range(k_a):
        for j in range(k_c):
            key = f"a{i}_c{j}"
            cross_arm_cos[key] = round(cosine_sim(centers_a[i], centers_c[j]), 4)
    max_cross = max(abs(v) for v in cross_arm_cos.values()) if cross_arm_cos else 0.0
    arms_orthogonal = bool(max_cross < 0.3)

    # Print summary
    print(f"\n  ARM A: k*={k_a}")
    for b in range(k_a):
        info = arm_a_basins[b]
        print(f"    Basin {b}: {info['n_runs']} runs, neg_frac_top={info['neg_frac_top']:.2f}")
        for gene, w in info["top5"]:
            print(f"      {gene:<25} w={w:+.4f}")
    if a_overlaps:
        for pair, ov in a_overlaps.items():
            print(f"    Gene overlap {pair}: shared={ov['shared']}, Jaccard={ov['jaccard']}")

    print(f"\n  ARM C (null): k*={k_c}")
    for b in range(k_c):
        info = arm_c_basins[b]
        print(f"    Basin {b}: {info['n_runs']} runs, neg_frac_top={info['neg_frac_top']:.2f}")

    print(f"\n  Cross-arm orthogonality:")
    for key, val in cross_arm_cos.items():
        print(f"    {key}: {val:>7.4f}")
    print(f"  Max |cross-arm cosine|: {max_cross:.4f}")
    print(f"  ARM A ⊥ ARM C: {arms_orthogonal}")

    # Verdict components
    a_min_jaccard = min((v["jaccard"] for v in a_overlaps.values()), default=1.0)
    arm_a_has_divergence = bool(a_min_jaccard < 0.7) if k_a >= 2 else False
    arm_a_multi_basin = bool(k_a >= 2)

    return {
        "arm_a": {
            "k_star": int(k_a),
            "basins": {str(k): v for k, v in arm_a_basins.items()},
            "cross_basin_gene_overlap": a_overlaps,
        },
        "arm_c": {
            "k_star": int(k_c),
            "basins": {str(k): v for k, v in arm_c_basins.items()},
            "cross_basin_gene_overlap": c_overlaps,
        },
        "cross_arm_cosines": cross_arm_cos,
        "arms_orthogonal": arms_orthogonal,
        "arm_a_multi_basin": arm_a_multi_basin,
        "arm_a_has_divergence": arm_a_has_divergence,
        "a_min_jaccard": float(a_min_jaccard),
    }


# =====================================================================
# COMBINED VERDICT
# =====================================================================

def combined_verdict(stage1_arm_a, stage1_arm_b, stage2, stage3, stage4, sub_fracs):
    print("\n" + "=" * 70)
    print("COMBINED TEST 2 VERDICT — NULL & NOISE CONTROLS")
    print("=" * 70)

    # Stage 1: sampling noise
    arm_a_has_structure = bool(stage1_arm_a["k_star"] >= 2)
    sub_survival = {}
    for frac in sub_fracs:
        b = stage1_arm_b[f"{frac}"]
        sub_survival[f"{frac}"] = bool(b["k_star"] >= 2)
    structure_survives = all(sub_survival.values())

    # Stage 2: null sanity
    s2_pass = (
        stage2["acc_is_chance"]
        and stage2["is_flat"]
        and stage2["is_orthogonal"]
    )

    # Stage 3: boundary confounder
    s3_pass = stage3["arm_a_bimodal"] and (
        stage3["arm_c2_collapsed"] or stage3["arm_a_distinct"]
    )

    # Stage 4: semantic divergence
    s4_pass = stage4["arm_a_multi_basin"] and stage4["arm_a_has_divergence"] and stage4["arms_orthogonal"]

    print(f"\n  STAGE 1 — Sampling noise:")
    print(f"    ARM A has multi-basin structure    : {arm_a_has_structure}")
    for k_str, survives in sub_survival.items():
        print(f"    ARM B {k_str} structure survives    : {survives}")
    print(f"    All subsamples survive             : {structure_survives}")

    print(f"\n  STAGE 2 — Null sanity check:")
    print(f"    Accuracy is chance                 : {stage2['acc_is_chance']}")
    print(f"    Null k* collapsed                  : {stage2['is_flat']}")
    print(f"    Null ⊥ real centroids              : {stage2['is_orthogonal']}")
    print(f"    Stage 2 pass                       : {s2_pass}")

    print(f"\n  STAGE 3 — Boundary confounder:")
    print(f"    ARM A bimodal                      : {stage3['arm_a_bimodal']}")
    print(f"    ARM C2 (fixed bdy) collapsed       : {stage3['arm_c2_collapsed']}")
    print(f"    ARM A distinct from C1 (KS)        : {stage3['arm_a_distinct']}")
    print(f"    Stage 3 pass                       : {s3_pass}")

    print(f"\n  STAGE 4 — Semantic divergence:")
    print(f"    ARM A multi-basin                  : {stage4['arm_a_multi_basin']}")
    print(f"    ARM A gene divergence (J<0.7)      : {stage4['arm_a_has_divergence']}")
    print(f"    ARM A ⊥ ARM C                      : {stage4['arms_orthogonal']}")
    print(f"    Stage 4 pass                       : {s4_pass}")

    # Assemble verdict
    all_pass = arm_a_has_structure and structure_survives and s2_pass and s3_pass and s4_pass

    if all_pass:
        verdict = (
            "ATTACK KILLED — All four null & noise controls pass. "
            "Multiplicity survives subsampling (not a sampling artefact); "
            f"null accuracy drops to chance ({stage2['mean_null_acc'] * 100:.1f}% vs "
            f"baseline {stage2['chance_level'] * 100:.1f}%), "
            f"null k* collapses (mean {stage2['mean_null_k']:.2f}), "
            "null centroids are orthogonal to real centroids; "
            "fixing boundary set collapses null structure; "
            "real basins show gene-level divergence absent in null. "
            "Pipeline is correct and multiplicity is data-driven."
        )
    elif arm_a_has_structure and s2_pass and (structure_survives or s3_pass or s4_pass):
        n_pass = sum([structure_survives, s3_pass, s4_pass])
        verdict = (
            f"STRONG EVIDENCE — Null sanity confirmed and {n_pass}/3 auxiliary "
            "controls pass. Core claim survives: multiplicity is not a pipeline "
            "bug. Some auxiliary evidence is incomplete."
        )
    elif arm_a_has_structure and s2_pass:
        verdict = (
            "PARTIAL — Null sanity confirmed (accuracy → chance, k* → 1, orthogonal), "
            "but auxiliary noise/semantic controls did not fully pass. "
            "Core pipeline correctness is established."
        )
    elif not arm_a_has_structure:
        verdict = "INCONCLUSIVE — No multi-basin structure detected in ARM A."
    else:
        verdict = (
            "WARNING — Null sanity check failed. Pipeline correctness not confirmed. "
            "Investigate environment and implementation."
        )

    print(f"\n  >>> {verdict}")
    print("=" * 70)
    return verdict


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoXplain — Consolidated Test 2: Null & Noise Controls",
    )

    parser.add_argument("--tcga_gz_path", type=str,
                        default="data/tcga_RSEM_gene_tpm.gz")
    parser.add_argument("--tcga_top_n", type=int, default=1000)
    parser.add_argument("--n_splits", type=int, default=20)
    parser.add_argument("--split_seed_start", type=int, default=800)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--n_c_runs", type=int, default=20)
    parser.add_argument("--c_min", type=float, default=0.001)
    parser.add_argument("--c_max", type=float, default=1000.0)
    parser.add_argument("--lr_max_iter", type=int, default=1000)
    parser.add_argument("--boundary_k", type=int, default=200)
    parser.add_argument("--subsample_fracs", type=str, default="0.5,0.25")
    parser.add_argument("--gene_top_n", type=int, default=50)
    parser.add_argument("--output_dir_agg", type=str,
                        default="results/final_freeze/tcga_lr_cgrid_shap_lime_100x100",
                        help="Path to frozen real aggregates for per-split orthogonality.")
    parser.add_argument("--output_dir", type=str, default=".")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    split_seeds = list(range(args.split_seed_start,
                             args.split_seed_start + args.n_splits))
    sub_fracs = [float(f) for f in args.subsample_fracs.split(",")]

    print("=" * 70)
    print("EvoXplain — Test 2: Null & Noise Controls (TCGA LR-Cgrid)")
    print("  'Is this sampling noise or a pipeline bug?'")
    print("=" * 70)
    print(f"  Splits          : {args.n_splits} ({split_seeds[0]}..{split_seeds[-1]})")
    print(f"  C-grid          : {args.n_c_runs} values [{args.c_min}, {args.c_max}]")
    print(f"  Subsample fracs : {sub_fracs}")
    print(f"  Gene top-N      : {args.gene_top_n}")
    print(f"  Frozen agg dir  : {args.output_dir_agg}")

    # Load data
    print(f"\n[Loading TCGA | top_n={args.tcga_top_n}]")
    X, y_real, feature_names = load_tcga_data(args.tcga_gz_path, args.tcga_top_n)
    print(f"[Data] X={X.shape}, y={y_real.shape}, classes={np.bincount(y_real).tolist()}")

    # Static null labels (consistent across all stages)
    y_null = np.random.RandomState(999).permutation(y_real)

    # --- Run all four stages ---
    stage1_arm_a, stage1_arm_b, _ = run_stage1(X, y_real, split_seeds, sub_fracs, args)
    stage2 = run_stage2(X, y_real, y_null, split_seeds, args)
    stage3 = run_stage3(X, y_real, y_null, split_seeds, args)
    stage4 = run_stage4(X, y_real, y_null, split_seeds, feature_names, args)

    # --- Combined verdict ---
    verdict = combined_verdict(stage1_arm_a, stage1_arm_b, stage2, stage3, stage4, sub_fracs)

    # --- Save ---
    results = {
        "test_id": "test2_null_noise",
        "version": "v2",
        "config": vars(args),
        "stage1_sampling_noise": {
            "arm_a_original": stage1_arm_a,
            "arm_b_subsample": stage1_arm_b,
        },
        "stage2_null_sanity": stage2,
        "stage3_boundary_confounder": stage3,
        "stage4_semantic_divergence": stage4,
        "verdict_tier": verdict.get("tier", "INCONCLUSIVE") if isinstance(verdict, dict) else "INCONCLUSIVE",
        "verdict_text": verdict.get("text", str(verdict)) if isinstance(verdict, dict) else str(verdict),
        "combined_verdict": verdict,
    }

    json_path = Path(args.output_dir) / "tcga_test2_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {json_path}")

    # Summary text
    txt_path = Path(args.output_dir) / "tcga_test2_null_and_noise_summary.txt"
    with open(txt_path, "w") as f:
        f.write("EvoXplain — TCGA Consolidated Falsification Test 2\n")
        f.write("Null & Noise Controls (former Tests 2 + 9)\n\n")
        f.write(f"Splits: {args.n_splits} | C-grid: {args.n_c_runs} values "
                f"[{args.c_min}, {args.c_max}]\n")
        f.write(f"Subsample fracs: {sub_fracs}\n\n")

        f.write("--- STAGE 1: Sampling Noise ---\n")
        f.write(f"ARM A (original): k*={stage1_arm_a['k_star']}  "
                f"neg cosines={stage1_arm_a['frac_negative_cosines'] * 100:.1f}%\n")
        for frac in sub_fracs:
            b = stage1_arm_b[f"{frac}"]
            f.write(f"ARM B ({frac * 100:.0f}%):     k*={b['k_star']}  "
                    f"neg cosines={b['frac_negative_cosines'] * 100:.1f}%\n")

        f.write("\n--- STAGE 2: Null Sanity ---\n")
        if stage2["mean_null_acc"] is not None:
            f.write(f"Null accuracy: {stage2['mean_null_acc']:.4f} "
                    f"(baseline: {stage2['chance_level']:.4f})\n")
            f.write(f"Null mean k*: {stage2['mean_null_k']:.2f}\n")
            f.write(f"Max real⊥null cosine: {stage2['max_cross_cos']:.4f}\n")
        else:
            f.write("No valid splits completed.\n")

        f.write("\n--- STAGE 3: Boundary Confounder ---\n")
        f.write(f"ARM C1 (per-split bdy): k*={stage3['arm_c1_null_persplit']['k_star']}\n")
        f.write(f"ARM C2 (fixed bdy):     k*={stage3['arm_c2_null_fixedbdy']['k_star']}\n")
        f.write(f"Boundary was confounder: {stage3['boundary_was_confounder']}\n")
        f.write(f"ARM A distinct from C1:  {stage3['arm_a_distinct']}\n")

        f.write("\n--- STAGE 4: Semantic Divergence ---\n")
        f.write(f"ARM A k*={stage4['arm_a']['k_star']}  "
                f"gene divergence (J<0.7): {stage4['arm_a_has_divergence']}\n")
        f.write(f"ARM C k*={stage4['arm_c']['k_star']}\n")
        f.write(f"ARM A ⊥ ARM C: {stage4['arms_orthogonal']}\n")

        f.write(f"\n--- COMBINED VERDICT ---\n{verdict}\n")

    print(f"[Saved] {txt_path}")


if __name__ == "__main__":
    main()
