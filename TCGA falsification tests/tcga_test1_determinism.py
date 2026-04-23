#!/usr/bin/env python3
"""
tcga_test1_determinism.py
===========================
EvoXplain Falsification Test 1 — "Cosine collapse / stochastic noise."
TCGA Tumour vs Normal, LR C-grid, DUAL LENS (SHAP + LIME).

Version: v2 (standardised, merged from 4 arm scripts + verdict)
Last revised: April 2026

Attack
------
The basin structure is an artefact of random seed variation or numerical
noise. Retraining the same model produces the same explanation every time.

Defence Strategy
----------------
Four arms isolate randomness sources:
  ARM A — Replay determinism: same seed, same C, N reps. Must give k*=1.
  ARM B — Seed-only variability: varied seeds, fixed C. Isolates seed noise.
  ARM C — Pure C-only: fixed seed, varied C across log grid. If basins
          appear, they are 100% from the regularisation landscape.
  ARM D — Factorial C×seed: multiple seeds per C value. Tests whether seed
          noise creates, destroys, or preserves C-driven basin structure.

Kill Condition
--------------
  ARM A replay deterministic (k*=1, cosine_min > 0.9999)
  AND ARM C shows basin structure aligned with C regimes
  AND ARM B seed variation is weaker than ARM C structure
  AND ARM D mixed design preserves structure
  → ATTACK KILLED

Modes
-----
  --mode all      Run A→B→C→D→verdict sequentially (default)
  --mode arm_a    Run ARM A only (for parallel SLURM)
  --mode arm_b    Run ARM B only
  --mode arm_c    Run ARM C only
  --mode arm_d    Run ARM D only
  --mode verdict  Aggregate arm JSONs and compute verdict (lightweight)

Data
----
Arms A-D require TCGA data + core engine. Verdict reads JSONs only.

Dependencies
------------
  evoxplain_core_engine.py (load_dataset, make_model, compute_attributions,
                            pick_best_k_kmeans, seed_everything)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


# =====================================================================
# STANDARD UTILITIES
# =====================================================================

def cosine_sim(a, b):
    return float(1.0 - cosine(a, b))


def pairwise_cosines(vecs):
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(cosine_sim(vecs[i], vecs[j]))
    return np.array(sims, dtype=np.float64) if sims else np.array([1.0])


def normalise(X):
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def c_grid_values(n_runs, c_min=0.001, c_max=1000.0):
    return np.logspace(np.log10(c_min), np.log10(c_max), n_runs)


# =====================================================================
# BOUNDARY SET — from X_test only
# =====================================================================

def compute_boundary_set(X_test, X_train, y_train,
                         boundary_k=200, boundary_seed=123,
                         prob_low=0.45, prob_high=0.55):
    ref = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=boundary_seed)
    ref.fit(X_train, y_train)
    probs = ref.predict_proba(X_test)[:, 1]
    pool = np.arange(X_test.shape[0])
    mask = (probs >= prob_low) & (probs <= prob_high)
    selected = pool[mask]
    if len(selected) < boundary_k:
        selected = pool[np.argsort(np.abs(probs - 0.5))[:boundary_k]]
    elif len(selected) > boundary_k:
        rng = np.random.RandomState(boundary_seed)
        selected = rng.choice(selected, boundary_k, replace=False)
    print(f"  [Boundary] {len(np.sort(selected))} samples", flush=True)
    return np.sort(selected)


# =====================================================================
# SINGLE RUN + SUMMARY
# =====================================================================

def run_single(args, X_train, y_train, X_boundary, C_val,
               model_seed, attr_seed, attr,
               make_model_fn, compute_attr_fn, seed_all_fn):
    seed_all_fn(model_seed)
    model = make_model_fn(
        args, X_train.shape[1], model_seed,
        C_override=float(C_val), l1_override=None)
    model.fit(X_train, y_train)
    attr_rng = np.random.RandomState(attr_seed)
    sv = compute_attr_fn(
        model, X_train, X_boundary, args,
        rng=attr_rng, attribution=attr)
    return np.mean(sv, axis=0)


def summarise_vectors(vecs, pick_k_fn, args):
    vecs_norm = normalise(np.array(vecs, dtype=np.float64))
    cos = pairwise_cosines(vecs_norm)
    k_star, labels, _ = pick_k_fn(
        vecs_norm, k_max=args.k_max, seed=0,
        silhouette_threshold=args.silhouette_threshold,
        cosine_collapse_threshold=args.cosine_collapse,
        silhouette_metric=args.silhouette_metric)

    result = {
        "k_star": int(k_star),
        "cosine_mean": float(np.mean(cos)),
        "cosine_min": float(np.min(cos)),
        "cosine_std": float(np.std(cos)),
        "labels": labels.tolist(),
        "cluster_sizes": {int(k): int(np.sum(labels == k))
                          for k in np.unique(labels)},
    }
    if k_star > 1:
        centers = [normalise(vecs_norm[labels == k].mean(axis=0, keepdims=True))[0]
                   for k in range(k_star)]
        result["cross_basin_cosines"] = [
            float(cosine_sim(centers[i], centers[j]))
            for i in range(k_star) for j in range(i + 1, k_star)]
    return result


def c_association_test(C_values, labels):
    C_values = np.asarray(C_values, dtype=np.float64)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return {"tested": False, "reason": "k*=1"}
    logC = np.log10(C_values)
    groups = [logC[labels == k] for k in unique]
    H, p = kruskal(*groups)
    n, k = len(logC), len(unique)
    eps2 = float(max(0, min(1, (H - k + 1) / (n - k)))) if n > k else None
    return {"tested": True, "kruskal_H": float(H), "kruskal_p": float(p),
            "epsilon_sq": eps2}


# =====================================================================
# ENGINE COMPATIBILITY
# =====================================================================

def set_engine_compat(args):
    """Set fields expected by evoxplain_core_engine."""
    args.dataset = "tcga"
    args.model = "lr"
    args.attribution = "shap,lime"
    args.lr_penalty = "l2"
    args.lr_C = args.fixed_C
    args.boundary_cache = 0
    args.boundary_source = "test"
    args.boundary_method = "prob_band"
    args.center = True
    args.normalize = "l2"
    args.lr_l1_ratio = None
    args.lr_C_mode = "fixed"
    args.lr_C_min = args.c_min
    args.lr_C_max = args.c_max
    args.lr_l1_ratio_mode = "fixed"
    args.rf_n_estimators = 100
    args.rf_max_depth = 10
    args.rf_max_features = "sqrt"
    args.train_resample = "none"
    args.shap_signed = True
    args.shap_bg_mode = "fixed"
    args.shap_background_size = 100
    args.shap_bg_seed = 12345
    args.explain_vector_agg = "mean"
    args.drop_features = None
    args.data_cache_dir = None
    args.data_path = None
    args.acs_states = None
    args.acs_year = None
    args.tcga_subtype_path = None
    args.base_seed = getattr(args, "base_seed", 42)
    args.chunk_id = 0
    args.chunk_size = args.n_reps
    args.n_runs = args.n_reps
    args.dnn_layers = "100,50"
    args.dnn_lr = 0.001
    args.dnn_epochs = 50
    args.dnn_batch_size = 64
    args.dnn_dropout = 0.2
    args.lime_discretize = 0
    args.attribution_lens = None
    return args


# =====================================================================
# ARM RUNNERS
# =====================================================================

def run_arm_a(args, X_train, y_train, X_boundary, fns):
    """ARM A: Replay determinism — same seed, same C, N reps."""
    print(f"\n{'='*70}\nARM A: Replay Determinism\n{'='*70}", flush=True)
    FIXED_MODEL_SEED, FIXED_ATTR_SEED = 42, 12345
    results = {}
    for attr in ["shap", "lime"]:
        vecs = []
        print(f"  [{attr.upper()}]", flush=True)
        for i in range(args.n_reps):
            vec = run_single(args, X_train, y_train, X_boundary,
                             args.fixed_C, FIXED_MODEL_SEED, FIXED_ATTR_SEED,
                             attr, fns["make"], fns["attr"], fns["seed"])
            vecs.append(vec)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{args.n_reps}", flush=True)
        results[attr] = summarise_vectors(vecs, fns["pick_k"], args)
        print(f"    k*={results[attr]['k_star']}  cos_min={results[attr]['cosine_min']:.6f}", flush=True)
    return {"arm": "A", "config": {"fixed_C": args.fixed_C, "n_reps": args.n_reps},
            "results": results}


def run_arm_b(args, X_train, y_train, X_boundary, fns):
    """ARM B: Seed-only variability — varied seeds, fixed C."""
    print(f"\n{'='*70}\nARM B: Seed-Only Variability\n{'='*70}", flush=True)
    BASE = args.base_seed + (args.split_seed * 10000)
    model_seeds = [BASE + i for i in range(args.n_reps)]
    attr_seeds = [BASE + 50000 + i for i in range(args.n_reps)]
    results = {}
    for attr in ["shap", "lime"]:
        vecs = []
        print(f"  [{attr.upper()}]", flush=True)
        for i in range(args.n_reps):
            vec = run_single(args, X_train, y_train, X_boundary,
                             args.fixed_C, model_seeds[i], attr_seeds[i],
                             attr, fns["make"], fns["attr"], fns["seed"])
            vecs.append(vec)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{args.n_reps}", flush=True)
        results[attr] = summarise_vectors(vecs, fns["pick_k"], args)
        print(f"    k*={results[attr]['k_star']}  cos_min={results[attr]['cosine_min']:.6f}", flush=True)
    return {"arm": "B", "config": {"fixed_C": args.fixed_C, "n_reps": args.n_reps},
            "results": results}


def run_arm_c(args, X_train, y_train, X_boundary, fns):
    """ARM C: Pure C-only — fixed seed, varied C across log grid."""
    print(f"\n{'='*70}\nARM C: Pure C-Only Variability\n{'='*70}", flush=True)
    FIXED_MODEL_SEED, FIXED_ATTR_SEED = 42, 12345
    C_grid = c_grid_values(args.n_reps, args.c_min, args.c_max)
    results = {}
    for attr in ["shap", "lime"]:
        vecs = []
        print(f"  [{attr.upper()}]", flush=True)
        for i, C_val in enumerate(C_grid):
            vec = run_single(args, X_train, y_train, X_boundary,
                             C_val, FIXED_MODEL_SEED, FIXED_ATTR_SEED,
                             attr, fns["make"], fns["attr"], fns["seed"])
            vecs.append(vec)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{args.n_reps} C={float(C_val):.4g}", flush=True)
        summary = summarise_vectors(vecs, fns["pick_k"], args)
        summary["c_association"] = c_association_test(C_grid, np.array(summary["labels"]))
        results[attr] = summary
        print(f"    k*={summary['k_star']}  cos_min={summary['cosine_min']:.6f}", flush=True)
        if summary["c_association"].get("tested"):
            print(f"    C-assoc: p={summary['c_association']['kruskal_p']:.4g}  "
                  f"eps2={summary['c_association']['epsilon_sq']}", flush=True)
    return {"arm": "C", "config": {"c_min": args.c_min, "c_max": args.c_max,
                                    "n_reps": args.n_reps, "C_grid": C_grid.tolist()},
            "results": results}


def run_arm_d(args, X_train, y_train, X_boundary, fns):
    """ARM D: Factorial C × seed design."""
    print(f"\n{'='*70}\nARM D: Factorial C x Seed\n{'='*70}", flush=True)
    C_grid = c_grid_values(args.n_c_values, args.c_min, args.c_max)
    BASE = args.base_seed + (args.split_seed * 10000)
    n_total = args.n_c_values * args.n_seeds_per_c
    # Override n_reps for engine compat
    args.n_reps = n_total
    args.chunk_size = n_total
    args.n_runs = n_total

    results = {}
    for attr in ["shap", "lime"]:
        vecs, run_C = [], []
        print(f"  [{attr.upper()}]", flush=True)
        run_idx = 0
        for ci, C_val in enumerate(C_grid):
            for si in range(args.n_seeds_per_c):
                ms = BASE + (ci * 100) + si
                at_s = BASE + 50000 + (ci * 100) + si
                vec = run_single(args, X_train, y_train, X_boundary,
                                 C_val, ms, at_s, attr,
                                 fns["make"], fns["attr"], fns["seed"])
                vecs.append(vec)
                run_C.append(float(C_val))
                run_idx += 1
            print(f"    C={float(C_val):.4g} done ({run_idx}/{n_total})", flush=True)
        summary = summarise_vectors(vecs, fns["pick_k"], args)
        summary["c_association"] = c_association_test(np.array(run_C), np.array(summary["labels"]))
        results[attr] = summary
        print(f"    k*={summary['k_star']}  cos_min={summary['cosine_min']:.6f}", flush=True)
    return {"arm": "D", "config": {"n_c_values": args.n_c_values,
                                    "n_seeds_per_c": args.n_seeds_per_c,
                                    "C_grid": C_grid.tolist()},
            "results": results}


# =====================================================================
# VERDICT
# =====================================================================

def compute_verdict(A, B, C, D, args):
    """Compute per-lens and combined verdict from arm results."""
    verdicts = {}
    for attr in ["shap", "lime"]:
        kA = A["results"][attr]["k_star"]
        kB = B["results"][attr]["k_star"]
        kC = C["results"][attr]["k_star"]
        kD = D["results"][attr]["k_star"]
        cA_min = A["results"][attr]["cosine_min"]
        cB_mean = B["results"][attr]["cosine_mean"]
        cC_mean = C["results"][attr]["cosine_mean"]
        cD_mean = D["results"][attr]["cosine_mean"]

        replay_ok = (kA == 1) and (cA_min >= args.replay_cosine_threshold)
        seed_weaker = (kC > kB) or (kC >= 2 and cC_mean < cB_mean - args.cosine_margin)
        c_structure = (kC >= 2)
        assoc = C["results"][attr].get("c_association", {})
        c_linked = (assoc.get("tested", False)
                    and assoc.get("kruskal_p", 1.0) < args.c_assoc_p_threshold
                    and (assoc.get("epsilon_sq") is None
                         or assoc.get("epsilon_sq", 0) >= args.c_assoc_effect_threshold))
        mixed_ok = (kD >= kC) or (kD >= 2 and cD_mean <= cC_mean + args.cosine_margin)

        if replay_ok and c_structure and c_linked and seed_weaker and mixed_ok:
            tier = "ATTACK KILLED"
            v = (f"ATTACK KILLED ({attr.upper()}) — Replay deterministic; "
                 f"seed-only variation weaker than C; pure C-only induces "
                 f"C-aligned basins (k*={kC}, p={assoc.get('kruskal_p', 'n/a'):.4g}).")
        elif replay_ok and c_structure and c_linked:
            tier = "STRONG"
            v = (f"STRONG ({attr.upper()}) — Replay deterministic; C-only "
                 f"induces C-aligned basins; seed-vs-C separation not maximal.")
        elif replay_ok:
            tier = "PARTIAL"
            v = f"PARTIAL ({attr.upper()}) — Replay deterministic but C-only evidence weak."
        else:
            tier = "INCONCLUSIVE"
            v = f"INCONCLUSIVE ({attr.upper()}) — Replay arm not deterministic."

        verdicts[attr] = {
            "verdict_tier": tier, "verdict_text": v,
            "metrics": {
                "ARM_A_k": kA, "ARM_B_k": kB, "ARM_C_k": kC, "ARM_D_k": kD,
                "ARM_A_cos_min": cA_min, "ARM_B_cos_mean": cB_mean,
                "ARM_C_cos_mean": cC_mean, "ARM_D_cos_mean": cD_mean,
                "replay_ok": replay_ok, "seed_weaker": seed_weaker,
                "c_structure": c_structure, "c_linked": c_linked, "mixed_ok": mixed_ok,
            },
        }
        print(f"\n  [{attr.upper()}]")
        print(f"    A: k*={kA} cos_min={cA_min:.6f} | B: k*={kB} | "
              f"C: k*={kC} | D: k*={kD}")
        print(f"    >>> {v}")

    tiers = [verdicts[a]["verdict_tier"] for a in ["shap", "lime"]]
    if all(t == "ATTACK KILLED" for t in tiers):
        combined_tier = "ATTACK KILLED"
        combined_text = "BOTH LENSES: ATTACK KILLED"
    elif all(t in ("ATTACK KILLED", "STRONG") for t in tiers):
        combined_tier = "STRONG"
        combined_text = "BOTH LENSES: STRONG EVIDENCE"
    else:
        combined_tier = "PARTIAL"
        combined_text = "MIXED — see per-lens verdicts."

    return verdicts, combined_tier, combined_text


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoXplain TCGA Test 1: Determinism / Cosine Collapse")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "arm_a", "arm_b", "arm_c", "arm_d", "verdict"],
                        help="Which arm(s) to run.")
    parser.add_argument("--tcga_gz_path", type=str, default="data/tcga_RSEM_gene_tpm.gz")
    parser.add_argument("--tcga_top_n", type=int, default=1000)
    parser.add_argument("--split_seed", type=int, default=800)
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--fixed_C", type=float, default=1.0)
    parser.add_argument("--c_min", type=float, default=0.001)
    parser.add_argument("--c_max", type=float, default=1000.0)
    parser.add_argument("--lr_max_iter", type=int, default=1000)
    parser.add_argument("--lime_n_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--base_seed", type=int, default=42)
    # Factorial (ARM D)
    parser.add_argument("--n_c_values", type=int, default=10)
    parser.add_argument("--n_seeds_per_c", type=int, default=5)
    # Clustering
    parser.add_argument("--cosine_collapse", type=float, default=0.99)
    parser.add_argument("--silhouette_threshold", type=float, default=0.15)
    parser.add_argument("--silhouette_metric", type=str, default="euclidean")
    parser.add_argument("--k_max", type=int, default=5)
    # Boundary
    parser.add_argument("--boundary_k", type=int, default=200)
    parser.add_argument("--boundary_seed", type=int, default=123)
    parser.add_argument("--boundary_prob_low", type=float, default=0.45)
    parser.add_argument("--boundary_prob_high", type=float, default=0.55)
    # Verdict thresholds
    parser.add_argument("--replay_cosine_threshold", type=float, default=0.9999)
    parser.add_argument("--cosine_margin", type=float, default=0.01)
    parser.add_argument("--c_assoc_p_threshold", type=float, default=0.01)
    parser.add_argument("--c_assoc_effect_threshold", type=float, default=0.10)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EvoXplain — Test 1: Determinism (TCGA LR-Cgrid)")
    print("  'Is the basin structure just stochastic noise?'")
    print("=" * 70)
    print(f"  Mode                   : {args.mode}")
    print(f"  Output                 : {args.output_dir}")

    # Verdict-only mode: just read JSONs
    if args.mode == "verdict":
        d = Path(args.output_dir)
        with open(d / "arm_a_results.json") as f: A = json.load(f)
        with open(d / "arm_b_results.json") as f: B = json.load(f)
        with open(d / "arm_c_results.json") as f: C = json.load(f)
        with open(d / "arm_d_results.json") as f: D = json.load(f)

        print(f"\n{'='*70}\nTEST 1 VERDICT\n{'='*70}")
        verdicts, combined_tier, combined_text = compute_verdict(A, B, C, D, args)
        print(f"\n  COMBINED: {combined_text}")

        results = {
            "test_id": "test1_determinism", "version": "v2",
            "config": vars(args),
            "per_lens_shap": verdicts["shap"],
            "per_lens_lime": verdicts["lime"],
            "verdict_tier": combined_tier, "verdict_text": combined_text,
        }
        json_path = d / "tcga_test1_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[Saved] {json_path}")
        return

    # Arms require core engine
    from evoxplain_core_engine import (
        load_dataset, make_model, compute_attributions,
        pick_best_k_kmeans, seed_everything,
    )
    args = set_engine_compat(args)
    fns = {"make": make_model, "attr": compute_attributions,
           "seed": seed_everything, "pick_k": pick_best_k_kmeans}

    # Load data once
    X, y, _ = load_dataset("tcga", tcga_gz_path=args.tcga_gz_path,
                           tcga_top_n=args.tcga_top_n)
    print(f"  [Data] X={X.shape}", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.split_seed, stratify=y)

    X_boundary = X_test[compute_boundary_set(
        X_test, X_train, y_train, args.boundary_k, args.boundary_seed,
        args.boundary_prob_low, args.boundary_prob_high)]

    d = Path(args.output_dir)
    run_modes = ["arm_a", "arm_b", "arm_c", "arm_d"] if args.mode == "all" else [args.mode]

    for mode in run_modes:
        if mode == "arm_a":
            out = run_arm_a(args, X_train, y_train, X_boundary, fns)
        elif mode == "arm_b":
            out = run_arm_b(args, X_train, y_train, X_boundary, fns)
        elif mode == "arm_c":
            out = run_arm_c(args, X_train, y_train, X_boundary, fns)
        elif mode == "arm_d":
            out = run_arm_d(args, X_train, y_train, X_boundary, fns)
        else:
            continue

        json_path = d / f"arm_{mode[-1]}_results.json"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  [Saved] {json_path}", flush=True)

    # Auto-verdict if running all
    if args.mode == "all":
        with open(d / "arm_a_results.json") as f: A = json.load(f)
        with open(d / "arm_b_results.json") as f: B = json.load(f)
        with open(d / "arm_c_results.json") as f: C = json.load(f)
        with open(d / "arm_d_results.json") as f: D = json.load(f)

        print(f"\n{'='*70}\nTEST 1 VERDICT\n{'='*70}")
        verdicts, combined_tier, combined_text = compute_verdict(A, B, C, D, args)
        print(f"\n  COMBINED: {combined_text}")
        print("=" * 70)

        results = {
            "test_id": "test1_determinism", "version": "v2",
            "config": vars(args),
            "per_lens_shap": verdicts["shap"],
            "per_lens_lime": verdicts["lime"],
            "verdict_tier": combined_tier, "verdict_text": combined_text,
        }
        json_path = d / "tcga_test1_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[Saved] {json_path}")


if __name__ == "__main__":
    main()
