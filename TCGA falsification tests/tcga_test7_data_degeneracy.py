#!/usr/bin/env python3
"""
tcga_test7_data_degeneracy.py
===============================
EvoXplain Falsification Test 7 — "Data degeneracy / feature redundancy."
TCGA Tumour vs Normal, LR C-grid, SHAP only.

Version: v2 (standardised, PCA verdict reframed, silhouette threshold)
Last revised: April 2026

Attack
------
The basin structure is driven by a handful of redundant or degenerate
features. Remove the top genes and the multiplicity vanishes.

Defence Strategy
----------------
1. ABLATION: Remove top-N genes, retrain. If multiplicity persists →
   not feature-specific.
2. NOISE INJECTION: Add 2σ Gaussian noise to top features. If persists →
   robust to perturbation.
3. PCA COMPRESSION: Reduce to 50/100 PCA components. If multiplicity
   vanishes → collinear feature geometry is the mechanistic precondition
   (confirmation of mechanism, not refutation of phenomenon).

Kill Condition
--------------
  Ablation AND noise survive → ATTACK KILLED on its own terms
  PCA collapse → mechanistic insight (collinearity precondition confirmed)

Data
----
Retrains LR models. Requires SLURM for reasonable runtime.
"""

import sys, json, random, argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import shap
from collections import Counter

def load_tcga_data(gz_path, top_n=1000):
    try:
        from tcga_xena_adapter import load_tcga_for_evoxplain
    except ImportError:
        sys.exit("[ERROR] tcga_xena_adapter.py not found.")
    X, y, fn = load_tcga_for_evoxplain(
        gz_path=gz_path, label_source="barcode",
        top_n=top_n, standardize=True, log2_transform=False)
    return X.astype(np.float64), y.astype(int), list(fn)

def seed_everything(s): random.seed(s); np.random.seed(s)
def cosine_sim(a, b): return float(1.0 - cosine(a, b))
def normalise(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)

def pairwise_cosines(vecs):
    n = len(vecs)
    if n < 2: return np.array([1.0])
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_sim(vecs[i], vecs[j]))
    return np.array(sims)

def pick_k_kmeans(vn, k_max=5, seed=0,
                  silhouette_threshold=0.15,
                  cosine_collapse_threshold=0.99):
    """Standard KMeans with silhouette threshold and cosine-collapse guard."""
    n = len(vn)
    if n < 4: return 1, np.zeros(n, dtype=int), vn.mean(axis=0, keepdims=True)

    sims = [cosine_sim(vn[i], vn[j]) for i in range(min(n, 100))
            for j in range(i+1, min(n, 100))]
    if sims and np.min(sims) >= cosine_collapse_threshold:
        return 1, np.zeros(n, dtype=int), vn.mean(axis=0, keepdims=True)

    bk, bs = 1, 0.0
    bl = np.zeros(n, dtype=int)
    bc = vn.mean(axis=0, keepdims=True)
    for k in range(2, min(k_max+1, n)):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        l = km.fit_predict(vn)
        if len(np.unique(l)) < 2: continue
        s = silhouette_score(vn, l)
        if s > bs and s >= silhouette_threshold:
            bs, bk, bl = s, k, l
            bc = np.array([vn[l==i].mean(axis=0) for i in range(k)])
    return bk, bl, bc

def get_boundary_indices(X_te, X_tr, y_tr, bk=200):
    seed_everything(123)
    ref = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=123)
    ref.fit(X_tr, y_tr)
    probs = ref.predict_proba(X_te)[:, 1]
    pool = np.arange(len(X_te))
    mask = (probs >= 0.45) & (probs <= 0.55)
    sel = pool[mask]
    if len(sel) < bk: sel = pool[np.argsort(np.abs(probs - 0.5))[:bk]]
    elif len(sel) > bk: sel = np.random.RandomState(123).choice(sel, bk, replace=False)
    return np.sort(sel)

def c_grid_values(n, c_min=0.001, c_max=1000.0):
    return np.logspace(np.log10(c_min), np.log10(c_max), n)

def compute_shap_vec(model, X_tr, X_b):
    ex = shap.LinearExplainer(model, X_tr)
    sv = ex.shap_values(X_b)
    if isinstance(sv, list): sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.array(sv)
    return np.mean(sv, axis=0) if sv.ndim == 2 else None

def run_cgrid_pipeline(X, y, split_seeds, feature_names, n_c_runs=20,
                       c_min=0.001, c_max=1000.0, base_seed=42,
                       lr_max_iter=1000, boundary_k=200, label=""):
    """Run C-grid LR + SHAP on given data with STRICT PER-SPLIT CLUSTERING."""
    c_values = c_grid_values(n_c_runs, c_min, c_max)
    
    split_metrics = []
    total_vecs = 0

    for i, ss in enumerate(split_seeds):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=ss, stratify=y)
        b_idx = get_boundary_indices(X_te, X_tr, y_tr, bk=boundary_k)
        X_b = X_te[b_idx]
        if len(X_b) == 0: continue

        vecs = []
        accs = []
        for r, C in enumerate(c_values):
            rs = base_seed + (ss * 10000) + r
            m = LogisticRegression(C=float(C), penalty="l2", solver="lbfgs",
                                   max_iter=lr_max_iter, random_state=rs, n_jobs=1)
            m.fit(X_tr, y_tr)
            accs.append(float(m.score(X_te, y_te)))
            sv = compute_shap_vec(m, X_tr, X_b)
            if sv is not None:
                vecs.append(sv)

        if len(vecs) < 2: continue
        total_vecs += len(vecs)

        vecs_n = normalise(np.array(vecs))
        cos = pairwise_cosines(vecs_n)
        
        # STRICT THRESHOLD: If minimum cosine similarity > 0.95, force k=1
        if len(cos) > 0 and (1.0 - np.min(cos)) < 0.05:
            k = 1
        else:
            k, _, _ = pick_k_kmeans(vecs_n, k_max=5, seed=0)
            
        split_metrics.append({
            "k": k,
            "acc": np.mean(accs),
            "cos_mean": np.mean(cos),
            "cos_std": np.std(cos),
            "neg_frac": np.mean(cos < 0)
        })

        if (i+1) % 5 == 0:
            print(f"    [{label}] {i+1}/{len(split_seeds)} splits done")

    if not split_metrics:
        return {"k_star_mean": 0, "n_vecs": 0, "error": "no vectors"}

    # Average metrics across independent splits
    avg_k = np.mean([m["k"] for m in split_metrics])
    avg_acc = np.mean([m["acc"] for m in split_metrics])
    avg_cos_mean = np.mean([m["cos_mean"] for m in split_metrics])
    avg_cos_std = np.mean([m["cos_std"] for m in split_metrics])
    avg_neg_frac = np.mean([m["neg_frac"] for m in split_metrics])

    result = {
        "k_star_mean": round(float(avg_k), 2),
        "n_vecs": total_vecs,
        "mean_acc": round(float(avg_acc), 4),
        "cosine_mean": round(float(avg_cos_mean), 4),
        "cosine_std": round(float(avg_cos_std), 4),
        "neg_frac": round(float(avg_neg_frac), 4),
    }

    print(f"    [{label}] Mean k*={avg_k:.2f}, acc={avg_acc:.4f}, "
          f"cos_mean={avg_cos_mean:.4f}, neg%={avg_neg_frac*100:.1f}%")

    return result

def get_top_gene_indices(output_dir_agg, feature_names, top_n=100,
                         split_start=800, split_end=900):
    gene_importance = np.zeros(len(feature_names))

    for seed in range(split_start, split_end):
        agg = Path(output_dir_agg) / f"split{seed}" / f"aggregate_split{seed}.npz"
        if not agg.exists(): continue
        data = np.load(agg, allow_pickle=True)
        c_key = "centroids_normed_shap"
        centroids = data[c_key] if c_key in data else data["centroids_normed"]
        for c in centroids:
            gene_importance += np.abs(c)

    top_idx = np.argsort(gene_importance)[::-1][:top_n]
    return top_idx

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TCGA Test 7: Data Degeneracy")
    parser.add_argument("--tcga_gz_path",    type=str,  default="data/tcga_RSEM_gene_tpm.gz")
    parser.add_argument("--tcga_top_n",      type=int,  default=1000)
    parser.add_argument("--output_dir_agg",  type=str,
                        default="results/final_freeze/tcga_lr_cgrid_shap_lime_100x100")
    parser.add_argument("--n_splits",        type=int,  default=10)
    parser.add_argument("--split_seed_start",type=int,  default=800)
    parser.add_argument("--n_c_runs",        type=int,  default=20)
    parser.add_argument("--c_min",           type=float,default=0.001)
    parser.add_argument("--c_max",           type=float,default=1000.0)
    parser.add_argument("--lr_max_iter",     type=int,  default=1000)
    parser.add_argument("--boundary_k",      type=int,  default=200)
    parser.add_argument("--ablation_n",      type=int,  default=100)
    parser.add_argument("--pca_components",  type=str,  default="50,100")
    parser.add_argument("--output_dir",      type=str,  default=".")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    split_seeds = list(range(args.split_seed_start, args.split_seed_start + args.n_splits))
    pca_dims = [int(d) for d in args.pca_components.split(",")]

    print("=" * 70)
    print("EvoXplain — Test 7: Data Degeneracy (TCGA LR-Cgrid)")
    print("  'Is multiplicity driven by a few redundant features?'")
    print("=" * 70)

    X, y, fnames = load_tcga_data(args.tcga_gz_path, args.tcga_top_n)
    top_idx = get_top_gene_indices(args.output_dir_agg, fnames, top_n=args.ablation_n)
    top_genes = [fnames[i].split(".")[0] for i in top_idx[:10]]

    results = {"config": vars(args)}

    # ARM A: BASELINE
    print(f"\n{'='*70}\nARM A: Baseline (all 1000 features)\n{'='*70}")
    res_a = run_cgrid_pipeline(X, y, split_seeds, fnames, n_c_runs=args.n_c_runs, 
                               c_min=args.c_min, c_max=args.c_max, lr_max_iter=args.lr_max_iter, 
                               boundary_k=args.boundary_k, label="Baseline")
    results["arm_a_baseline"] = res_a

    # ARM B: ABLATION
    print(f"\n{'='*70}\nARM B: Ablation (remove top {args.ablation_n} features)\n{'='*70}")
    keep_mask = np.ones(X.shape[1], dtype=bool)
    keep_mask[top_idx] = False
    X_ablated = X[:, keep_mask]
    fnames_ablated = [f for i, f in enumerate(fnames) if keep_mask[i]]
    from sklearn.preprocessing import StandardScaler
    X_ablated = StandardScaler().fit_transform(X_ablated)

    res_b = run_cgrid_pipeline(X_ablated, y, split_seeds, fnames_ablated, n_c_runs=args.n_c_runs, 
                               c_min=args.c_min, c_max=args.c_max, lr_max_iter=args.lr_max_iter, 
                               boundary_k=args.boundary_k, label=f"Ablated-{args.ablation_n}")
    results["arm_b_ablation"] = res_b

    # ARM C: NOISE INJECTION
    print(f"\n{'='*70}\nARM C: Noise injection (Gaussian noise on top {args.ablation_n} features)\n{'='*70}")
    rng = np.random.RandomState(42)
    X_noisy = X.copy()
    noise_std = 2.0 * np.std(X[:, top_idx], axis=0)
    X_noisy[:, top_idx] += rng.normal(0, noise_std, size=X[:, top_idx].shape)
    X_noisy = StandardScaler().fit_transform(X_noisy)

    res_c = run_cgrid_pipeline(X_noisy, y, split_seeds, fnames, n_c_runs=args.n_c_runs, 
                               c_min=args.c_min, c_max=args.c_max, lr_max_iter=args.lr_max_iter, 
                               boundary_k=args.boundary_k, label="Noisy")
    results["arm_c_noise"] = res_c

    # ARM D: PCA COMPRESSION
    results["arm_d_pca"] = {}
    for n_comp in pca_dims:
        print(f"\n{'='*70}\nARM D: PCA compression ({n_comp} components)\n{'='*70}")
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X)
        X_pca = StandardScaler().fit_transform(X_pca)
        pca_fnames = [f"PC{i}" for i in range(n_comp)]
        var_explained = float(np.sum(pca.explained_variance_ratio_))

        res_d = run_cgrid_pipeline(X_pca, y, split_seeds, pca_fnames, n_c_runs=args.n_c_runs, 
                                   c_min=args.c_min, c_max=args.c_max, lr_max_iter=args.lr_max_iter, 
                                   boundary_k=args.boundary_k, label=f"PCA-{n_comp}")
        res_d["var_explained"] = round(var_explained, 4)
        results["arm_d_pca"][str(n_comp)] = res_d

    # VERDICT
    print(f"\n{'='*70}\nTEST 7 VERDICT\n{'='*70}")

    # Threshold for defining "multiplicity persists" across splits: mean k* > 1.5
    baseline_mult = bool(res_a["k_star_mean"] > 1.5)
    ablation_mult = bool(res_b["k_star_mean"] > 1.5)
    noise_mult = bool(res_c["k_star_mean"] > 1.5)
    pca_mult = {d: bool(results["arm_d_pca"][str(d)]["k_star_mean"] > 1.5) for d in pca_dims}

    print(f"  Baseline (1000 features): Mean k*={res_a['k_star_mean']}  mult={baseline_mult}")
    print(f"  Ablation (top-{args.ablation_n} removed): Mean k*={res_b['k_star_mean']}  mult={ablation_mult}")
    print(f"  Noise injection:          Mean k*={res_c['k_star_mean']}  mult={noise_mult}")
    for d in pca_dims:
        print(f"  PCA-{d}:                   Mean k*={results['arm_d_pca'][str(d)]['k_star_mean']}  mult={pca_mult[d]}")

    survives_ablation = ablation_mult
    survives_noise = noise_mult
    survives_pca = any(pca_mult.values())

    pca_summary = ', '.join(f"PCA-{d}: Mean k*={results['arm_d_pca'][str(d)]['k_star_mean']}" for d in pca_dims)

    if baseline_mult and survives_ablation and survives_noise and survives_pca:
        tier = "ATTACK KILLED"
        verdict = (
            f"ATTACK KILLED — Multiplicity persists through feature ablation "
            f"(k*={res_b['k_star_mean']:.1f}), noise injection "
            f"(k*={res_c['k_star_mean']:.1f}), and PCA compression. "
            f"Not driven by a specific redundant subset."
        )
    elif baseline_mult and survives_ablation and survives_noise and not survives_pca:
        tier = "ATTACK KILLED"
        verdict = (
            f"ATTACK KILLED — Feature-specific degeneracy refuted: ablation "
            f"(k*={res_b['k_star_mean']:.1f}) and noise injection "
            f"(k*={res_c['k_star_mean']:.1f}) both preserve multiplicity. "
            f"PCA collapses multiplicity ({pca_summary}) — this confirms that "
            f"collinear feature geometry is the mechanistic precondition for "
            f"C-dependent weight redistribution, consistent with the EvoXplain thesis. "
            f"PCA destroys collinearity by construction; the collapse is expected, "
            f"not a refutation."
        )
    elif baseline_mult and (survives_ablation or survives_noise):
        tier = "PARTIAL"
        verdict = f"PARTIAL — Multiplicity survives some perturbations but not all."
    elif baseline_mult:
        tier = "INCONCLUSIVE"
        verdict = "INCONCLUSIVE — Baseline shows multiplicity but all perturbations destroy it."
    else:
        tier = "INCONCLUSIVE"
        verdict = "INCONCLUSIVE — Baseline fails to show multiplicity."

    print(f"\n  >>> {verdict}")
    print("=" * 70)

    results["test_id"] = "test7_data_degeneracy"
    results["version"] = "v2"
    results["evidence"] = {
        "baseline_mult": baseline_mult,
        "survives_ablation": survives_ablation,
        "survives_noise": survives_noise,
        "survives_pca": survives_pca,
    }
    results["verdict_tier"] = tier
    results["verdict_text"] = verdict

    json_path = Path(args.output_dir) / "tcga_test7_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {json_path}")


if __name__ == "__main__":
    main()
