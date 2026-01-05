#!/usr/bin/env python3
"""
evoxplain_core_engine.py - EvoXplain HPC SHAP Runner + Aggregator


1. SHAP 3D array indexing for TreeExplainer (SHAP 0.42+)
2. RF hyperparameters saved in aggregate NPZ files  
3. RF hyperparameters used in disagreement analysis
4. Proper split_seed handling in disagreement for universal mode

Clean refactor of EvoXplain HPC SHAP runner + aggregator for multiple train/test splits.
Supports both chunk-based parallel execution and aggregate + clustering + viz helpers.
"""

import os
import json
import time
import math
import argparse
import numpy as np
from pathlib import Path

# Optional imports will be done inside functions to keep CLI lightweight


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(dataset_name: str):
    """
    Load supported datasets and return (X, y, feature_names).
    """
    if dataset_name.lower() in ("breast_cancer", "breast-cancer", "bc"):
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = list(data.feature_names)
        # Paper Section 6.2: "Linear models are trained on standardised features"
        # Scale all features for consistency (needed for LogReg convergence)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y, feature_names

    if dataset_name.lower() in ("compas",):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        # Load COMPAS dataset
        compas_path = "data/compas-scores-two-years.csv"
        if not os.path.exists(compas_path):
            raise FileNotFoundError(f"COMPAS dataset not found at {compas_path}")
        
        df = pd.read_csv(compas_path)
        
        # Standard COMPAS preprocessing
        # Filter to relevant rows
        df = df[(df['days_b_screening_arrest'] <= 30) & 
                (df['days_b_screening_arrest'] >= -30) &
                (df['is_recid'] != -1) &
                (df['c_charge_degree'] != 'O') &
                (df['score_text'] != 'N/A')]
        
        # Select features (standard subset)
        feature_cols = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 
                       'sex', 'priors_count', 'days_b_screening_arrest', 
                       'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
        
        # Keep only columns that exist in the dataset
        feature_cols = [col for col in feature_cols if col in df.columns]
        df = df[feature_cols]
        
        # Target: two_year_recid
        y = df['two_year_recid'].values
        
        # Features: one-hot encode categorical variables
        categorical_cols = ['c_charge_degree', 'race', 'age_cat', 'score_text', 'sex']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        df_encoded = pd.get_dummies(df.drop(columns=['two_year_recid', 'is_recid'] + 
                                                     [c for c in ['c_jail_in', 'c_jail_out'] if c in df.columns], 
                                                     errors='ignore'), 
                                    columns=categorical_cols, drop_first=True)
        
        X = df_encoded.values.astype(float)
        feature_names = list(df_encoded.columns)
        
        # Standardize ONLY numerical features, not one-hot encoded
        numerical_features = ['age', 'priors_count', 'days_b_screening_arrest', 'decile_score']
        numerical_indices = [i for i, name in enumerate(feature_names) if any(nf in name for nf in numerical_features)]
        
        if len(numerical_indices) > 0:
            scaler = StandardScaler()
            X[:, numerical_indices] = scaler.fit_transform(X[:, numerical_indices])
        
        return X, y, feature_names

    if dataset_name.lower() in ("adult", "adult_income", "adult-income"):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        # Load Adult dataset
        adult_path = "data/adult.csv"
        if not os.path.exists(adult_path):
            raise FileNotFoundError(f"Adult dataset not found at {adult_path}")
        
        df = pd.read_csv(adult_path)
        
        # Handle column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Target: income (>50K = 1, <=50K = 0)
        if 'income' in df.columns:
            y = (df['income'].str.strip().str.contains('>50K')).astype(int).values
            df = df.drop(columns=['income'])
        else:
            raise ValueError("Adult dataset missing 'income' column")
        
        # Identify categorical and numerical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # One-hot encode categorical
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        X = df_encoded.values.astype(float)
        feature_names = list(df_encoded.columns)
        
        # Standardize numerical features
        numerical_indices = [i for i, name in enumerate(feature_names) if name in numerical_cols]
        if len(numerical_indices) > 0:
            scaler = StandardScaler()
            X[:, numerical_indices] = scaler.fit_transform(X[:, numerical_indices])
        
        return X, y, feature_names

    raise ValueError(f"Unknown dataset: {dataset_name}")


def make_model(args, C_value=None, rf_params=None):
    """
    Construct model according to args.model.
    """
    model_name = args.model.lower()

    if model_name in ("logreg", "logistic", "lr", "logistic_regression"):
        from sklearn.linear_model import LogisticRegression
        C_use = C_value if C_value is not None else args.C
        model = LogisticRegression(
            C=C_use,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=args.seed
        )
        return model

    if model_name in ("rf", "random_forest", "randomforest"):
        from sklearn.ensemble import RandomForestClassifier
        params = dict(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth if args.rf_max_depth > 0 else None,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=args.rf_max_features,
            bootstrap=True,
            random_state=args.seed,
            n_jobs=1,
        )
        if rf_params is not None:
            params.update(rf_params)
        return RandomForestClassifier(**params)

    raise ValueError(f"Unknown model: {args.model}")


def sample_C(args, rng):
    """
    Sample regularization C according to args.c_mode.
    """
    if args.c_mode == "fixed":
        return args.C
    if args.c_mode == "varied":
        # log-uniform in [c_min, c_max]
        lo = math.log10(args.c_min)
        hi = math.log10(args.c_max)
        return 10 ** rng.uniform(lo, hi)
    raise ValueError(f"Unknown c_mode: {args.c_mode}")


def compute_shap_importance(model, X_train, X_test, feature_names, args):
    """
    Compute SHAP importance vector.
    Returns a 1D vector of length d (number of features).
    
    Updated for SHAP 0.50.0 compatibility:
    - LinearExplainer: use maskers.Independent or feature_perturbation="interventional"
    - TreeExplainer: handles 3D output (n_samples, n_features, n_classes)
    """
    import shap

    # Choose explainer based on model family
    model_name = args.model.lower()

    if model_name in ("logreg", "logistic", "lr", "logistic_regression"):
        # LinearExplainer with masker for SHAP 0.50.0+
        # Use Independent masker which is equivalent to old "independent" feature_perturbation
        try:
            masker = shap.maskers.Independent(X_train)
            explainer = shap.LinearExplainer(model, masker)
        except (TypeError, AttributeError):
            # Fallback for older SHAP versions
            explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_test)
    elif model_name in ("rf", "random_forest", "randomforest"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        # Fallback (could be slow)
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

    # shap_values can be list (per class) or array
    if isinstance(shap_values, list):
        # Old SHAP API: binary -> list of 2 arrays, each (n_samples, n_features)
        # Use the positive class by default if available
        if len(shap_values) > 1:
            sv = shap_values[1]
        else:
            sv = shap_values[0]
    else:
        sv = shap_values

    sv = np.array(sv)

    # Reduce to (n_samples, n_features)
    # SHAP 0.42+ TreeExplainer returns (n_samples, n_features, n_classes)
    if sv.ndim == 3:
        # Shape is (n_samples, n_features, n_classes) -> select class 1 (positive class)
        if sv.shape[2] > 1:
            sv = sv[:, :, 1]  # All samples, all features, class 1
        else:
            sv = sv[:, :, 0]  # All samples, all features, only class

    if sv.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

    # Global importance vector: mean absolute SHAP per feature
    importance = np.mean(np.abs(sv), axis=0)

    if importance.shape[0] != len(feature_names):
        # In case feature_names not provided correctly
        feature_names = [f"f{i}" for i in range(importance.shape[0])]

    return importance


def train_and_explain(run_id, X_train, y_train, X_test, y_test, feature_names, args, rng):
    """
    Train model and compute:
    - accuracy on test
    - SHAP global importance vector
    """
    from sklearn.metrics import accuracy_score

    C_val = None
    rf_params = None

    if args.model.lower() in ("logreg", "logistic", "lr", "logistic_regression"):
        C_val = sample_C(args, rng)

    if args.model.lower() in ("rf", "random_forest", "randomforest") and args.rf_varied:
        # Example varied RF params; adjust as needed
        rf_params = dict(
            n_estimators=int(rng.integers(args.rf_n_estimators_min, args.rf_n_estimators_max + 1)),
            max_depth=int(rng.integers(args.rf_max_depth_min, args.rf_max_depth_max + 1)) if args.rf_max_depth_max > 0 else None,
            max_features=float(rng.uniform(args.rf_max_features_min, args.rf_max_features_max)),
            min_samples_split=int(rng.integers(args.rf_min_samples_split_min, args.rf_min_samples_split_max + 1)),
            min_samples_leaf=int(rng.integers(args.rf_min_samples_leaf_min, args.rf_min_samples_leaf_max + 1)),
        )

    model = make_model(args, C_value=C_val, rf_params=rf_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    importance = compute_shap_importance(model, X_train, X_test, feature_names, args)

    meta = {
        "run_id": int(run_id),
        "acc": acc,
    }
    if C_val is not None:
        meta["C"] = float(C_val)
    if rf_params is not None:
        meta.update({f"rf_{k}": v for k, v in rf_params.items()})

    return meta, importance, model


def run_chunk(args):
    """
    Execute one chunk of runs for a given split_seed.
    Saves per-run JSON + importance vectors as .npy.
    """
    from sklearn.model_selection import train_test_split

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    X, y, feature_names = load_dataset(args.dataset)

    # Split defined by split_seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.split_seed, stratify=y
    )

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Determine run indices for this chunk
    start = args.chunk_id * args.chunk_size
    end = min(args.n_runs, start + args.chunk_size)

    print(f"[chunk {args.chunk_id}] runs {start}..{end-1} (split_seed={args.split_seed})")

    # Save split indices (optional)
    if args.save_split:
        np.save(out_dir / f"split_{args.split_seed}_Xtest.npy", X_test)
        np.save(out_dir / f"split_{args.split_seed}_ytest.npy", y_test)

    metas = []
    importances = []

    for run_id in range(start, end):
        # Derive a per-run seed to diversify training randomness
        run_seed = args.seed + 100000 * int(args.split_seed) + int(run_id)
        seed_everything(run_seed)
        rng_run = np.random.default_rng(run_seed)

        meta, imp, _model = train_and_explain(run_id, X_train, y_train, X_test, y_test, feature_names, args, rng_run)
        metas.append(meta)
        importances.append(imp)

        # Write per-run meta
        if args.write_per_run:
            save_json(meta, out_dir / f"meta_split{args.split_seed}_run{run_id}.json")

    importances = np.array(importances, dtype=float)

    # Write chunk arrays
    np.save(out_dir / f"importance_split{args.split_seed}_chunk{args.chunk_id}.npy", importances)
    save_json(metas, out_dir / f"meta_split{args.split_seed}_chunk{args.chunk_id}.json")

    print(f"[chunk {args.chunk_id}] saved to {out_dir}")


def aggregate_split(args, split_seed):
    """
    Aggregate all chunks for one split_seed into a single NPZ file.
    FIX: Now saves RF hyperparameters in the NPZ for disagreement analysis.
    """
    out_dir = Path(args.output_dir)
    metas = []
    imps = []

    # Find chunk files
    chunk_files = sorted(out_dir.glob(f"importance_split{split_seed}_chunk*.npy"))
    meta_files = sorted(out_dir.glob(f"meta_split{split_seed}_chunk*.json"))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found for split_seed={split_seed} in {out_dir}")

    for cf in chunk_files:
        imps.append(np.load(cf))

    for mf in meta_files:
        metas.extend(load_json(mf))

    imps = np.vstack(imps)

    # Sort metas by run_id just in case
    metas = sorted(metas, key=lambda m: m["run_id"])

    # Align to run order
    run_indices = np.array([m["run_id"] for m in metas], dtype=int)
    accs = np.array([m["acc"] for m in metas], dtype=float)

    # C array if present (for logistic regression)
    c_values = None
    if "C" in metas[0]:
        c_values = np.array([m.get("C", np.nan) for m in metas], dtype=float)

    # FIX: Save RF hyperparameters if present
    rf_params_arrays = {}
    rf_param_keys = ["rf_n_estimators", "rf_max_depth", "rf_max_features", 
                     "rf_min_samples_split", "rf_min_samples_leaf"]
    
    for key in rf_param_keys:
        if key in metas[0]:
            # Handle None values (e.g., max_depth=None)
            values = []
            for m in metas:
                val = m.get(key, np.nan)
                if val is None:
                    val = -1  # Use -1 to represent None for max_depth
                values.append(val)
            rf_params_arrays[key] = np.array(values, dtype=float)

    # Build save dict
    save_dict = {
        "run_indices": run_indices,
        "accs": accs,
        "c_values": c_values if c_values is not None else np.array([]),
        "importance": imps,
    }
    save_dict.update(rf_params_arrays)

    np.savez(out_dir / f"aggregate_split{split_seed}.npz", **save_dict)

    print(f"[aggregate] saved aggregate_split{split_seed}.npz (n={len(metas)}, keys={list(save_dict.keys())})")


def pick_best_k_kmeans(X, k_min=2, k_max=8, seed=0):
    """
    Select k by silhouette score for k in [k_min, k_max].
    Returns (best_k, labels, silhouette_by_k)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    if n < k_min:
        return 1, np.zeros(n, dtype=int), {}

    best_k = 1
    best_score = -1.0
    best_labels = np.zeros(n, dtype=int)
    sil_by_k = {}

    # Degenerate check: if all vectors nearly identical, return k=1
    # This avoids silhouette errors and aligns with the "collapse" case.
    if np.allclose(X, X.mean(axis=0), atol=1e-12):
        return 1, np.zeros(n, dtype=int), {}

    for k in range(k_min, min(k_max, n) + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(X)
        # If clustering collapses (rare), skip
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(X, labels, metric="euclidean")
        sil_by_k[k] = float(score)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_k == 1:
        return 1, np.zeros(n, dtype=int), sil_by_k

    return best_k, best_labels, sil_by_k


def compute_entropy(labels, k):
    """
    Normalized Shannon entropy over cluster membership.
    """
    if k <= 1:
        return 0.0
    counts = np.array([(labels == i).sum() for i in range(k)], dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    Hn = float(H / np.log2(k))
    return Hn


def aggregate_and_cluster(args, split_seeds):
    """
    Stack multiple split aggregates into a "universal" NPZ file and cluster in explanation space.
    Saves:
      - universal_importance.npy (raw)
      - universal_normed_importance.npy (centered + l2 normalized)
      - universal_labels.npy
      - universal_summary.json (k, entropy, cluster sizes, etc.)
    
    FIX: Now loads and passes RF hyperparameters for disagreement analysis.
    """
    out_dir = Path(args.output_dir)
    all_importance = []
    all_accs = []
    all_split = []
    all_run = []
    all_C = []
    
    # FIX: Collect RF hyperparameters
    all_rf_params = {
        "rf_n_estimators": [],
        "rf_max_depth": [],
        "rf_max_features": [],
        "rf_min_samples_split": [],
        "rf_min_samples_leaf": [],
    }
    has_rf_params = False

    for s in split_seeds:
        agg_path = out_dir / f"aggregate_split{s}.npz"
        if not agg_path.exists():
            raise FileNotFoundError(f"Missing {agg_path}; run aggregate_split first.")
        data = np.load(agg_path, allow_pickle=True)
        imp = data["importance"]
        acc = data["accs"]
        run_idx = data["run_indices"]

        c_vals = data["c_values"]
        if c_vals.size == 0:
            c_vals = np.full(len(run_idx), np.nan)

        all_importance.append(imp)
        all_accs.append(acc)
        all_split.append(np.full(len(run_idx), int(s)))
        all_run.append(run_idx.astype(int))
        all_C.append(c_vals.astype(float))
        
        # FIX: Load RF hyperparameters if present
        for key in all_rf_params.keys():
            if key in data:
                has_rf_params = True
                all_rf_params[key].append(data[key])
            else:
                all_rf_params[key].append(np.full(len(run_idx), np.nan))

    importance = np.vstack(all_importance).astype(float)
    accs = np.concatenate(all_accs).astype(float)
    split_ids = np.concatenate(all_split).astype(int)
    run_indices = np.concatenate(all_run).astype(int)
    c_values = np.concatenate(all_C).astype(float)
    
    # FIX: Concatenate RF params
    rf_params_universal = {}
    if has_rf_params:
        for key in all_rf_params.keys():
            rf_params_universal[key] = np.concatenate(all_rf_params[key]).astype(float)

    # Normalisation: center by mean and L2-normalise each run vector
    mu = importance.mean(axis=0)
    centered = importance - mu
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed_importance = centered / norms

    # Cluster in normed space
    best_k, labels, sil_by_k = pick_best_k_kmeans(normed_importance, k_min=2, k_max=args.k_max, seed=args.seed)
    entropy = compute_entropy(labels, best_k)

    # Compute cluster centroids in normed space
    centroids = []
    cluster_sizes = []
    for i in range(best_k):
        mask = labels == i
        cluster_sizes.append(int(mask.sum()))
        if mask.sum() > 0:
            centroids.append(normed_importance[mask].mean(axis=0))
        else:
            centroids.append(np.zeros(normed_importance.shape[1], dtype=float))
    centroids = np.array(centroids)

    # Save outputs
    np.save(out_dir / "universal_importance.npy", importance)
    np.save(out_dir / "universal_normed_importance.npy", normed_importance)
    np.save(out_dir / "universal_labels.npy", labels.astype(int))
    np.save(out_dir / "universal_accs.npy", accs)
    np.save(out_dir / "universal_split_ids.npy", split_ids)
    np.save(out_dir / "universal_run_indices.npy", run_indices)
    np.save(out_dir / "universal_c_values.npy", c_values)
    np.save(out_dir / "universal_centroids.npy", centroids)
    
    # FIX: Save RF params
    for key, values in rf_params_universal.items():
        np.save(out_dir / f"universal_{key}.npy", values)

    summary = {
        "n_total": int(len(labels)),
        "best_k": int(best_k),
        "entropy": float(entropy),
        "cluster_sizes": cluster_sizes,
        "silhouette_by_k": sil_by_k,
    }
    save_json(summary, out_dir / "universal_summary.json")

    print("[universal] saved universal_* files")
    print(f"[universal] best_k={best_k}, entropy={entropy:.4f}, cluster_sizes={cluster_sizes}")

    # Optionally inspect disagreements
    if args.disagreement:
        # FIX: Pass RF params to disagreement analysis
        inspect_disagreements(
            run_indices, importance, c_values, labels, best_k, args, 
            feature_names=None, split_ids=split_ids, rf_params=rf_params_universal
        )

    return summary


def inspect_disagreements(run_indices, importance, c_values, labels, best_k, args, 
                          feature_names, split_ids=None, rf_params=None):
    """
    Inspect disagreements (Dataset Agnostic).
    
    FIX: Now accepts and uses RF hyperparameters to recreate representative models.
    FIX: Uses the correct split_seed for each representative run.
    """
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    if best_k == 1:
        print("Only 1 cluster found - no disagreement analysis possible.")
        return None
    
    cluster_sizes = [(i, (labels == i).sum()) for i in range(best_k)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    selected_clusters = [c[0] for c in cluster_sizes[:min(3, best_k)]]
    
    print(f"\nAnalyzing disagreements across clusters: {selected_clusters}")
    
    # Load dataset
    X, y, feature_names_local = load_dataset(args.dataset)
    if feature_names is None:
        feature_names = feature_names_local
    
    # For universal mode, we need to pick a split_seed for the representative
    # Use the first split_seed from args or the most common one
    if split_ids is not None:
        # Use the split_seed of each representative run
        pass  # Will be handled per-cluster below
    
    # Train representative models for each cluster
    rep_runs = []
    rep_C = []
    rep_rf_params = []
    rep_split_seeds = []
    rep_models = []
    
    for cluster_id in selected_clusters:
        mask = labels == cluster_id
        cluster_indices = run_indices[mask]
        cluster_C = c_values[mask]
        
        # Reconstruct the same normalised explanation space used for clustering
        mu = importance.mean(axis=0)
        centered = importance - mu
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed_importance = centered / norms

        cluster_normed = normed_importance[mask]

        # Closest to the cluster centroid in cosine distance (paper method)
        from scipy.spatial.distance import cosine
        centroid = cluster_normed.mean(axis=0)
        distances = np.array([cosine(imp, centroid) for imp in cluster_normed])
        rep_idx_in_cluster = int(np.argmin(distances))
        
        # Get the global index of this representative
        global_idx = np.where(mask)[0][rep_idx_in_cluster]
        
        rep_runs.append(int(cluster_indices[rep_idx_in_cluster]))
        rep_C.append(float(cluster_C[rep_idx_in_cluster]) if not np.isnan(cluster_C[rep_idx_in_cluster]) else None)
        
        # FIX: Get split_seed for this representative
        if split_ids is not None:
            rep_split_seed = int(split_ids[global_idx])
        else:
            rep_split_seed = args.split_seed
        rep_split_seeds.append(rep_split_seed)
        
        # FIX: Get RF hyperparameters for this representative
        if rf_params is not None and len(rf_params) > 0:
            rep_rf = {}
            for key in rf_params.keys():
                val = rf_params[key][global_idx]
                # Convert -1 back to None for max_depth
                if key == "rf_max_depth" and val == -1:
                    val = None
                elif not np.isnan(val):
                    # All RF int params: n_estimators, max_depth, min_samples_split, min_samples_leaf
                    if key in ["rf_n_estimators", "rf_max_depth", "rf_min_samples_split", "rf_min_samples_leaf"]:
                        val = int(val)
                    else:
                        val = float(val)
                else:
                    val = None
                rep_rf[key.replace("rf_", "")] = val
            rep_rf_params.append(rep_rf)
        else:
            rep_rf_params.append(None)
        
        # FIX: Recreate train/test split with correct split_seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=rep_split_seed, stratify=y
        )
        
        # Train representative model with correct hyperparameters
        run_seed = args.seed + 100000 * rep_split_seed + int(rep_runs[-1])
        seed_everything(run_seed)
        
        # FIX: Use the representative's actual hyperparameters
        if args.model.lower() in ("logreg", "logistic", "lr", "logistic_regression"):
            model = make_model(args, C_value=rep_C[-1])
        elif args.model.lower() in ("rf", "random_forest", "randomforest") and rep_rf_params[-1] is not None:
            model = make_model(args, rf_params=rep_rf_params[-1])
        else:
            model = make_model(args)
        
        model.fit(X_train, y_train)
        rep_models.append(model)
        
        print(f"Cluster {cluster_id}: rep run {rep_runs[-1]}, split {rep_split_seed}, "
              f"C={rep_C[-1]}, rf_params={rep_rf_params[-1]}")
    
    # Use the first representative's split for computing disagreements
    # (all test sets should be from the same split for fair comparison)
    # FIX: Use the most common split_seed among representatives
    if len(set(rep_split_seeds)) > 1:
        print(f"Warning: Representatives from different splits: {rep_split_seeds}")
        print("Using first split for disagreement computation.")
    
    main_split_seed = rep_split_seeds[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=main_split_seed, stratify=y
    )
    
    # Compute probability disagreements
    print("\nComputing disagreements...")
    def _positive_class_proba(model, X):
        """Return P(y=1|x) robustly (matches paper definition)."""
        proba = model.predict_proba(X)
        classes = getattr(model, "classes_", None)
        if classes is not None and proba.ndim == 2 and proba.shape[1] > 1:
            if 1 in classes:
                pos_idx = list(classes).index(1)
            else:
                pos_idx = -1
            return proba[:, pos_idx]
        if proba.ndim == 2:
            return proba[:, -1]
        return proba

    probs = np.array([_positive_class_proba(model, X_test) for model in rep_models])
    disagreement = probs.max(axis=0) - probs.min(axis=0)
    sorted_indices = np.argsort(disagreement)[::-1]
    
    # Create report - include ALL instances, not just top 20
    report = []
    for idx in sorted_indices:
        instance_probs = {f"cluster_{selected_clusters[i]}": float(probs[i, idx]) for i in range(len(selected_clusters))}
        report.append({
            "test_index": int(idx),
            "true_label": int(y_test[idx]),
            "disagreement": float(disagreement[idx]),
            **instance_probs
        })
    
    df = pd.DataFrame(report)
    out_path = Path(args.output_dir) / f"disagreement_report_split{main_split_seed}.csv"
    df.to_csv(out_path, index=False)
    
    print(f"Saved disagreement report to {out_path}")
    print(f"\nDisagreement statistics:")
    print(f"  Max disagreement: {disagreement.max():.4f}")
    print(f"  Mean disagreement: {disagreement.mean():.4f}")
    print(f"  Instances with >0.1 disagreement: {(disagreement > 0.1).sum()}")
    print(f"  Instances with >0.2 disagreement: {(disagreement > 0.2).sum()}")
    print("\nTop 10 disagreement instances:")
    print(df.head(10))
    
    return df


def build_arg_parser():
    p = argparse.ArgumentParser(description="EvoXplain core engine (HPC SHAP splits + aggregate + cluster).")

    p.add_argument("--dataset", type=str, required=True, help="Dataset name (breast_cancer, compas, adult, ...)")
    p.add_argument("--model", type=str, default="logreg", help="Model (logreg, rf)")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")

    # Split + runs
    p.add_argument("--split_seed", type=int, default=101, help="Train/test split seed")
    p.add_argument("--test_size", type=float, default=0.3, help="Test set fraction")
    p.add_argument("--n_runs", type=int, default=1000, help="Total runs per split")
    p.add_argument("--chunk_id", type=int, default=0, help="Chunk id (0..)")
    p.add_argument("--chunk_size", type=int, default=100, help="Runs per chunk")

    p.add_argument("--write_per_run", action="store_true", help="Write per-run JSON meta")
    p.add_argument("--save_split", action="store_true", help="Save split Xtest/ytest arrays")

    # Logistic regression C variation
    p.add_argument("--c_mode", type=str, default="fixed", choices=["fixed", "varied"])
    p.add_argument("--C", type=float, default=1.0, help="Fixed C value")
    p.add_argument("--c_min", type=float, default=1e-2, help="Min C (varied)")
    p.add_argument("--c_max", type=float, default=1e2, help="Max C (varied)")

    # RF params
    p.add_argument("--rf_n_estimators", type=int, default=200)
    p.add_argument("--rf_max_depth", type=int, default=-1)
    p.add_argument("--rf_min_samples_split", type=int, default=2)
    p.add_argument("--rf_min_samples_leaf", type=int, default=1)
    p.add_argument("--rf_max_features", type=float, default=0.7)

    p.add_argument("--rf_varied", action="store_true")
    p.add_argument("--rf_n_estimators_min", type=int, default=50)
    p.add_argument("--rf_n_estimators_max", type=int, default=600)
    p.add_argument("--rf_max_depth_min", type=int, default=1)
    p.add_argument("--rf_max_depth_max", type=int, default=30)
    p.add_argument("--rf_max_features_min", type=float, default=0.2)
    p.add_argument("--rf_max_features_max", type=float, default=1.0)
    p.add_argument("--rf_min_samples_split_min", type=int, default=2)
    p.add_argument("--rf_min_samples_split_max", type=int, default=20)
    p.add_argument("--rf_min_samples_leaf_min", type=int, default=1)
    p.add_argument("--rf_min_samples_leaf_max", type=int, default=10)

    # Clustering / universal
    p.add_argument("--k_max", type=int, default=8, help="Max k for silhouette selection")
    p.add_argument("--disagreement", action="store_true", help="Compute disagreement report on representative basins")

    # Mode
    p.add_argument("--mode", type=str, required=True, choices=["chunk", "aggregate_split", "aggregate_universal"])

    # For universal aggregation
    p.add_argument("--split_seeds", type=str, default="", help="Comma-separated split seeds for universal aggregation")

    return p


def main():
    args = build_arg_parser().parse_args()

    if args.mode == "chunk":
        run_chunk(args)
        return

    if args.mode == "aggregate_split":
        aggregate_split(args, args.split_seed)
        return

    if args.mode == "aggregate_universal":
        if not args.split_seeds.strip():
            raise ValueError("--split_seeds is required for aggregate_universal")
        split_seeds = [int(s.strip()) for s in args.split_seeds.split(",") if s.strip()]
        aggregate_and_cluster(args, split_seeds)
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
