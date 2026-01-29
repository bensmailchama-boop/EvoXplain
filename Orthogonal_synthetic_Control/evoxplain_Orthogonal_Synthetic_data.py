#!/usr/bin/env python3
"""
evoxplain_engine_synthetic.py - EvoXplain HPC SHAP Runner + Aggregator + Synthetic Control

MODIFIED VERSION WITH k=1 NULL HYPOTHESIS:
- Formal statistical testing for k=1 as null hypothesis
- Multiple criteria: Silhouette threshold, BIC/AIC, Variance Ratio (Calinski-Harabasz)
- Gap statistic option for rigorous null model comparison
- Includes 'SYNTH_ORTHO' dataset generation (uncorrelated features).
- Runs control experiments to test if multiplicity arises without correlation.
- Includes specific interpretation printouts for synthetic benchmarks.

Original Features:
1. SHAP 3D array indexing for TreeExplainer (SHAP 0.42+)
2. RF hyperparameters saved in aggregate NPZ files  
3. RF hyperparameters used in disagreement analysis
4. Proper split_seed handling in disagreement for universal mode
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


def make_synth_ortho_dataset(n_samples, n_features, snr, seed):
    """
    Generate synthetic dataset with orthogonal (uncorrelated) features.
    X ~ N(0, I)
    y ~ Bernoulli(sigmoid(X @ w + noise))
    """
    from scipy.special import expit
    
    # Local rng for dataset generation stability regardless of training seeds
    rng = np.random.default_rng(seed)
    
    # 1. Generate uncorrelated features X ~ N(0, 1)
    X = rng.standard_normal((n_samples, n_features))
    
    # Check correlations (sanity check)
    if n_samples > 1:
        corr_matrix = np.corrcoef(X, rowvar=False)
        np.fill_diagonal(corr_matrix, 0)
        max_corr = np.max(np.abs(corr_matrix))
        print(f"[SYNTH_ORTHO] Max off-diagonal feature correlation: {max_corr:.4f}")
    
    # 2. Generate Ground Truth weights w ~ N(0, 1)
    # Fixed linear relationship implies feature importance is strictly defined by w
    w = rng.standard_normal(n_features)
    
    # 3. Generate labels with noise
    # Signal
    signal = X @ w
    
    # Noise scale determined by SNR
    # signal_var approx n_features (since X, w ~ N(0,1))
    # we want var(signal) / var(noise) = snr
    # var(noise) = n_features / snr
    noise_std = np.sqrt(n_features / snr)
    noise = rng.normal(0, noise_std, size=n_samples)
    
    logits = signal + noise
    probs = expit(logits)
    y = rng.binomial(1, probs)
    
    feature_names = [f"f{i}" for i in range(n_features)]
    
    return X, y, feature_names


def load_dataset(dataset_name: str, args=None):
    """
    Load supported datasets and return (X, y, feature_names).
    Now accepts 'args' to configure synthetic generation.
    """
    # ---------------------------------------------------------
    # SYNTHETIC ORTHOGONAL DATASET
    # ---------------------------------------------------------
    if dataset_name.upper() == "SYNTH_ORTHO":
        if args is None:
            raise ValueError("args must be provided for SYNTH_ORTHO to read n_samples/seed params")
        
        # Use a fixed seed for dataset generation so all chunks see the same data
        # We use args.seed (global seed) + 999 to differentiate from split/run seeds
        dataset_seed = args.seed + 999
        
        print(f"Generating SYNTH_ORTHO: N={args.synth_n_samples}, D={args.synth_n_features}, SNR={args.synth_snr}")
        X, y, feature_names = make_synth_ortho_dataset(
            n_samples=args.synth_n_samples,
            n_features=args.synth_n_features,
            snr=args.synth_snr,
            seed=dataset_seed
        )
        
        # Standardize features (even though already N(0,1), good practice for consistency)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y, feature_names

    # ---------------------------------------------------------
    # REAL DATASETS
    # ---------------------------------------------------------
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
        
        adult_path = "data/adult.csv"
        if not os.path.exists(adult_path):
            raise FileNotFoundError(f"Adult dataset not found at {adult_path}")
        
        df = pd.read_csv(adult_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Target
        target_col = 'income' if 'income' in df.columns else df.columns[-1]
        y = (df[target_col].str.strip() == '>50K').astype(int).values
        
        # Drop target
        df = df.drop(columns=[target_col])
        
        # Identify categorical vs numerical
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # One-hot encode categorical
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        X = df_encoded.values.astype(float)
        feature_names = list(df_encoded.columns)
        
        # Standardize numerical features
        numerical_indices = [i for i, name in enumerate(feature_names) 
                           if any(name.startswith(nc) for nc in numerical_cols)]
        
        if len(numerical_indices) > 0:
            scaler = StandardScaler()
            X[:, numerical_indices] = scaler.fit_transform(X[:, numerical_indices])
        
        return X, y, feature_names

    raise ValueError(f"Unknown dataset: {dataset_name}")


def make_model(args, C_value=None, rf_params=None):
    """
    Factory to create model instances with optional hyperparameter overrides.
    """
    if args.model.lower() in ("logreg", "logistic", "lr", "logistic_regression"):
        from sklearn.linear_model import LogisticRegression
        C = C_value if C_value is not None else args.C
        return LogisticRegression(C=C, max_iter=5000, solver="lbfgs", random_state=args.seed)
    
    if args.model.lower() in ("rf", "random_forest", "randomforest"):
        from sklearn.ensemble import RandomForestClassifier
        
        if rf_params is not None:
            # Use provided RF hyperparameters
            n_estimators = rf_params.get("n_estimators", args.rf_n_estimators)
            max_depth = rf_params.get("max_depth", args.rf_max_depth)
            max_features = rf_params.get("max_features", args.rf_max_features)
            min_samples_split = rf_params.get("min_samples_split", args.rf_min_samples_split)
            min_samples_leaf = rf_params.get("min_samples_leaf", args.rf_min_samples_leaf)
        else:
            n_estimators = args.rf_n_estimators
            max_depth = args.rf_max_depth if args.rf_max_depth > 0 else None
            max_features = args.rf_max_features
            min_samples_split = args.rf_min_samples_split
            min_samples_leaf = args.rf_min_samples_leaf
        
        if max_depth == -1:
            max_depth = None
            
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1
        )
    
    raise ValueError(f"Unknown model: {args.model}")


def run_chunk(args):
    """
    Run a chunk of experiments and save SHAP importance.
    """
    import shap
    from sklearn.model_selection import train_test_split
    
    seed_everything(args.seed)
    
    X, y, feature_names = load_dataset(args.dataset, args=args)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.split_seed, stratify=y
    )
    
    ensure_dir(args.output_dir)
    
    start_run = args.chunk_id * args.chunk_size
    end_run = min(start_run + args.chunk_size, args.n_runs)
    
    print(f"[chunk {args.chunk_id}] Runs {start_run}..{end_run - 1} for split_seed={args.split_seed}")
    
    all_importance = []
    all_meta = []
    
    for run_id in range(start_run, end_run):
        run_seed = args.seed + 100000 * args.split_seed + run_id
        seed_everything(run_seed)
        
        # Optional C variation for LogReg
        C_value = None
        rf_run_params = None
        
        if args.model.lower() in ("logreg", "logistic", "lr", "logistic_regression"):
            if args.c_mode == "varied":
                rng = np.random.default_rng(run_seed)
                log_c = rng.uniform(np.log10(args.c_min), np.log10(args.c_max))
                C_value = 10 ** log_c
            else:
                C_value = args.C
        
        if args.model.lower() in ("rf", "random_forest", "randomforest") and args.rf_varied:
            rng = np.random.default_rng(run_seed)
            rf_run_params = {
                "n_estimators": int(rng.integers(args.rf_n_estimators_min, args.rf_n_estimators_max + 1)),
                "max_depth": int(rng.integers(args.rf_max_depth_min, args.rf_max_depth_max + 1)),
                "max_features": float(rng.uniform(args.rf_max_features_min, args.rf_max_features_max)),
                "min_samples_split": int(rng.integers(args.rf_min_samples_split_min, args.rf_min_samples_split_max + 1)),
                "min_samples_leaf": int(rng.integers(args.rf_min_samples_leaf_min, args.rf_min_samples_leaf_max + 1)),
            }
        
        model = make_model(args, C_value=C_value, rf_params=rf_run_params)
        model.fit(X_train, y_train)
        acc = float(model.score(X_test, y_test))
        
        # SHAP values
        if args.model.lower() in ("logreg", "logistic", "lr", "logistic_regression"):
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            # Handle 3D array for TreeExplainer (SHAP 0.42+)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            elif shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]  # (n_samples, n_features, 2) -> positive class
        
        # Mean absolute SHAP importance
        importance = np.abs(shap_values).mean(axis=0)
        all_importance.append(importance)
        
        meta = {
            "run_id": run_id,
            "split_seed": args.split_seed,
            "run_seed": run_seed,
            "acc": acc,
        }
        
        if C_value is not None:
            meta["C"] = float(C_value)
        
        if rf_run_params is not None:
            for k, v in rf_run_params.items():
                meta[f"rf_{k}"] = v
        
        all_meta.append(meta)
        
        if args.write_per_run:
            out_path = Path(args.output_dir) / f"run_{args.split_seed}_{run_id}.json"
            save_json(meta, out_path)
    
    # Save chunk
    out_dir = Path(args.output_dir)
    np.save(out_dir / f"importance_split{args.split_seed}_chunk{args.chunk_id}.npy", np.array(all_importance))
    save_json(all_meta, out_dir / f"meta_split{args.split_seed}_chunk{args.chunk_id}.json")
    
    if args.save_split:
        np.savez(out_dir / f"split_{args.split_seed}.npz", 
                 X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                 feature_names=np.array(feature_names, dtype=object))
    
    print(f"[chunk {args.chunk_id}] Done. Saved {len(all_importance)} runs.")


# =============================================================================
# k=1 NULL HYPOTHESIS TESTING
# =============================================================================

def compute_bic(X, labels, k):
    """
    Compute Bayesian Information Criterion for Gaussian mixture model assumption.
    Lower BIC is better.
    
    BIC = -2 * log_likelihood + k * log(n)
    
    For k=1: single Gaussian, for k>1: mixture of Gaussians with cluster assignments.
    """
    n, d = X.shape
    
    if k == 1:
        # Single Gaussian: all data from one distribution
        mu = X.mean(axis=0)
        # Covariance (use diagonal for simplicity and numerical stability)
        var = np.var(X, axis=0) + 1e-10
        # Log-likelihood under diagonal Gaussian
        log_lik = -0.5 * n * d * np.log(2 * np.pi)
        log_lik -= 0.5 * n * np.sum(np.log(var))
        log_lik -= 0.5 * np.sum(((X - mu) ** 2) / var)
        # Parameters: d means + d variances
        n_params = 2 * d
    else:
        # Mixture of k Gaussians
        log_lik = 0.0
        n_params = 0
        for cluster_id in range(k):
            mask = labels == cluster_id
            n_c = mask.sum()
            if n_c == 0:
                continue
            X_c = X[mask]
            mu_c = X_c.mean(axis=0)
            var_c = np.var(X_c, axis=0) + 1e-10
            # Log-likelihood for this cluster
            log_lik_c = -0.5 * n_c * d * np.log(2 * np.pi)
            log_lik_c -= 0.5 * n_c * np.sum(np.log(var_c))
            log_lik_c -= 0.5 * np.sum(((X_c - mu_c) ** 2) / var_c)
            log_lik += log_lik_c
            # Parameters: d means + d variances per cluster
            n_params += 2 * d
        # Add mixing proportions (k-1 free parameters)
        n_params += (k - 1)
    
    bic = -2 * log_lik + n_params * np.log(n)
    return float(bic)


def compute_aic(X, labels, k):
    """
    Compute Akaike Information Criterion.
    Lower AIC is better.
    
    AIC = -2 * log_likelihood + 2 * k
    """
    n, d = X.shape
    
    if k == 1:
        mu = X.mean(axis=0)
        var = np.var(X, axis=0) + 1e-10
        log_lik = -0.5 * n * d * np.log(2 * np.pi)
        log_lik -= 0.5 * n * np.sum(np.log(var))
        log_lik -= 0.5 * np.sum(((X - mu) ** 2) / var)
        n_params = 2 * d
    else:
        log_lik = 0.0
        n_params = 0
        for cluster_id in range(k):
            mask = labels == cluster_id
            n_c = mask.sum()
            if n_c == 0:
                continue
            X_c = X[mask]
            mu_c = X_c.mean(axis=0)
            var_c = np.var(X_c, axis=0) + 1e-10
            log_lik_c = -0.5 * n_c * d * np.log(2 * np.pi)
            log_lik_c -= 0.5 * n_c * np.sum(np.log(var_c))
            log_lik_c -= 0.5 * np.sum(((X_c - mu_c) ** 2) / var_c)
            log_lik += log_lik_c
            n_params += 2 * d
        n_params += (k - 1)
    
    aic = -2 * log_lik + 2 * n_params
    return float(aic)


def compute_gap_statistic(X, labels, k, n_references=20, seed=42):
    """
    Compute Gap Statistic comparing clustering quality to uniform null reference.
    
    Gap(k) = E*[log(W_k)] - log(W_k)
    
    Where W_k is within-cluster dispersion, and E* is expectation under uniform null.
    
    Returns gap, gap_std (standard error of the reference distribution).
    """
    from sklearn.cluster import KMeans
    
    n, d = X.shape
    rng = np.random.default_rng(seed)
    
    def compute_wk(X_data, cluster_labels, n_clusters):
        """Within-cluster sum of squared distances from centroid."""
        wk = 0.0
        for c in range(n_clusters):
            mask = cluster_labels == c
            if mask.sum() == 0:
                continue
            X_c = X_data[mask]
            centroid = X_c.mean(axis=0)
            wk += np.sum((X_c - centroid) ** 2)
        return wk
    
    # Observed W_k
    if k == 1:
        wk_obs = compute_wk(X, np.zeros(n, dtype=int), 1)
    else:
        wk_obs = compute_wk(X, labels, k)
    
    log_wk_obs = np.log(wk_obs + 1e-10)
    
    # Reference distribution: uniform over bounding box
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    
    log_wk_refs = []
    for _ in range(n_references):
        # Generate uniform reference data
        X_ref = rng.uniform(mins, maxs, size=(n, d))
        
        if k == 1:
            labels_ref = np.zeros(n, dtype=int)
        else:
            km = KMeans(n_clusters=k, random_state=int(rng.integers(0, 10000)), n_init="auto")
            labels_ref = km.fit_predict(X_ref)
        
        wk_ref = compute_wk(X_ref, labels_ref, k)
        log_wk_refs.append(np.log(wk_ref + 1e-10))
    
    log_wk_refs = np.array(log_wk_refs)
    gap = log_wk_refs.mean() - log_wk_obs
    gap_std = np.std(log_wk_refs) * np.sqrt(1 + 1 / n_references)
    
    return float(gap), float(gap_std)


def pick_best_k_kmeans(X, k_max=8, seed=0, silhouette_threshold=0.25, use_bic=True, use_gap=False):
    """
    Select optimal k with k=1 as formal null hypothesis.
    
    Criteria for accepting k > 1:
    1. Silhouette score >= silhouette_threshold
    2. BIC(k) < BIC(1) (if use_bic=True)
    3. Gap(k) > Gap(k-1) + s_{k-1} (if use_gap=True) - Tibshirani rule
    
    Returns (best_k, labels, metrics_dict)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    n = X.shape[0]
    metrics = {
        "method": "k1_null_hypothesis_testing",
        "silhouette_threshold": silhouette_threshold,
        "use_bic": use_bic,
        "use_gap": use_gap,
        "silhouette_by_k": {},
        "bic_by_k": {},
        "aic_by_k": {},
        "calinski_harabasz_by_k": {},
        "gap_by_k": {},
        "gap_std_by_k": {},
        "k1_accepted": False,
    }
    
    # Degenerate case: all vectors identical
    if np.allclose(X, X.mean(axis=0), atol=1e-12):
        metrics["k1_reason"] = "degenerate_identical_vectors"
        metrics["k1_accepted"] = True
        return 1, np.zeros(n, dtype=int), metrics
    
    total_variance = np.var(X, axis=0).sum()
    metrics["total_variance"] = float(total_variance)
    
    if total_variance < 1e-10:
        metrics["k1_reason"] = "negligible_variance"
        metrics["k1_accepted"] = True
        return 1, np.zeros(n, dtype=int), metrics
    
    # Compute BIC/AIC for k=1
    labels_k1 = np.zeros(n, dtype=int)
    bic_k1 = compute_bic(X, labels_k1, 1)
    aic_k1 = compute_aic(X, labels_k1, 1)
    metrics["bic_by_k"][1] = bic_k1
    metrics["aic_by_k"][1] = aic_k1
    
    if use_gap:
        gap_k1, gap_std_k1 = compute_gap_statistic(X, labels_k1, 1, seed=seed)
        metrics["gap_by_k"][1] = gap_k1
        metrics["gap_std_by_k"][1] = gap_std_k1
    
    # Track best k based on silhouette (among k >= 2)
    best_k_silhouette = None
    best_silhouette = -1.0
    best_labels = None
    
    # Store all clustering results
    kmeans_results = {}
    
    for k in range(2, min(k_max + 1, n)):
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(X)
        
        # Check for degenerate clustering
        unique_labels = len(set(labels))
        if unique_labels < k:
            continue
        
        kmeans_results[k] = labels.copy()
        
        # Silhouette score
        sil = silhouette_score(X, labels, metric="euclidean")
        metrics["silhouette_by_k"][k] = float(sil)
        
        # BIC/AIC
        bic_k = compute_bic(X, labels, k)
        aic_k = compute_aic(X, labels, k)
        metrics["bic_by_k"][k] = bic_k
        metrics["aic_by_k"][k] = aic_k
        
        # Calinski-Harabasz (variance ratio)
        ch = calinski_harabasz_score(X, labels)
        metrics["calinski_harabasz_by_k"][k] = float(ch)
        
        # Gap statistic
        if use_gap:
            gap_k, gap_std_k = compute_gap_statistic(X, labels, k, seed=seed)
            metrics["gap_by_k"][k] = gap_k
            metrics["gap_std_by_k"][k] = gap_std_k
        
        # Track best by silhouette
        if sil > best_silhouette:
            best_silhouette = sil
            best_k_silhouette = k
            best_labels = labels.copy()
    
    # Decision logic: Does any k > 1 beat the null hypothesis?
    if best_k_silhouette is None:
        # No valid k > 1 found
        metrics["k1_reason"] = "no_valid_k_gt_1_found"
        metrics["k1_accepted"] = True
        return 1, np.zeros(n, dtype=int), metrics
    
    # Criterion 1: Silhouette threshold
    silhouette_passes = best_silhouette >= silhouette_threshold
    metrics["silhouette_criterion_passed"] = silhouette_passes
    
    # Criterion 2: BIC comparison (k should have lower BIC than k=1)
    if use_bic:
        bic_best_k = metrics["bic_by_k"].get(best_k_silhouette, float('inf'))
        bic_passes = bic_best_k < bic_k1
        metrics["bic_criterion_passed"] = bic_passes
        metrics["bic_improvement"] = float(bic_k1 - bic_best_k)
    else:
        bic_passes = True
    
    # Criterion 3: Gap statistic (Tibshirani's rule: Gap(k) >= Gap(k-1) - s_{k-1})
    if use_gap:
        gap_passes = False
        for k in sorted(metrics["gap_by_k"].keys()):
            if k == 1:
                continue
            gap_k = metrics["gap_by_k"][k]
            gap_k_prev = metrics["gap_by_k"].get(k - 1, metrics["gap_by_k"][1])
            gap_std_prev = metrics["gap_std_by_k"].get(k - 1, metrics["gap_std_by_k"][1])
            # Tibshirani rule: choose smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
            # We modify to: reject k=1 if Gap(k) > Gap(1) for some k > 1
            if gap_k > gap_k_prev:
                gap_passes = True
                metrics["gap_first_improvement_k"] = k
                break
        metrics["gap_criterion_passed"] = gap_passes
    else:
        gap_passes = True
    
    # Final decision: Accept k > 1 only if ALL criteria pass
    accept_k_gt_1 = silhouette_passes and bic_passes and gap_passes
    
    if not accept_k_gt_1:
        # k=1 is accepted as null
        reasons = []
        if not silhouette_passes:
            reasons.append(f"silhouette_{best_silhouette:.3f}_below_{silhouette_threshold}")
        if use_bic and not bic_passes:
            reasons.append(f"bic_k{best_k_silhouette}_not_better_than_k1")
        if use_gap and not gap_passes:
            reasons.append("gap_statistic_no_improvement")
        
        metrics["k1_reason"] = "; ".join(reasons)
        metrics["k1_accepted"] = True
        metrics["rejected_k"] = best_k_silhouette
        metrics["rejected_silhouette"] = float(best_silhouette)
        
        return 1, np.zeros(n, dtype=int), metrics
    
    # Accept k > 1
    metrics["k1_accepted"] = False
    metrics["accepted_k"] = best_k_silhouette
    metrics["accepted_silhouette"] = float(best_silhouette)
    
    return best_k_silhouette, best_labels, metrics


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


def aggregate_split(args, split_seed):
    """
    Aggregate all chunks for one split, compute SHAP importance vectors, and cluster.
    Now with formal k=1 null hypothesis testing.
    
    Output NPZ contains:
    - run_indices, accs, c_values (if LogReg)
    - importance: raw mean |SHAP| vectors
    - best_k, labels, entropy, entropy_norm
    - All silhouette, BIC, AIC, Gap scores
    - Clustering decision rationale
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

    # RF hyperparameters if present
    rf_params_arrays = {}
    rf_param_keys = ["rf_n_estimators", "rf_max_depth", "rf_max_features", 
                     "rf_min_samples_split", "rf_min_samples_leaf"]
    
    for key in rf_param_keys:
        if key in metas[0]:
            values = []
            for m in metas:
                val = m.get(key, np.nan)
                if val is None:
                    val = -1  # Use -1 to represent None for max_depth
                values.append(val)
            rf_params_arrays[key] = np.array(values, dtype=float)

    # ==========================================================================
    # CLUSTERING WITH k=1 NULL HYPOTHESIS
    # ==========================================================================
    
    # L2 normalize importance vectors (paper methodology)
    mu = imps.mean(axis=0)
    centered = imps - mu
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed_importance = centered / norms
    
    # Perform clustering with k=1 as null hypothesis
    silhouette_threshold = getattr(args, 'silhouette_threshold', 0.25)
    use_bic = getattr(args, 'use_bic', True)
    use_gap = getattr(args, 'use_gap', False)
    
    best_k, labels, metrics = pick_best_k_kmeans(
        normed_importance, 
        k_max=args.k_max, 
        seed=args.seed,
        silhouette_threshold=silhouette_threshold,
        use_bic=use_bic,
        use_gap=use_gap
    )
    
    # Compute normalized entropy
    if best_k > 1:
        counts = np.array([(labels == i).sum() for i in range(best_k)])
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        entropy_norm = entropy / np.log2(best_k)
    else:
        entropy = 0.0
        entropy_norm = 0.0
    
    # Build save dict
    save_dict = {
        "run_indices": run_indices,
        "accs": accs,
        "c_values": c_values if c_values is not None else np.array([]),
        "importance": imps,
        # Clustering results
        "best_k": np.array([best_k]),
        "labels": labels,
        "entropy": np.array([entropy]),
        "entropy_norm": np.array([entropy_norm]),
        "silhouette_threshold": np.array([silhouette_threshold]),
        "k1_accepted": np.array([metrics.get("k1_accepted", False)]),
    }
    
    # Save all metric scores
    for k, score in metrics.get("silhouette_by_k", {}).items():
        save_dict[f"silhouette_k{k}"] = np.array([score])
    
    for k, score in metrics.get("bic_by_k", {}).items():
        save_dict[f"bic_k{k}"] = np.array([score])
    
    for k, score in metrics.get("aic_by_k", {}).items():
        save_dict[f"aic_k{k}"] = np.array([score])
    
    for k, score in metrics.get("calinski_harabasz_by_k", {}).items():
        save_dict[f"calinski_harabasz_k{k}"] = np.array([score])
    
    for k, score in metrics.get("gap_by_k", {}).items():
        save_dict[f"gap_k{k}"] = np.array([score])
        if k in metrics.get("gap_std_by_k", {}):
            save_dict[f"gap_std_k{k}"] = np.array([metrics["gap_std_by_k"][k]])
    
    # Save clustering decision rationale
    if "k1_reason" in metrics:
        save_dict["k1_reason"] = np.array([metrics["k1_reason"]], dtype=object)
    if "rejected_k" in metrics:
        save_dict["rejected_k"] = np.array([metrics["rejected_k"]])
    if "rejected_silhouette" in metrics:
        save_dict["rejected_silhouette"] = np.array([metrics["rejected_silhouette"]])
    if "accepted_silhouette" in metrics:
        save_dict["accepted_silhouette"] = np.array([metrics["accepted_silhouette"]])
    
    # Save criterion pass/fail
    for crit in ["silhouette_criterion_passed", "bic_criterion_passed", "gap_criterion_passed"]:
        if crit in metrics:
            save_dict[crit] = np.array([metrics[crit]])
    
    if "bic_improvement" in metrics:
        save_dict["bic_improvement"] = np.array([metrics["bic_improvement"]])
    
    save_dict.update(rf_params_arrays)

    np.savez(out_dir / f"aggregate_split{split_seed}.npz", **save_dict)

    print(f"[aggregate] saved aggregate_split{split_seed}.npz")
    print(f"  n_runs={len(metas)}, best_k={best_k}, entropy_norm={entropy_norm:.4f}")
    
    if metrics.get("k1_accepted"):
        print(f"  k=1 ACCEPTED (null hypothesis): {metrics.get('k1_reason', 'unknown')}")
        if "rejected_k" in metrics:
            print(f"    Rejected k={metrics['rejected_k']} with silhouette={metrics.get('rejected_silhouette', 'N/A'):.3f}")
    else:
        print(f"  k={best_k} ACCEPTED (null rejected)")
        print(f"    Silhouette: {metrics.get('accepted_silhouette', 'N/A'):.3f}")
        if "bic_improvement" in metrics:
            print(f"    BIC improvement over k=1: {metrics['bic_improvement']:.1f}")

    # SYNTH_ORTHO Interpretation Block
    if args.dataset.upper() == "SYNTH_ORTHO":
        print("\n" + "="*60)
        print("[SYNTH_ORTHO INTERPRETATION - k=1 NULL HYPOTHESIS]")
        print("="*60)
        if best_k == 1:
            print(">> RESULT: k*=1 (Null hypothesis ACCEPTED)")
            print(">> On orthogonal synthetic features, explanations collapse to single basin.")
            print(">> This confirms: without correlations, the Rashomon set is convex/unimodal.")
            print(">> Implication: Multiplicity in real datasets stems from collinearity")
            print("   or non-identifiability, not from stochastic training alone.")
        else:
            print(f">> RESULT: k*={best_k} (Null hypothesis REJECTED)")
            print(">> WARNING: Found MULTIPLE basins even on orthogonal features!")
            print(">> Possible causes:")
            print("   - Low SNR making ground truth unrecoverable")
            print("   - Hyperparameter variance creating distinct regimes")
            print("   - Underspecification in the model class")
            print(">> This is a surprising result requiring investigation.")
        print("="*60)


def aggregate_and_cluster(args, split_seeds):
    """
    Stack multiple split aggregates into a "universal" NPZ file and cluster in explanation space.
    NOTE: This aggregates ACROSS splits, which may violate the causal context principle.
    """
    out_dir = Path(args.output_dir)
    all_importance = []
    all_accs = []
    all_split = []
    all_run = []
    all_C = []
    
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
        
        # Check for RF params
        for key in all_rf_params.keys():
            if key in data:
                has_rf_params = True
                all_rf_params[key].append(data[key])
            else:
                all_rf_params[key].append(np.full(len(run_idx), np.nan))

    stacked = np.vstack(all_importance)
    stacked_accs = np.concatenate(all_accs)
    stacked_split = np.concatenate(all_split)
    stacked_run = np.concatenate(all_run)
    stacked_C = np.concatenate(all_C)
    
    rf_params_stacked = {}
    if has_rf_params:
        for key, arrays in all_rf_params.items():
            rf_params_stacked[key] = np.concatenate(arrays)

    # Center and normalize
    mu = stacked.mean(axis=0)
    centered = stacked - mu
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = centered / norms

    print(f"[WARNING] Universal aggregation mixes {len(split_seeds)} splits.")
    print(f"  This may violate causal context principle - use with caution.")
    
    # Cluster with k=1 null
    silhouette_threshold = getattr(args, 'silhouette_threshold', 0.25)
    use_bic = getattr(args, 'use_bic', True)
    use_gap = getattr(args, 'use_gap', False)
    
    best_k, labels, metrics = pick_best_k_kmeans(
        normed, 
        k_max=args.k_max, 
        seed=args.seed,
        silhouette_threshold=silhouette_threshold,
        use_bic=use_bic,
        use_gap=use_gap
    )
    
    if best_k > 1:
        counts = np.array([(labels == i).sum() for i in range(best_k)])
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        H = -np.sum(probs * np.log2(probs))
        H_norm = H / np.log2(best_k)
    else:
        H_norm = 0.0

    save_dict = {
        "importance": stacked,
        "accs": stacked_accs,
        "split_ids": stacked_split,
        "run_ids": stacked_run,
        "c_values": stacked_C,
        "best_k": np.array([best_k]),
        "labels": labels,
        "entropy_norm": np.array([H_norm]),
        "k1_accepted": np.array([metrics.get("k1_accepted", False)]),
    }
    
    # Save all metrics
    for k, score in metrics.get("silhouette_by_k", {}).items():
        save_dict[f"silhouette_k{k}"] = np.array([score])
    for k, score in metrics.get("bic_by_k", {}).items():
        save_dict[f"bic_k{k}"] = np.array([score])
    
    if "k1_reason" in metrics:
        save_dict["k1_reason"] = np.array([metrics["k1_reason"]], dtype=object)
    
    save_dict.update(rf_params_stacked)

    out_path = out_dir / "aggregate_universal.npz"
    np.savez(out_path, **save_dict)
    
    print(f"[universal] Saved {out_path}")
    print(f"  Total runs: {len(stacked)}, best_k={best_k}, entropy_norm={H_norm:.4f}")
    if metrics.get("k1_accepted"):
        print(f"  k=1 ACCEPTED: {metrics.get('k1_reason', 'unknown')}")

    if args.disagreement:
        print("\nComputing disagreement analysis...")
        run_disagreement_analysis(args, out_path)


def run_disagreement_analysis(args, npz_path):
    """
    Load the universal NPZ, pick representative models from each cluster,
    and compute prediction disagreement on test data.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    data = np.load(npz_path, allow_pickle=True)
    labels = data["labels"]
    best_k = int(data["best_k"][0])
    
    if best_k <= 1:
        print("  Skipping disagreement: k=1 means no distinct basins.")
        return None
    
    importance = data["importance"]
    split_ids = data.get("split_ids", None)
    run_ids = data["run_ids"]
    c_values = data.get("c_values", None)
    
    # Get RF params if available
    rf_params = {}
    for key in ["rf_n_estimators", "rf_max_depth", "rf_max_features", 
                "rf_min_samples_split", "rf_min_samples_leaf"]:
        if key in data:
            rf_params[key] = data[key]
    
    # Load dataset
    X, y, feature_names = load_dataset(args.dataset, args=args)
    
    # Select representative from each cluster
    selected_clusters = list(range(best_k))
    
    # L2 normalize for distance computation
    mu = importance.mean(axis=0)
    centered = importance - mu
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = centered / norms
    
    rep_runs = []
    rep_C = []
    rep_split_seeds = []
    rep_rf_params = []
    rep_models = []
    
    for cluster_id in selected_clusters:
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_normed = normed[mask]
        cluster_C = c_values[mask] if c_values is not None else np.full(mask.sum(), np.nan)
        
        from scipy.spatial.distance import cosine
        centroid = cluster_normed.mean(axis=0)
        distances = np.array([cosine(imp, centroid) for imp in cluster_normed])
        rep_idx_in_cluster = int(np.argmin(distances))
        
        global_idx = np.where(mask)[0][rep_idx_in_cluster]
        
        rep_runs.append(int(cluster_indices[rep_idx_in_cluster]))
        rep_C.append(float(cluster_C[rep_idx_in_cluster]) if not np.isnan(cluster_C[rep_idx_in_cluster]) else None)
        
        if split_ids is not None:
            rep_split_seed = int(split_ids[global_idx])
        else:
            rep_split_seed = args.split_seed
        rep_split_seeds.append(rep_split_seed)
        
        if rf_params is not None and len(rf_params) > 0:
            rep_rf = {}
            for key in rf_params.keys():
                val = rf_params[key][global_idx]
                if key == "rf_max_depth" and val == -1:
                    val = None
                elif not np.isnan(val):
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
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=rep_split_seed, stratify=y
        )
        
        run_seed = args.seed + 100000 * rep_split_seed + int(rep_runs[-1])
        seed_everything(run_seed)
        
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
    
    if len(set(rep_split_seeds)) > 1:
        print(f"Warning: Representatives from different splits: {rep_split_seeds}")
        print("Using first split for disagreement computation.")
    
    main_split_seed = rep_split_seeds[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=main_split_seed, stratify=y
    )
    
    print("\nComputing disagreements...")
    def _positive_class_proba(model, X):
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
    p = argparse.ArgumentParser(description="EvoXplain core engine with k=1 null hypothesis testing.")

    p.add_argument("--dataset", type=str, required=True, help="Dataset name (breast_cancer, compas, adult, SYNTH_ORTHO)")
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

    # Synthetic Dataset Controls
    p.add_argument("--synth_n_samples", type=int, default=1000, help="Samples for SYNTH_ORTHO")
    p.add_argument("--synth_n_features", type=int, default=10, help="Features for SYNTH_ORTHO")
    p.add_argument("--synth_snr", type=float, default=2.0, help="Signal-to-Noise Ratio for SYNTH_ORTHO ground truth")

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

    # Clustering / k=1 null hypothesis
    p.add_argument("--k_max", type=int, default=8, help="Max k for clustering")
    p.add_argument("--silhouette_threshold", type=float, default=0.25, 
                   help="Minimum silhouette score to reject k=1 null (default 0.25)")
    p.add_argument("--use_bic", action="store_true", default=True,
                   help="Use BIC criterion for k=1 null testing (default: True)")
    p.add_argument("--no_bic", action="store_false", dest="use_bic",
                   help="Disable BIC criterion")
    p.add_argument("--use_gap", action="store_true", default=False,
                   help="Use Gap statistic for k=1 null testing (slower, default: False)")
    
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
