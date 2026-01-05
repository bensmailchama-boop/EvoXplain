#!/usr/bin/env python3
"""
evoxplain_disagreement_within_split.py

Compute disagreement analysis WITHIN each split (not across splits).
This is the methodologically correct approach: compare mechanistic basins
that were trained on the same train/test split.

For each split:
1. Load the aggregated data
2. Cluster in normed explanation space
3. Select representative model from each cluster
4. Retrain representatives and compute prediction disagreements on test set
5. Save per-split disagreement report
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(dataset_name: str):
    """Load supported datasets."""
    if dataset_name.lower() in ("breast_cancer", "breast-cancer", "bc"):
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = list(data.feature_names)
        # Scale features for LogReg convergence (matches core engine)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y, feature_names
    
    if dataset_name.lower() in ("compas",):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        compas_path = "data/compas-scores-two-years.csv"
        df = pd.read_csv(compas_path)
        
        df = df[(df['days_b_screening_arrest'] <= 30) & 
                (df['days_b_screening_arrest'] >= -30) &
                (df['is_recid'] != -1) &
                (df['c_charge_degree'] != 'O') &
                (df['score_text'] != 'N/A')]
        
        feature_cols = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 
                       'sex', 'priors_count', 'days_b_screening_arrest', 
                       'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
        feature_cols = [col for col in feature_cols if col in df.columns]
        df = df[feature_cols]
        
        y = df['two_year_recid'].values
        
        categorical_cols = ['c_charge_degree', 'race', 'age_cat', 'score_text', 'sex']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        df_encoded = pd.get_dummies(df.drop(columns=['two_year_recid', 'is_recid'] + 
                                                     [c for c in ['c_jail_in', 'c_jail_out'] if c in df.columns], 
                                                     errors='ignore'), 
                                    columns=categorical_cols, drop_first=True)
        
        X = df_encoded.values.astype(float)
        feature_names = list(df_encoded.columns)
        
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
        df = pd.read_csv(adult_path)
        df.columns = df.columns.str.strip()
        
        if 'income' in df.columns:
            y = (df['income'].str.strip().str.contains('>50K')).astype(int).values
            df = df.drop(columns=['income'])
        else:
            raise ValueError("Adult dataset missing 'income' column")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        X = df_encoded.values.astype(float)
        feature_names = list(df_encoded.columns)
        
        numerical_indices = [i for i, name in enumerate(feature_names) if name in numerical_cols]
        if len(numerical_indices) > 0:
            scaler = StandardScaler()
            X[:, numerical_indices] = scaler.fit_transform(X[:, numerical_indices])
        
        return X, y, feature_names

    raise ValueError(f"Unknown dataset: {dataset_name}")


def l2_normalize(X):
    """Center and L2 normalize each row."""
    mu = X.mean(axis=0)
    centered = X - mu
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normed = centered / (norms + 1e-12)
    return normed, mu


def find_best_k_silhouette(X_normed, k_min=2, k_max=8, seed=42):
    """Find optimal k using silhouette score."""
    n = X_normed.shape[0]
    
    if np.allclose(X_normed, X_normed[0], atol=1e-10):
        return 1, np.zeros(n, dtype=int), {}
    
    best_k = 1
    best_score = -1.0
    best_labels = np.zeros(n, dtype=int)
    sil_scores = {}
    
    for k in range(k_min, min(k_max + 1, n)):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X_normed)
        
        if len(np.unique(labels)) < k:
            continue
            
        score = silhouette_score(X_normed, labels)
        sil_scores[k] = score
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels.copy()
    
    return best_k, best_labels, sil_scores


def make_rf_model(rf_params, random_state):
    """Create RF model with given hyperparameters."""
    from sklearn.ensemble import RandomForestClassifier
    
    n_estimators = int(rf_params.get('n_estimators', 100))
    max_depth = rf_params.get('max_depth', None)
    if max_depth == -1:
        max_depth = None
    elif max_depth is not None:
        max_depth = int(max_depth)
    
    min_samples_split = int(rf_params.get('min_samples_split', 2))
    min_samples_leaf = int(rf_params.get('min_samples_leaf', 1))
    max_features = rf_params.get('max_features', 0.7)
    
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1
    )


def make_logreg_model(C_value, random_state):
    """Create Logistic Regression model."""
    from sklearn.linear_model import LogisticRegression
    
    return LogisticRegression(
        C=C_value,
        solver='lbfgs',
        max_iter=10000,
        random_state=random_state,
        n_jobs=-1
    )


def analyze_split(npz_path, dataset_name, model_type, test_size=0.3, 
                  k_min=2, k_max=8, seed=42):
    """
    Analyze disagreement within a single split.
    
    Returns: (split_seed, summary_dict, disagreement_df)
    """
    # Extract split seed from filename
    stem = npz_path.stem
    split_seed = int(stem.split("split")[-1])
    
    print(f"\n{'='*70}")
    print(f"Analyzing split {split_seed}")
    print(f"{'='*70}")
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    importance = data['importance']
    run_indices = data['run_indices']
    accs = data['accs']
    
    # Get hyperparameters
    c_values = data.get('c_values', None)
    if c_values is not None and len(c_values) == 0:
        c_values = None
        
    rf_params = {}
    for key in ['rf_n_estimators', 'rf_max_depth', 'rf_max_features', 
                'rf_min_samples_split', 'rf_min_samples_leaf']:
        if key in data:
            rf_params[key] = data[key]
    
    n_runs = len(run_indices)
    print(f"  Loaded {n_runs} runs")
    
    # L2 normalize and cluster
    normed_importance, mu = l2_normalize(importance)
    best_k, labels, sil_scores = find_best_k_silhouette(
        normed_importance, k_min=k_min, k_max=k_max, seed=seed
    )
    
    # Compute entropy
    if best_k > 1:
        counts = np.array([(labels == i).sum() for i in range(best_k)])
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy_norm = -np.sum(probs * np.log2(probs)) / np.log2(best_k)
    else:
        entropy_norm = 0.0
    
    print(f"  Clustering: k={best_k}, entropy_norm={entropy_norm:.4f}")
    print(f"  Silhouette scores: {sil_scores}")
    
    if best_k < 2:
        print(f"  WARNING: Only 1 cluster found, no disagreement analysis possible")
        return split_seed, {
            'k': best_k,
            'entropy_norm': entropy_norm,
            'silhouette_scores': sil_scores,
            'n_runs': n_runs,
        }, None
    
    # Load dataset and create train/test split
    X, y, feature_names = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed, stratify=y
    )
    
    print(f"  Test set size: {len(X_test)}")
    
    # Find representative from each cluster (closest to centroid)
    representatives = []
    for cluster_id in range(best_k):
        mask = (labels == cluster_id)
        cluster_indices = np.where(mask)[0]
        cluster_normed = normed_importance[mask]
        
        # Centroid in normed space
        centroid = cluster_normed.mean(axis=0)
        
        # Find closest point to centroid
        distances = np.linalg.norm(cluster_normed - centroid, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        
        # Get hyperparameters for this run
        run_id = run_indices[closest_idx]
        acc = accs[closest_idx]
        
        rep_info = {
            'cluster_id': cluster_id,
            'run_idx': closest_idx,
            'run_id': int(run_id),
            'acc': float(acc),
        }
        
        if c_values is not None:
            rep_info['C'] = float(c_values[closest_idx])
        
        for key, vals in rf_params.items():
            rep_info[key] = float(vals[closest_idx])
        
        representatives.append(rep_info)
        print(f"  Cluster {cluster_id}: rep run {run_id}, acc={acc:.4f}")
    
    # Retrain representative models and compute disagreements
    print(f"\n  Retraining {len(representatives)} representative models...")
    
    rep_models = []
    for rep in representatives:
        run_seed = seed + 100000 * split_seed + rep['run_id']
        seed_everything(run_seed)
        
        if model_type.lower() in ('rf', 'random_forest'):
            rf_hp = {
                'n_estimators': rep.get('rf_n_estimators', 100),
                'max_depth': rep.get('rf_max_depth', None),
                'min_samples_split': rep.get('rf_min_samples_split', 2),
                'min_samples_leaf': rep.get('rf_min_samples_leaf', 1),
                'max_features': rep.get('rf_max_features', 0.7),
            }
            model = make_rf_model(rf_hp, run_seed)
        else:
            C_val = rep.get('C', 1.0)
            model = make_logreg_model(C_val, run_seed)
        
        model.fit(X_train, y_train)
        rep_models.append(model)
    
    # Compute probability predictions
    def positive_class_proba(model, X):
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
    
    probs = np.array([positive_class_proba(m, X_test) for m in rep_models])
    disagreement = probs.max(axis=0) - probs.min(axis=0)
    sorted_indices = np.argsort(disagreement)[::-1]
    
    # Create report
    report = []
    for idx in sorted_indices:
        instance_probs = {f"cluster_{i}": float(probs[i, idx]) for i in range(best_k)}
        report.append({
            "test_index": int(idx),
            "true_label": int(y_test[idx]),
            "disagreement": float(disagreement[idx]),
            **instance_probs
        })
    
    df = pd.DataFrame(report)
    
    # Print summary
    print(f"\n  Disagreement statistics:")
    print(f"    Max disagreement: {disagreement.max():.4f}")
    print(f"    Mean disagreement: {disagreement.mean():.4f}")
    print(f"    Median disagreement: {np.median(disagreement):.4f}")
    print(f"    Instances with >0.1 disagreement: {(disagreement > 0.1).sum()}")
    print(f"    Instances with >0.2 disagreement: {(disagreement > 0.2).sum()}")
    print(f"    Instances with >0.3 disagreement: {(disagreement > 0.3).sum()}")
    
    summary = {
        'k': best_k,
        'entropy_norm': entropy_norm,
        'silhouette_scores': sil_scores,
        'n_runs': n_runs,
        'n_test': len(X_test),
        'max_disagreement': float(disagreement.max()),
        'mean_disagreement': float(disagreement.mean()),
        'median_disagreement': float(np.median(disagreement)),
        'n_disagree_gt_0.1': int((disagreement > 0.1).sum()),
        'n_disagree_gt_0.2': int((disagreement > 0.2).sum()),
        'n_disagree_gt_0.3': int((disagreement > 0.3).sum()),
        'representatives': representatives,
    }
    
    return split_seed, summary, df


def main():
    parser = argparse.ArgumentParser(
        description="Compute within-split disagreement analysis for EvoXplain"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing aggregate_split*.npz files")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (breast_cancer, compas, adult)")
    parser.add_argument("--model", type=str, default="rf",
                       help="Model type (rf, logreg)")
    parser.add_argument("--test_size", type=float, default=0.3,
                       help="Test set fraction (must match original experiment)")
    parser.add_argument("--k_min", type=int, default=2,
                       help="Minimum k for silhouette scan")
    parser.add_argument("--k_max", type=int, default=8,
                       help="Maximum k for silhouette scan")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for clustering")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: input_dir)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all split files
    files = sorted(input_dir.glob("aggregate_split*.npz"))
    
    if not files:
        raise FileNotFoundError(f"No aggregate_split*.npz files found in {input_dir}")
    
    print(f"Found {len(files)} split files")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Clustering: k_min={args.k_min}, k_max={args.k_max}")
    
    # Analyze each split
    all_summaries = {}
    
    for f in files:
        split_seed, summary, df = analyze_split(
            f, args.dataset, args.model, args.test_size,
            args.k_min, args.k_max, args.seed
        )
        
        all_summaries[split_seed] = summary
        
        if df is not None:
            out_path = output_dir / f"disagreement_within_split{split_seed}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}")
    
    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY (Within-Split Disagreement)")
    print(f"{'='*70}")
    
    ks = [s['k'] for s in all_summaries.values()]
    entropies = [s['entropy_norm'] for s in all_summaries.values()]
    max_disagrees = [s.get('max_disagreement', 0) for s in all_summaries.values()]
    mean_disagrees = [s.get('mean_disagreement', 0) for s in all_summaries.values()]
    
    print(f"Splits analyzed: {len(all_summaries)}")
    print(f"k values: {ks}")
    print(f"Mean k: {np.mean(ks):.2f}")
    print(f"Entropy (norm): {[f'{e:.3f}' for e in entropies]}")
    print(f"Mean entropy: {np.mean(entropies):.3f}")
    print(f"Max disagreement per split: {[f'{d:.3f}' for d in max_disagrees]}")
    print(f"Mean of max disagreements: {np.mean(max_disagrees):.3f}")
    print(f"Mean disagreement per split: {[f'{d:.3f}' for d in mean_disagrees]}")
    print(f"Mean of mean disagreements: {np.mean(mean_disagrees):.3f}")
    
    # Save overall summary
    import json
    summary_path = output_dir / "disagreement_within_split_summary.json"
    with open(summary_path, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        clean_summaries = {}
        for k, v in all_summaries.items():
            clean_summaries[str(k)] = {
                key: (val if not isinstance(val, np.ndarray) else val.tolist())
                for key, val in v.items()
            }
        json.dump(clean_summaries, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path.name}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
