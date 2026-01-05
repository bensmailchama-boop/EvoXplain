#!/usr/bin/env python3
"""
evoxplain_per_split_summary.py

Generate a CSV table summarizing per-split clustering metrics.
This is useful for both varied C (showing multiplicity) and fixed C (showing collapse to k=1).

Output columns:
- split_seed
- n_runs
- best_k
- silhouette (at best_k)
- entropy_norm
- accuracy_mean
- accuracy_std
- C_mean (if available)
- C_std (if available)
- cluster_sizes
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
    
    # Check for degenerate case (all vectors identical)
    if np.allclose(X_normed, X_normed[0], atol=1e-10):
        return 1, np.zeros(n, dtype=int), {}, None
    
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
    
    return best_k, best_labels, sil_scores, best_score if best_k > 1 else None


def analyze_split(npz_path, k_min=2, k_max=8, seed=42):
    """Analyze a single split and return metrics dict."""
    
    # Extract split seed from filename
    stem = npz_path.stem
    try:
        split_seed = int(stem.split("split")[-1])
    except:
        split_seed = -1
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    
    importance = data.get('importance')
    if importance is None:
        return None
    
    run_indices = data.get('run_indices', np.arange(len(importance)))
    accs = data.get('accs', np.full(len(importance), np.nan))
    c_values = data.get('c_values', None)
    
    n_runs = len(importance)
    
    # L2 normalize and cluster
    normed_importance, mu = l2_normalize(importance)
    best_k, labels, sil_scores, best_sil = find_best_k_silhouette(
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
    
    # Cluster sizes as string
    cluster_sizes = [int((labels == i).sum()) for i in range(best_k)]
    cluster_sizes_str = str(cluster_sizes)
    
    # Build metrics dict
    metrics = {
        'split_seed': split_seed,
        'n_runs': n_runs,
        'best_k': best_k,
        'silhouette': best_sil if best_sil is not None else np.nan,
        'entropy_norm': entropy_norm,
        'accuracy_mean': np.nanmean(accs),
        'accuracy_std': np.nanstd(accs),
        'cluster_sizes': cluster_sizes_str,
    }
    
    # Add C statistics if available
    if c_values is not None and len(c_values) > 0 and not np.all(np.isnan(c_values)):
        metrics['C_mean'] = np.nanmean(c_values)
        metrics['C_std'] = np.nanstd(c_values)
        metrics['C_min'] = np.nanmin(c_values)
        metrics['C_max'] = np.nanmax(c_values)
    else:
        metrics['C_mean'] = np.nan
        metrics['C_std'] = np.nan
        metrics['C_min'] = np.nan
        metrics['C_max'] = np.nan
    
    # Add all silhouette scores
    for k, score in sil_scores.items():
        metrics[f'sil_k{k}'] = score
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-split summary CSV for EvoXplain experiments"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing aggregate_split*.npz files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV path (default: input_dir/per_split_summary.csv)")
    parser.add_argument("--k_min", type=int, default=2,
                       help="Minimum k for silhouette scan")
    parser.add_argument("--k_max", type=int, default=8,
                       help="Maximum k for silhouette scan")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for clustering")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output) if args.output else input_dir / "per_split_summary.csv"
    
    # Find all split files
    files = sorted(input_dir.glob("aggregate_split*.npz"))
    
    if not files:
        raise FileNotFoundError(f"No aggregate_split*.npz files found in {input_dir}")
    
    print(f"Found {len(files)} split files")
    print(f"Clustering: k_min={args.k_min}, k_max={args.k_max}")
    print()
    
    # Analyze each split
    all_metrics = []
    
    for f in files:
        print(f"Processing {f.name}...")
        metrics = analyze_split(f, args.k_min, args.k_max, args.seed)
        if metrics:
            all_metrics.append(metrics)
            sil_str = f"{metrics['silhouette']:.4f}" if not np.isnan(metrics['silhouette']) else "N/A"
            print(f"  k={metrics['best_k']}, sil={sil_str}, "
                  f"H={metrics['entropy_norm']:.4f}, acc={metrics['accuracy_mean']:.4f}")
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns for clarity
    primary_cols = ['split_seed', 'n_runs', 'best_k', 'silhouette', 'entropy_norm', 
                    'accuracy_mean', 'accuracy_std', 'C_mean', 'C_std', 'C_min', 'C_max', 
                    'cluster_sizes']
    other_cols = [c for c in df.columns if c not in primary_cols]
    df = df[primary_cols + other_cols]
    
    # Save to CSV
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"\nSaved: {output_path}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Splits analyzed: {len(df)}")
    print(f"k values: {df['best_k'].tolist()}")
    print(f"Mean k: {df['best_k'].mean():.2f}")
    print(f"Mean silhouette: {df['silhouette'].mean():.4f}")
    print(f"Mean entropy: {df['entropy_norm'].mean():.4f}")
    print(f"Mean accuracy: {df['accuracy_mean'].mean():.4f} ± {df['accuracy_std'].mean():.4f}")
    
    if not df['C_mean'].isna().all():
        print(f"C range: [{df['C_min'].min():.4f}, {df['C_max'].max():.4f}]")
    
    # Check if this looks like a fixed C experiment (all k=1)
    if (df['best_k'] == 1).all():
        print(f"\n*** FIXED C CONTROL: All splits show k=1 (explanations collapsed) ***")
    
    print(f"{'='*70}")
    
    # Also print the table
    print("\nFull table:")
    print(df[['split_seed', 'n_runs', 'best_k', 'silhouette', 'entropy_norm', 
              'accuracy_mean', 'cluster_sizes']].to_string(index=False))


if __name__ == "__main__":
    main()
