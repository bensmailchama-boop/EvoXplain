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


def find_best_k_silhouette(X_normed, k_max=8, seed=42, silhouette_threshold=0.25):
    """
    Find optimal k, allowing k=1 as null hypothesis.
    k>1 accepted only if silhouette > threshold.
    """
    n = X_normed.shape[0]
    
    # Check for degenerate case (all vectors identical)
    if np.allclose(X_normed, X_normed[0], atol=1e-10):
        return 1, np.zeros(n, dtype=int), {}, None
    
    # Check for negligible variance
    if np.var(X_normed, axis=0).sum() < 1e-10:
        return 1, np.zeros(n, dtype=int), {}, None
    
    best_k = 1
    best_score = -1.0
    best_labels = np.zeros(n, dtype=int)
    sil_scores = {}
    
    for k in range(2, min(k_max + 1, n)):
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
    
    # Accept k>1 only if silhouette exceeds threshold
    if best_k > 1 and best_score < silhouette_threshold:
        return 1, np.zeros(n, dtype=int), sil_scores, None
    
    return best_k, best_labels, sil_scores, best_score if best_k > 1 else None


def analyze_split(npz_path, k_max=8, seed=42, silhouette_threshold=0.25):
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
        normed_importance, k_max=k_max, seed=seed, silhouette_threshold=silhouette_threshold
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
        description="Generate per-split summary for EvoXplain experiments"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing aggregate_split*.npz files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (default: input_dir/per_split_summary)")
    parser.add_argument("--k_max", type=int, default=8,
                       help="Maximum k for silhouette scan")
    parser.add_argument("--silhouette_threshold", type=float, default=0.25,
                       help="Min silhouette to accept k>1 (default 0.25)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for clustering")
    parser.add_argument("--latex", action="store_true",
                       help="Output LaTeX table format")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_base = Path(args.output) if args.output else input_dir / "per_split_summary"
    
    # Find all split files
    files = sorted(input_dir.glob("aggregate_split*.npz"))
    
    if not files:
        raise FileNotFoundError(f"No aggregate_split*.npz files found in {input_dir}")
    
    print(f"Found {len(files)} split files")
    print(f"Clustering: k_max={args.k_max}, silhouette_threshold={args.silhouette_threshold}")
    print(f"  (k=1 is null hypothesis; k>1 accepted only if silhouette > threshold)")
    print()
    
    # Analyze each split
    all_metrics = []
    
    for f in files:
        print(f"Processing {f.name}...")
        metrics = analyze_split(f, args.k_max, args.seed, args.silhouette_threshold)
        if metrics:
            all_metrics.append(metrics)
            sil_str = f"{metrics['silhouette']:.3f}" if not np.isnan(metrics['silhouette']) else "—"
            print(f"  k={metrics['best_k']}, sil={sil_str}, "
                  f"H={metrics['entropy_norm']:.3f}, acc={metrics['accuracy_mean']:.3f}")
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns for clarity
    primary_cols = ['split_seed', 'n_runs', 'best_k', 'silhouette', 'entropy_norm', 
                    'accuracy_mean', 'accuracy_std', 'C_mean', 'C_std', 'C_min', 'C_max', 
                    'cluster_sizes']
    other_cols = [c for c in df.columns if c not in primary_cols]
    df = df[primary_cols + other_cols]
    
    # Save CSV
    csv_path = str(output_base) + ".csv"
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nSaved CSV: {csv_path}")
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{tabular}{ccccccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Split & Runs & $k$ & Silh.\ ($k$) & $H_{\mathrm{norm}}$ & Acc.\ mean & Acc.\ std \\")
    latex_lines.append(r"\midrule")
    
    for _, row in df.iterrows():
        split = int(row['split_seed'])
        runs = int(row['n_runs'])
        k = int(row['best_k'])
        sil = f"{row['silhouette']:.3f}" if not np.isnan(row['silhouette']) else "—"
        H = f"{row['entropy_norm']:.3f}"
        acc_mean = f"{row['accuracy_mean']:.3f}"
        acc_std = f"{row['accuracy_std']:.3f}"
        
        latex_lines.append(f"{split} & {runs} & {k} & {sil} & {H} & {acc_mean} & {acc_std} \\\\")
    
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    
    latex_table = "\n".join(latex_lines)
    
    # Save LaTeX
    latex_path = str(output_base) + ".tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX: {latex_path}")
    
    # Print LaTeX to console
    print(f"\n{'='*70}")
    print("LaTeX TABLE")
    print(f"{'='*70}")
    print(latex_table)
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Splits analyzed: {len(df)}")
    print(f"k values: {df['best_k'].tolist()}")
    print(f"Mean k: {df['best_k'].mean():.2f}")
    if not df['silhouette'].isna().all():
        print(f"Mean silhouette: {df['silhouette'].mean():.3f}")
    print(f"Mean entropy: {df['entropy_norm'].mean():.3f}")
    print(f"Mean accuracy: {df['accuracy_mean'].mean():.3f} ± {df['accuracy_std'].mean():.3f}")
    
    if not df['C_mean'].isna().all():
        print(f"C range: [{df['C_min'].min():.4f}, {df['C_max'].max():.4f}]")
    
    # Check if this looks like a fixed C experiment (all k=1)
    if (df['best_k'] == 1).all():
        print(f"\n*** FIXED C CONTROL: All splits show k=1 (single basin) ***")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
