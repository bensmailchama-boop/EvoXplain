#!/usr/bin/env python3
"""
evoxplain_visualize_logreg_clustered.py

Visualizer for Logistic Regression outputs that performs clustering on-the-fly.

Key features:
- Performs K-means clustering with silhouette-based k selection
- L2 normalizes explanation vectors before clustering
- Shows distinct cluster colors
- Supports C value overlays (regularization strength)
- Shows regularization regime analysis

IMPORTANT: For universal manifold, k_max = n_splits because each split is 
an independent experiment and clusters cannot mix runs from different splits.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# -----------------------------
# Loading helpers
# -----------------------------
def _npz_get(data, keys, default=None):
    """Try multiple key names, return first match."""
    for k in keys:
        if k in data:
            return data[k]
    return default


def l2_normalize(X):
    """Center and L2 normalize each row."""
    mu = X.mean(axis=0)
    centered = X - mu
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normed = centered / (norms + 1e-12)
    return normed, mu


def find_best_k_silhouette(X_normed, k_min=2, k_max=8, seed=42):
    """
    Find optimal k using silhouette score.
    Returns (best_k, labels, silhouette_scores_dict, best_silhouette)
    """
    n = X_normed.shape[0]
    
    # Check for degenerate case (all vectors identical)
    if np.allclose(X_normed, X_normed[0], atol=1e-10):
        print("  [WARN] Degenerate case: all explanation vectors nearly identical")
        return 1, np.zeros(n, dtype=int), {}, None
    
    best_k = 1
    best_score = -1.0
    best_labels = np.zeros(n, dtype=int)
    sil_scores = {}
    
    for k in range(k_min, min(k_max + 1, n)):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X_normed)
        
        # Check if clustering collapsed
        if len(np.unique(labels)) < k:
            continue
            
        score = silhouette_score(X_normed, labels)
        sil_scores[k] = score
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels.copy()
    
    return best_k, best_labels, sil_scores, best_score if best_k > 1 else None


def load_and_cluster_split(npz_path: Path, k_min=2, k_max=8, seed=42):
    """
    Load a split NPZ file, perform clustering, and return arrays dict + meta dict.
    """
    meta = {
        "split_seed": None,
        "best_k": 1,
        "silhouette_scores": {},
        "entropy": 0.0,
    }

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"[ERROR] Failed to load {npz_path.name}: {e}")
        return None, meta

    available_keys = list(data.keys())
    
    # Load required fields
    importance = _npz_get(data, ["importance"])
    run_indices = _npz_get(data, ["run_indices"])

    if importance is None or run_indices is None:
        print(f"[ERROR] Missing required arrays in {npz_path.name}")
        print(f"  Available keys: {available_keys}")
        return None, meta
    
    importance = np.array(importance, dtype=float)
    
    # L2 normalize for clustering
    normed_importance, mu = l2_normalize(importance)
    
    # Perform clustering
    best_k, labels, sil_scores, best_sil = find_best_k_silhouette(
        normed_importance, k_min=k_min, k_max=k_max, seed=seed
    )
    
    # Compute entropy
    if best_k > 1:
        counts = np.array([(labels == i).sum() for i in range(best_k)])
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        entropy_norm = entropy / np.log2(best_k)  # Normalized entropy
    else:
        entropy = 0.0
        entropy_norm = 0.0

    # Accuracy
    acc = _npz_get(data, ["accs", "acc", "accuracy"], default=None)
    if acc is None:
        acc = np.full(shape=(len(run_indices),), fill_value=np.nan, dtype=float)

    # Overlays
    overlays = {}
    
    # C values for logistic regression (primary overlay)
    c_vals = _npz_get(data, ["c_values", "C"])
    if c_vals is not None and len(c_vals) > 0 and not np.all(np.isnan(c_vals)):
        overlays["C"] = np.array(c_vals, dtype=float)
        overlays["log_C"] = np.log10(np.array(c_vals, dtype=float) + 1e-10)

    arrays = {
        "importance": importance,
        "normed_importance": normed_importance,
        "labels": np.array(labels),
        "run_indices": np.array(run_indices),
        "acc": np.array(acc, dtype=float),
        "overlays": overlays,
    }
    
    # Extract split seed from filename
    try:
        stem = npz_path.stem
        if "split" in stem:
            seed_str = stem.split("split")[-1]
            meta["split_seed"] = int(seed_str)
    except:
        pass
    
    meta["best_k"] = best_k
    meta["silhouette_scores"] = sil_scores
    meta["entropy"] = entropy
    meta["entropy_norm"] = entropy_norm

    return arrays, meta


# -----------------------------
# Plotting helpers
# -----------------------------
def resolve_overlay(arrays: dict, overlay: str):
    """Resolve overlay name to actual array and label."""
    overlays = arrays.get("overlays", {})
    
    if overlay.lower() in ["acc", "accuracy"]:
        return arrays.get("acc", None), "Accuracy"
    
    if overlay.lower() in ["c", "regularization"]:
        return overlays.get("C", None), "C (Regularization)"
    
    if overlay.lower() in ["log_c", "logc"]:
        return overlays.get("log_C", None), "log₁₀(C)"
    
    if overlay in overlays:
        return overlays[overlay], overlay
        
    return None, None


def pca_project(X: np.ndarray):
    """Project data to 2D via PCA."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    return X_pca, evr, pca


def get_cluster_colors(n_clusters):
    """Get distinct colors for clusters."""
    if n_clusters <= 10:
        # Use tab10 for up to 10 clusters
        cmap = plt.cm.tab10
        return [cmap(i) for i in range(n_clusters)]
    else:
        # Use viridis for more clusters
        cmap = plt.cm.viridis
        return [cmap(i / n_clusters) for i in range(n_clusters)]


def plot_split_manifold_clustered(split_id: str, arrays: dict, meta: dict, 
                                   out_path: Path, overlay: str, use_normed: bool):
    """Plot a single split's explanation manifold with clustering."""
    
    if use_normed:
        X = arrays["normed_importance"]
    else:
        X = arrays["importance"]
        
    labels = arrays["labels"]
    best_k = meta["best_k"]
    entropy_norm = meta.get("entropy_norm", 0.0)
    
    X_pca, evr, pca = pca_project(X)
    
    # Global mean projection
    ghost = pca.transform(X.mean(axis=0).reshape(1, -1))

    pc1_lbl = f"PC1 ({evr[0]:.1%} var)"
    pc2_lbl = f"PC2 ({evr[1]:.1%} var)"

    fig = plt.figure(figsize=(16, 7))

    # --- PANEL A: Clusters ---
    ax1 = fig.add_subplot(1, 2, 1)
    
    unique_labels = np.unique(labels)
    colors = get_cluster_colors(len(unique_labels))
    
    # Store centroids for plotting after scatter
    centroids_pca = []
    
    # Calculate C stats per cluster for legend
    C_vals = arrays["overlays"].get("C", None)
    
    for i, lbl in enumerate(unique_labels):
        mask = (labels == lbl)
        n_in_cluster = mask.sum()
        
        # Build legend label with C stats if available
        legend_label = f"Cluster {lbl} (n={n_in_cluster})"
        if C_vals is not None:
            cluster_C = C_vals[mask]
            mean_C = np.mean(cluster_C)
            legend_label = f"Cluster {lbl} (n={n_in_cluster}, C̄={mean_C:.2f})"
        
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   color=colors[i], s=20, alpha=0.6, 
                   label=legend_label)
        
        # Compute and store centroid in PCA space
        centroid_pca = X_pca[mask].mean(axis=0)
        centroids_pca.append((centroid_pca, colors[i], lbl))
    
    # Plot cluster centroids with star markers
    for centroid_pca, color, lbl in centroids_pca:
        ax1.scatter(centroid_pca[0], centroid_pca[1], 
                   c=[color], marker="*", s=400, edgecolors="black", 
                   linewidths=1.5, zorder=11)

    # Global Mean
    ax1.scatter(ghost[0, 0], ghost[0, 1], c="black", marker="X", s=200, 
               label="Average Attribution", zorder=10)

    space_tag = "Normed" if use_normed else "Raw"
    ax1.set_title(f"{split_id}: Explanation Clusters\n"
                  f"k={best_k}, H_norm={entropy_norm:.3f}, N={len(X)}")
    ax1.set_xlabel(pc1_lbl)
    ax1.set_ylabel(pc2_lbl)
    ax1.legend(loc='best', fontsize=8)

    # --- PANEL B: Overlay ---
    ax2 = fig.add_subplot(1, 2, 2)
    ov, ov_label = resolve_overlay(arrays, overlay=overlay)

    if ov is None or np.all(np.isnan(ov)):
        ov = arrays.get("acc", np.zeros(len(labels)))
        ov_label = "Accuracy (Default)"
        
    # Contrast stretching for accuracy
    vmin, vmax = None, None
    if "ccurac" in ov_label.lower():
        valid = ov[~np.isnan(ov)]
        if len(valid) > 0:
            med = np.median(valid)
            std = np.std(valid)
            if std > 0:
                vmin = max(0, med - 2.5*std)
                vmax = min(1, med + 2.5*std)

    # Use log scale colormap for C values
    cmap = "viridis" if "C" in ov_label else "plasma"
    
    sc2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=ov, cmap=cmap, 
                     s=20, alpha=0.7, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(sc2, ax=ax2)
    cb.set_label(ov_label)

    ax2.set_title(f"{split_id}: {ov_label}")
    ax2.set_xlabel(pc1_lbl)
    ax2.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.name} (k={best_k}, entropy={entropy_norm:.3f})")


def plot_universal_clustered(files, out_path: Path, overlay: str, use_normed: bool,
                              k_min=2, k_max=None, seed=42):
    """
    Plot universal manifold stacking all splits.
    
    IMPORTANT: Each split is clustered INDEPENDENTLY because they are separate
    experiments with different train/test data. We do NOT re-cluster across splits.
    The universal manifold shows all splits together, colored by their per-split clusters.
    """
    
    n_splits = len(files)
    per_split_k_max = k_max if k_max else 8  # k_max for within-split clustering
    
    print(f"\nBuilding Universal Manifold from {n_splits} splits...")
    print(f"  NOTE: Each split is clustered INDEPENDENTLY (k_max={per_split_k_max})")
    print(f"        Clusters do NOT mix runs from different splits.")
    
    # Load and cluster each split separately
    all_importance = []
    all_acc = []
    all_C = []
    all_split_ids = []
    all_labels = []  # Per-split cluster labels
    
    split_summaries = []
    
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
        except:
            continue
            
        imp = data.get("importance")
        if imp is None:
            continue
        
        n_runs = len(imp)
        importance = np.array(imp, dtype=float)
        
        # Cluster THIS split independently
        normed_imp, mu = l2_normalize(importance)
        best_k, labels, sil_scores, best_sil = find_best_k_silhouette(
            normed_imp, k_min=k_min, k_max=per_split_k_max, seed=seed
        )
        
        # Track split membership
        try:
            split_seed = int(f.stem.split("split")[-1])
        except:
            split_seed = 0
        
        all_importance.append(importance)
        all_split_ids.append(np.full(n_runs, split_seed))
        all_labels.append(labels)  # Per-split cluster labels
        
        acc = _npz_get(data, ["accs", "acc", "accuracy"])
        if acc is not None:
            all_acc.append(np.array(acc, dtype=float))
        else:
            all_acc.append(np.full(n_runs, np.nan))
        
        # C values
        c_vals = _npz_get(data, ["c_values", "C"])
        if c_vals is not None and len(c_vals) > 0:
            all_C.append(np.array(c_vals, dtype=float))
        else:
            all_C.append(np.full(n_runs, np.nan))
        
        # Compute entropy for this split
        if best_k > 1:
            counts = np.array([(labels == i).sum() for i in range(best_k)])
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy_norm = -np.sum(probs * np.log2(probs)) / np.log2(best_k)
        else:
            entropy_norm = 0.0
        
        split_summaries.append({
            'split_seed': split_seed,
            'n_runs': n_runs,
            'best_k': best_k,
            'entropy_norm': entropy_norm,
            'silhouette': best_sil,
        })
        
        print(f"  Split {split_seed}: n={n_runs}, k={best_k}, H={entropy_norm:.3f}")
    
    if not all_importance:
        print("[ERROR] No valid data found!")
        return
    
    # Stack
    importance = np.vstack(all_importance)
    acc = np.concatenate(all_acc)
    C_values = np.concatenate(all_C)
    split_ids = np.concatenate(all_split_ids)
    per_split_labels = np.concatenate(all_labels)
    n = len(importance)
    
    print(f"\n  Total runs: {n}")
    print(f"  Runs per split: {n // n_splits}")
    
    # Create unique global labels: split_id * 100 + cluster_id
    # This ensures clusters from different splits are visually distinct
    unique_splits = np.unique(split_ids)
    global_labels = np.zeros(n, dtype=int)
    for i, split_id in enumerate(unique_splits):
        mask = (split_ids == split_id)
        global_labels[mask] = i * 100 + per_split_labels[mask]
    
    # L2 normalize for visualization (but NOT for re-clustering)
    normed_importance, mu = l2_normalize(importance)
    
    # Select data for plotting
    if use_normed:
        X = normed_importance
    else:
        X = importance
    
    # PCA
    X_pca, evr, pca = pca_project(X)
    ghost = pca.transform(X.mean(axis=0).reshape(1, -1))

    pc1_lbl = f"PC1 ({evr[0]:.1%} var)"
    pc2_lbl = f"PC2 ({evr[1]:.1%} var)"

    fig = plt.figure(figsize=(16, 7))

    # --- PANEL A: Color by Split ID ---
    ax1 = fig.add_subplot(1, 2, 1)
    
    colors = get_cluster_colors(n_splits)
    
    for i, split_id in enumerate(unique_splits):
        mask = (split_ids == split_id)
        n_in_split = mask.sum()
        split_k = split_summaries[i]['best_k']
        split_H = split_summaries[i]['entropy_norm']
        
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   color=colors[i], s=8, alpha=0.5, 
                   label=f"Split {int(split_id)} (k={split_k}, H={split_H:.2f})")
    
    # Global Mean
    ax1.scatter(ghost[0, 0], ghost[0, 1], c="black", marker="X", s=200, 
               label="Global Mean", zorder=10)

    space_tag = "Normed" if use_normed else "Raw"
    mean_k = np.mean([s['best_k'] for s in split_summaries])
    mean_H = np.mean([s['entropy_norm'] for s in split_summaries])
    
    ax1.set_title(f"Universal Manifold: By Split ({space_tag})\n"
                  f"n_splits={n_splits}, N={n}")
    ax1.set_xlabel(pc1_lbl)
    ax1.set_ylabel(pc2_lbl)
    ax1.legend(loc='best', fontsize=7)

    # --- PANEL B: Overlay ---
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Select overlay
    if overlay.lower() in ["c", "regularization"]:
        ov = C_values
        ov_label = "C (Regularization)"
    elif overlay.lower() in ["log_c", "logc"]:
        ov = np.log10(C_values + 1e-10)
        ov_label = "log₁₀(C)"
    else:
        ov = acc
        ov_label = "Accuracy"

    # Contrast stretching for accuracy
    vmin, vmax = None, None
    if "ccurac" in ov_label.lower():
        valid = ov[~np.isnan(ov)]
        if len(valid) > 0:
            med = np.median(valid)
            std = np.std(valid)
            if std > 0:
                vmin = max(0, med - 2.5*std)
                vmax = min(1, med + 2.5*std)

    cmap = "viridis" if "C" in ov_label else "plasma"
    
    sc2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=ov, cmap=cmap, 
                     s=8, alpha=0.6, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(sc2, ax=ax2)
    cb.set_label(ov_label)

    ax2.set_title(f"Universal Manifold: {ov_label}")
    ax2.set_xlabel(pc1_lbl)
    ax2.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path.name}")
    
    # Print per-split C analysis
    if not np.all(np.isnan(C_values)):
        print(f"\n  Regularization (C) analysis per SPLIT:")
        for i, split_id in enumerate(unique_splits):
            mask = (split_ids == split_id)
            split_C = C_values[mask]
            split_C = split_C[~np.isnan(split_C)]
            if len(split_C) > 0:
                print(f"    Split {int(split_id)}: n={mask.sum()}, "
                      f"C_mean={np.mean(split_C):.3f}, "
                      f"C_range=[{np.min(split_C):.3f}, {np.max(split_C):.3f}]")
    
    return {
        'n_splits': n_splits,
        'n_total': n,
        'split_summaries': split_summaries,
        'mean_k': mean_k,
        'mean_entropy': mean_H,
    }


def plot_logreg_param_grid_clustered(files, out_dir: Path, use_normed: bool,
                                      k_min=2, k_max=None, seed=42):
    """
    Create a grid showing clusters + C distribution + accuracy.
    
    Each split is clustered INDEPENDENTLY - we don't re-cluster across splits.
    """
    n_splits = len(files)
    per_split_k_max = k_max if k_max else 8
    
    print(f"\nBuilding LogReg parameter grid...")
    print(f"  Each split clustered independently (k_max={per_split_k_max})")
    
    # Load and cluster each split separately
    all_importance = []
    all_acc = []
    all_C = []
    all_split_ids = []
    all_labels = []
    
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
        except:
            continue
            
        imp = data.get("importance")
        if imp is None:
            continue
            
        n_runs = len(imp)
        importance = np.array(imp, dtype=float)
        
        # Cluster THIS split independently
        normed_imp, mu = l2_normalize(importance)
        best_k, labels, sil_scores, best_sil = find_best_k_silhouette(
            normed_imp, k_min=k_min, k_max=per_split_k_max, seed=seed
        )
        
        # Track split membership
        try:
            split_seed = int(f.stem.split("split")[-1])
        except:
            split_seed = 0
        
        all_importance.append(importance)
        all_split_ids.append(np.full(n_runs, split_seed))
        all_labels.append(labels)
        
        acc = _npz_get(data, ["accs", "acc", "accuracy"])
        all_acc.append(np.array(acc, dtype=float) if acc is not None 
                       else np.full(n_runs, np.nan))
        
        c_vals = _npz_get(data, ["c_values", "C"])
        if c_vals is not None and len(c_vals) > 0:
            all_C.append(np.array(c_vals, dtype=float))
        else:
            all_C.append(np.full(n_runs, np.nan))
    
    if not all_importance:
        print("[ERROR] No valid data!")
        return
        
    # Stack
    importance = np.vstack(all_importance)
    acc = np.concatenate(all_acc)
    C_values = np.concatenate(all_C)
    split_ids = np.concatenate(all_split_ids)
    per_split_labels = np.concatenate(all_labels)
    
    n = len(importance)
    
    # L2 normalize for visualization
    normed_importance, mu = l2_normalize(importance)
    
    # Select space
    X = normed_importance if use_normed else importance
    
    # PCA
    X_pca, evr, pca = pca_project(X)
    
    # Create 2x2 grid: splits, accuracy, C, log(C)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Panel 0: Color by Split
    ax = axes[0]
    unique_splits = np.unique(split_ids)
    colors = get_cluster_colors(len(unique_splits))
    
    for i, split_id in enumerate(unique_splits):
        mask = (split_ids == split_id)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  color=colors[i], s=8, alpha=0.5, 
                  label=f"Split {int(split_id)} (n={mask.sum()})")
    
    ax.set_title(f"By Split (n_splits={n_splits})")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.legend(fontsize=6, loc='best')
    
    # Panel 1: Accuracy
    ax = axes[1]
    valid_acc = acc[~np.isnan(acc)]
    if len(valid_acc) > 0:
        med = np.median(valid_acc)
        std = np.std(valid_acc)
        vmin = max(0, med - 3*std) if std > 0 else None
        vmax = min(1, med + 3*std) if std > 0 else None
    else:
        vmin, vmax = None, None
    
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=acc, cmap="plasma", 
                   s=8, alpha=0.5, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Accuracy")
    ax.set_title("Accuracy Overlay")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    
    # Panel 2: C (raw)
    ax = axes[2]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=C_values, cmap="viridis", 
                   s=8, alpha=0.5)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("C (Regularization)")
    ax.set_title("Regularization Strength (C)")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    
    # Panel 3: log(C)
    ax = axes[3]
    log_C = np.log10(C_values + 1e-10)
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=log_C, cmap="coolwarm", 
                   s=8, alpha=0.5)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log₁₀(C)")
    ax.set_title("log₁₀(C) - Regularization Regime")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    
    space_tag = "Normed" if use_normed else "Raw"
    fig.suptitle(f"LogReg Explanation Space ({space_tag}, N={n}, n_splits={n_splits})", 
                fontsize=14, y=1.02)
    fig.tight_layout()
    
    out_path = out_dir / f"LogReg_param_grid_clustered_{space_tag}.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved LogReg param grid: {out_path.name}")


def main():
    ap = argparse.ArgumentParser(description="Visualize LogReg EvoXplain results with clustering")
    ap.add_argument("--input_dir", type=str, required=True, 
                   help="Folder containing aggregate_split*.npz")
    ap.add_argument("--overlay", type=str, default="acc", 
                   help="Overlay variable (acc, C, log_C)")
    ap.add_argument("--space", type=str, default="normed", choices=["raw", "normed"],
                   help="Use normed (L2-normalized) or raw importance space")
    ap.add_argument("--universal", action="store_true", 
                   help="Also plot universal manifold")
    ap.add_argument("--param_grid", action="store_true", 
                   help="Plot LogReg parameter grid with clusters")
    ap.add_argument("--k_min", type=int, default=2, help="Min k for silhouette scan")
    ap.add_argument("--k_max", type=int, default=None, 
                   help="Max k for silhouette scan (default: n_splits for universal, 8 for per-split)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for clustering")
    
    args = ap.parse_args()
    use_normed = (args.space == "normed")

    input_dir = Path(args.input_dir)
    
    # Match core engine output pattern
    pattern = "aggregate_split*.npz"
    files = sorted(input_dir.glob(pattern))
    
    if not files:
        print(f"[CRITICAL] No files found matching {pattern} in {input_dir}")
        print(f"  Contents of {input_dir}:")
        for f in input_dir.iterdir():
            print(f"    {f.name}")
        raise SystemExit(1)

    out_dir = input_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_splits = len(files)
    
    # For per-split analysis, use k_max=8 (or user-specified)
    # For universal analysis, use k_max=n_splits (unless user overrides)
    per_split_k_max = args.k_max if args.k_max else 8
    universal_k_max = args.k_max if args.k_max else n_splits

    print(f"Found {n_splits} split files")
    print(f"Visualizing in '{args.space}' space with overlay='{args.overlay}'")
    print(f"Clustering: k_min={args.k_min}, per-split k_max={per_split_k_max}, universal k_max={universal_k_max}")
    print(f"Output directory: {out_dir}")
    print()

    # Summary stats
    all_k = []
    all_entropy = []

    # Plot Individual Splits with clustering
    for f in files:
        split_id = f.stem
        print(f"Processing {split_id}...")
        
        arrays, meta = load_and_cluster_split(
            f, k_min=args.k_min, k_max=per_split_k_max, seed=args.seed
        )
        if arrays is None: 
            continue
        
        all_k.append(meta["best_k"])
        all_entropy.append(meta.get("entropy_norm", 0))
        
        # Show available overlays for first file
        if f == files[0]:
            print(f"  Available overlays: {list(arrays['overlays'].keys())}")
        
        fname = f"{split_id}_{args.space}_{args.overlay}.png"
        plot_split_manifold_clustered(
            split_id, arrays, meta, out_dir / fname, 
            overlay=args.overlay, use_normed=use_normed
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"Per-split summary:")
    print(f"  k values: {all_k}")
    print(f"  Mean k: {np.mean(all_k):.2f}")
    print(f"  Entropy (norm): {all_entropy}")
    print(f"  Mean entropy: {np.mean(all_entropy):.3f}")
    print(f"{'='*60}")

    # Plot Universal with clustering (k_max = n_splits)
    if args.universal:
        fname = f"UNIVERSAL_{args.space}_{args.overlay}.png"
        summary = plot_universal_clustered(
            files, out_dir / fname, overlay=args.overlay, 
            use_normed=use_normed, k_min=args.k_min, k_max=universal_k_max, seed=args.seed
        )

    # Plot LogReg Parameter Grid with clustering (k_max = n_splits)
    if args.param_grid:
        plot_logreg_param_grid_clustered(
            files, out_dir, use_normed=use_normed,
            k_min=args.k_min, k_max=universal_k_max, seed=args.seed
        )
        
    print("\nDone!")


if __name__ == "__main__":
    main()
