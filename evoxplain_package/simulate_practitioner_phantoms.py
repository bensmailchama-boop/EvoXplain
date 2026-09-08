#!/usr/bin/env python3
"""
simulate_practitioner_phantoms.py

Simulates 50 independent bioinformatician CV-grid runs on TCGA-LR-Cgrid LIME/SHAP.

Each simulated practitioner:
  - Draws 10 random splits (without replacement) from the 100 available.
  - Within each split, scans 10 log-spaced C values in [0.01, 10].
  - For each (split, C) pair, picks the bootstrap run with the closest
    actual C value (nearest-neighbour on log10 C).
  - Stacks 100 explanation vectors (10 splits x 10 C values) and averages
    them to produce the 'phantom explanation' — what they would report.
  - Records the phantom's top-50 genes by |weight|.

Outputs (in OUT_DIR):
  - phantom_gene_lists/practitioner_NNN_top50.tsv  (50 files, gprofiler-ready)
  - phantom_metadata.csv     one row per practitioner: mean acc, C grid hits,
                              split seeds drawn, basin assignment of constituent
                              runs (so we can show 'each phantom is a mixture')
  - phantom_centroids.npz    full 1000-dim phantom vectors (50, 1000)
                              for downstream geometric analysis if needed

Run on HPC via run_practitioner_phantoms.csh
"""

import os
import sys
import json
import glob
import re
import argparse
import numpy as np
import pandas as pd


def _split_id(path):
    m = re.search(r"split(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_split(path, lens="lime"):
    """Load a single aggregate_split*.npz, return what we need or None."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        sys.stderr.write(f"WARN: failed to load {path}: {e}\n")
        return None
    required = [
        f"expvec_normed_{lens}",
        f"cluster_labels_{lens}",
        "run_C_values",
        "test_acc",
        "feature_names",
    ]
    if not all(k in z.files for k in required):
        missing = [k for k in required if k not in z.files]
        sys.stderr.write(f"WARN: {path} missing keys {missing}\n")
        return None
    return {
        "seed": _split_id(path),
        "X": z[f"expvec_normed_{lens}"],            # (100, p)
        "labels": z[f"cluster_labels_{lens}"],       # (100,)
        "C": z["run_C_values"],                      # (100,)
        "acc": z["test_acc"],                        # (100,)
        "features": z["feature_names"],              # (p,) — ensembl IDs versioned
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir",
                    help="dir containing aggregate_split*.npz "
                         "(e.g., /home2/chamabens/evoxplain/results/final_freeze/"
                         "tcga_lr_cgrid_cross_split_cosine OR wherever the per-split "
                         "aggregates live — check both locations).")
    ap.add_argument("output_dir",
                    help="where to write phantom outputs")
    ap.add_argument("--lens", default="lime", choices=["lime", "shap"])
    ap.add_argument("--n_practitioners", type=int, default=50,
                    help="number of simulated bioinformaticians (default 50)")
    ap.add_argument("--n_splits_per_practitioner", type=int, default=10,
                    help="folds per practitioner (default 10)")
    ap.add_argument("--n_C_per_practitioner", type=int, default=10,
                    help="C-grid points per practitioner (default 10)")
    ap.add_argument("--C_min", type=float, default=0.01,
                    help="lower bound of practitioner C-grid (default 0.01)")
    ap.add_argument("--C_max", type=float, default=10.0,
                    help="upper bound of practitioner C-grid (default 10.0)")
    ap.add_argument("--rng_seed", type=int, default=2026,
                    help="master RNG seed for reproducibility")
    ap.add_argument("--top_n_genes", type=int, default=50,
                    help="top-N genes per phantom for gprofiler enrichment")
    ap.add_argument("--glob_pattern", default="aggregate_split*.npz")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gene_list_dir = os.path.join(args.output_dir, "phantom_gene_lists")
    os.makedirs(gene_list_dir, exist_ok=True)

    # 1. Find all per-split aggregates
    paths = sorted(glob.glob(os.path.join(args.input_dir, args.glob_pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(args.input_dir, "**",
                                              args.glob_pattern),
                                 recursive=True))
    if not paths:
        sys.exit(f"ERROR: no files matching {args.glob_pattern} under "
                 f"{args.input_dir}. Per-split aggregates must include "
                 f"expvec_normed_{args.lens} (the individual 100 bootstrap "
                 f"vectors), not just the per-split mean.")
    print(f"[sim] found {len(paths)} candidate per-split aggregate files")

    # 2. Load all splits (memory-cheap: float32 helps; 100 splits x 100 runs x
    #    ~17k genes = ~700 MB at float32, fits comfortably on a compute node)
    splits = []
    for p in paths:
        d = load_split(p, args.lens)
        if d is not None:
            splits.append(d)
    print(f"[sim] loaded {len(splits)} splits with expvec_normed_{args.lens}")
    if len(splits) < args.n_splits_per_practitioner:
        sys.exit(f"ERROR: only {len(splits)} splits available; need "
                 f"{args.n_splits_per_practitioner} per practitioner")

    # Sanity-check feature alignment across splits
    p_dim = splits[0]["X"].shape[1]
    features_ref = splits[0]["features"]
    for s in splits[1:]:
        if s["X"].shape[1] != p_dim or not np.array_equal(s["features"],
                                                            features_ref):
            sys.exit(f"ERROR: feature mismatch in split {s['seed']}")
    print(f"[sim] feature dimension consistent across splits: p={p_dim}")

    # 3. Define the practitioner's C-grid
    practitioner_C_grid = np.logspace(np.log10(args.C_min),
                                       np.log10(args.C_max),
                                       args.n_C_per_practitioner)
    print(f"[sim] practitioner C-grid ({args.n_C_per_practitioner} pts in "
          f"[{args.C_min}, {args.C_max}]):")
    print(f"      {practitioner_C_grid}")

    rng = np.random.default_rng(args.rng_seed)
    metadata_rows = []
    phantom_centroids = np.zeros((args.n_practitioners, p_dim), dtype=np.float64)

    for prac_i in range(args.n_practitioners):
        # (a) Sample 10 splits without replacement
        chosen_split_idx = rng.choice(len(splits),
                                       size=args.n_splits_per_practitioner,
                                       replace=False)
        chosen_seeds = [int(splits[i]["seed"]) for i in chosen_split_idx]

        # (b) For each chosen split, nearest-neighbour the practitioner C-grid
        #     against that split's actual C-values; gather one bootstrap run
        #     per C-grid point. (Same run may be picked twice if the split's
        #     C-coverage is sparse near the practitioner's chosen C — we keep
        #     duplicates because that reflects what the practitioner would do.)
        explanation_vectors = []
        basin_picks = []
        C_picks = []
        acc_picks = []
        run_indices = []

        for s_i in chosen_split_idx:
            sd = splits[s_i]
            log_actual = np.log10(sd["C"])
            for target_C in practitioner_C_grid:
                log_target = np.log10(target_C)
                # find the run whose log10(C) is closest to log10(target_C)
                idx = int(np.argmin(np.abs(log_actual - log_target)))
                explanation_vectors.append(sd["X"][idx])
                basin_picks.append(int(sd["labels"][idx]))
                C_picks.append(float(sd["C"][idx]))
                acc_picks.append(float(sd["acc"][idx]))
                run_indices.append((int(sd["seed"]), idx))

        # (c) Build the phantom = mean across all selected explanation vectors
        explanation_vectors = np.stack(explanation_vectors, axis=0)
        phantom = explanation_vectors.mean(axis=0)
        # store BOTH raw mean and re-normalised; phantom for top-genes uses
        # the raw mean (matches what a practitioner would inspect)
        phantom_centroids[prac_i] = phantom

        # (d) Top-N genes by |weight|
        order = np.argsort(np.abs(phantom))[::-1][:args.top_n_genes]
        top_features_versioned = [str(features_ref[i]) for i in order]
        top_features_unversioned = [v.split(".")[0]
                                     for v in top_features_versioned]
        top_weights = [float(phantom[i]) for i in order]

        # Write gprofiler-ready TSV
        gene_list_path = os.path.join(gene_list_dir,
                                       f"practitioner_{prac_i:03d}_top50.tsv")
        with open(gene_list_path, "w") as f:
            f.write("ensembl_id_versioned\tensembl_id\tweight\n")
            for v_id, uv_id, w in zip(top_features_versioned,
                                       top_features_unversioned,
                                       top_weights):
                f.write(f"{v_id}\t{uv_id}\t{w:.6f}\n")

        # (e) Record metadata
        # Basin coverage stats — the key forensic detail: how often did the
        # practitioner's grid hit each basin?
        basin_counts = {b: basin_picks.count(b)
                         for b in sorted(set(basin_picks))}

        # Modified dict unpacked dynamically mapped for k basins
        metadata_rows.append({
            "practitioner_id":         prac_i,
            "n_fits":                  len(explanation_vectors),
            "split_seeds_chosen":      ";".join(map(str, chosen_seeds)),
            "mean_acc_across_fits":    float(np.mean(acc_picks)),
            "std_acc_across_fits":     float(np.std(acc_picks)),
            **{f"frac_fits_in_basin{b}": basin_counts.get(b, 0) / len(basin_picks)
               for b in range(max(basin_counts.keys()) + 1)},
            "min_C_actual":            float(np.min(C_picks)),
            "max_C_actual":            float(np.max(C_picks)),
            "phantom_L2_norm":         float(np.linalg.norm(phantom)),
            "gene_list_path":          os.path.relpath(gene_list_path,
                                                         args.output_dir),
        })

        if (prac_i + 1) % 10 == 0:
            print(f"[sim] completed practitioner {prac_i+1}/"
                  f"{args.n_practitioners}")

    # 4. Write metadata CSV
    df = pd.DataFrame(metadata_rows)
    # Fill NA with 0.0 in case a basin was entirely missed by some practitioners but hit by others
    basin_cols = [c for c in df.columns if c.startswith("frac_fits_in_basin")]
    df[basin_cols] = df[basin_cols].fillna(0.0)
    
    metadata_path = os.path.join(args.output_dir, "phantom_metadata.csv")
    df.to_csv(metadata_path, index=False)
    print(f"\n[sim] metadata written to {metadata_path}")

    # 5. Save phantom centroids for downstream geometric analysis
    centroids_path = os.path.join(args.output_dir, "phantom_centroids.npz")
    np.savez_compressed(centroids_path,
                         phantoms=phantom_centroids,
                         feature_names=features_ref)
    print(f"[sim] phantom centroids written to {centroids_path}")

    # 6. Summary
    print(f"\n[sim] === SUMMARY ===")
    print(f"  practitioners simulated:     {args.n_practitioners}")
    print(f"  fits per practitioner:       "
          f"{args.n_splits_per_practitioner * args.n_C_per_practitioner}")
    
    # Dynamically print mean/std for each basin column
    for col in sorted(basin_cols):
        b_num = col.replace("frac_fits_in_basin", "")
        print(f"  mean fraction in basin {b_num}:    "
              f"{df[col].mean():.3f} ± {df[col].std():.3f}")

    print(f"  fits-mean accuracy:          "
          f"{df['mean_acc_across_fits'].mean():.4f} "
          f"± {df['std_acc_across_fits'].mean():.4f}")
    print(f"\n  Gene lists ready for enrichment in:")
    print(f"    {gene_list_dir}")
    print(f"  Next step: run enrich_phantoms.py on these gene lists.")


if __name__ == "__main__":
    main()
