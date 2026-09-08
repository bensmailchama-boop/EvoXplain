#!/usr/bin/env python3
"""
enrich_phantoms.py

Enriches each simulated-practitioner phantom gene list with gprofiler,
then produces a phantom_recurrence_summary.json directly comparable to
the per-basin recurrence files (lime_recurrence_summary.json,
shap_recurrence_summary.json).

Outputs:
  phantom_per_practitioner_enrichment_records.json    one record per phantom
  phantom_recurrence_summary.json                     across-practitioner recurrence

Schema of phantom_recurrence_summary.json matches the per-basin schema:
  {
    "lens": "lime",
    "n_practitioners": 50,
    "phantom_term_recurrence": [
      {"source": "...", "native": "...", "name": "...", "count": N,
       "fraction_of_practitioners": N/50}, ...
    ]
  }

Dependency: gprofiler-official.  Install once on the cluster with:
    pip install --user --break-system-packages gprofiler-official
"""

import os
import sys
import json
import glob
import argparse

import pandas as pd

try:
    from gprofiler import GProfiler
except ImportError:
    sys.exit("ERROR: gprofiler-official not installed. Install with:\n"
             "  pip install --user --break-system-packages gprofiler-official")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phantom_output_dir",
                    help="dir containing phantom_gene_lists/ from "
                         "simulate_practitioner_phantoms.py")
    ap.add_argument("--lens", default="lime")
    ap.add_argument("--organism", default="hsapiens")
    ap.add_argument("--significance_threshold_method",
                    default="g_SCS",
                    choices=["g_SCS", "bonferroni", "fdr"])
    ap.add_argument("--user_threshold", type=float, default=0.05)
    ap.add_argument("--sources", nargs="+",
                    default=["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"],
                    help="gprofiler annotation sources")
    args = ap.parse_args()

    gene_list_dir = os.path.join(args.phantom_output_dir, "phantom_gene_lists")
    if not os.path.isdir(gene_list_dir):
        sys.exit(f"ERROR: {gene_list_dir} not found. Run "
                 f"simulate_practitioner_phantoms.py first.")

    tsv_files = sorted(glob.glob(os.path.join(gene_list_dir,
                                                "practitioner_*_top50.tsv")))
    if not tsv_files:
        sys.exit(f"ERROR: no practitioner_*_top50.tsv in {gene_list_dir}")
    print(f"[enrich] found {len(tsv_files)} phantom gene lists to enrich")

    gp = GProfiler(return_dataframe=False)

    records = []
    for i, tsv in enumerate(tsv_files):
        df = pd.read_csv(tsv, sep="\t")
        query_genes = df["ensembl_id"].tolist()

        try:
            result = gp.profile(
                organism=args.organism,
                query=query_genes,
                sources=args.sources,
                user_threshold=args.user_threshold,
                significance_threshold_method=args.significance_threshold_method,
                no_evidences=True,
            )
        except Exception as e:
            sys.stderr.write(f"WARN: gprofiler failed for {tsv}: {e}\n")
            result = []

        # Filter to significant only and trim to a stable schema
        terms = []
        for t in result:
            if t.get("significant", False):
                terms.append({
                    "source": t.get("source"),
                    "native": t.get("native"),
                    "name": t.get("name"),
                    "p_value": t.get("p_value"),
                    "intersection_size": t.get("intersection_size"),
                    "term_size": t.get("term_size"),
                    "query_size": t.get("query_size"),
                })

        records.append({
            "practitioner_id": i,
            "tsv_file": os.path.basename(tsv),
            "lens": args.lens,
            "n_query_genes": len(query_genes),
            "n_significant_terms": len(terms),
            "top_terms": terms,
        })

        if (i + 1) % 10 == 0:
            print(f"[enrich] processed {i+1}/{len(tsv_files)}")

    # Write per-practitioner records (parallels per-basin records)
    per_path = os.path.join(args.phantom_output_dir,
                             "phantom_per_practitioner_enrichment_records.json")
    with open(per_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[enrich] wrote {per_path}")

    # Build recurrence summary (parallels per-basin recurrence)
    term_counts = {}      # key: (source, native) -> {"name": ..., "count": N}
    for rec in records:
        seen = set()
        for t in rec["top_terms"]:
            key = (t["source"], t["native"])
            if key in seen:
                continue
            seen.add(key)
            if key not in term_counts:
                term_counts[key] = {"source": t["source"],
                                     "native": t["native"],
                                     "name":   t["name"],
                                     "count":  0}
            term_counts[key]["count"] += 1

    phantom_term_recurrence = sorted(
        [{**v,
          "fraction_of_practitioners": v["count"] / len(records)}
         for v in term_counts.values()],
        key=lambda d: (-d["count"], d["name"]),
    )

    summary = {
        "lens":             args.lens,
        "n_practitioners":  len(records),
        "phantom_term_recurrence": phantom_term_recurrence,
    }
    summary_path = os.path.join(args.phantom_output_dir,
                                 "phantom_recurrence_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[enrich] wrote {summary_path}")

    # Console summary: top recurring phantom pathways
    print(f"\n[enrich] === TOP-15 RECURRING PHANTOM PATHWAYS ===")
    for t in phantom_term_recurrence[:15]:
        print(f"  {t['source']:<8}{t['name'][:55]:<58}"
              f"{t['count']}/{len(records)}")


if __name__ == "__main__":
    main()
