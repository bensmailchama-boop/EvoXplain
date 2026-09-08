#!/usr/bin/env python3
"""
lewis_enrichment.py

Enrichment of the Lewis & Kemp audit gene lists: each refit's own top-N genes
enriched separately, plus the averaged "consensus" panel.

Uses the measured genes as an explicit background (domain_scope='custom').
This matters: against g:Profiler's default genome-wide domain, a 50-gene query
drawn from a 22,835-gene assay returns terms that reflect the assay's gene
coverage rather than the model. With the correct background most models return
no significant terms at all.

Answers two questions:
  1. do the refits enrich to the same biology, or different?
  2. does the averaged panel contain terms no individual model supports?

Outputs:
  <out_prefix>_enrichment.json   per-model and panel term records

Usage:
  python lewis_enrichment.py [--in_prefix lewis] [--out_prefix lewis]

  Reads <in_prefix>_genelists.json and <in_prefix>_background.json,
  as written by lewis_build_genelists.py.
"""

import argparse
import json
import sys
from collections import Counter

try:
    from gprofiler import GProfiler
except ImportError:
    sys.exit("ERROR: gprofiler-official not installed.\n"
             "  pip install gprofiler-official --break-system-packages")


SOURCES = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]


def trim(t):
    return {"source": t["source"], "native": t["native"], "name": t["name"],
            "p_value": t["p_value"],
            "intersection_size": t.get("intersection_size"),
            "term_size": t.get("term_size"),
            "intersections": t.get("intersections")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_prefix", default="lewis",
                    help="prefix of the genelists/background json (default: lewis)")
    ap.add_argument("--out_prefix", default=None,
                    help="prefix for output json (defaults to --in_prefix)")
    ap.add_argument("--organism", default="hsapiens")
    ap.add_argument("--user_threshold", type=float, default=0.05)
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--no_background", action="store_true",
                    help="use g:Profiler's default domain instead of the "
                         "measured genes (NOT recommended; see docstring)")
    ap.add_argument("--top_report", type=int, default=25,
                    help="how many recurrent terms to print (default 25)")
    args = ap.parse_args()

    out_prefix = args.out_prefix or args.in_prefix

    d = json.load(open("%s_genelists.json" % args.in_prefix))
    bg = json.load(open("%s_background.json" % args.in_prefix))
    n_models = len(d["models"])
    print("models: %d | background genes: %d | top-N: %s"
          % (n_models, len(bg), d.get("top_n", "?")))
    if args.no_background:
        print("WARNING: running without an explicit background")

    gp = GProfiler(return_dataframe=False)

    def enrich(genes, label):
        kw = dict(organism=args.organism, query=list(genes),
                  sources=args.sources, user_threshold=args.user_threshold,
                  no_evidences=False,
                  significance_threshold_method="g_SCS")
        if not args.no_background:
            kw.update(background=bg, domain_scope="custom")
        r = gp.profile(**kw)
        sig = [t for t in r if t.get("significant", False)]
        print("%-14s %d significant terms" % (label, len(sig)))
        return sig

    out = {}
    counter = Counter()
    names = {}

    for i, m in enumerate(d["models"]):
        sig = enrich(m, "model %d" % (i + 1))
        out["model_%d" % (i + 1)] = [trim(t) for t in sig]
        for t in sig:
            counter[t["native"]] += 1
            names[t["native"]] = (t["source"], t["name"])

    sig = enrich(d["panel"], "PANEL(avg)")
    out["panel"] = [trim(t) for t in sig]
    for t in sig:
        names.setdefault(t["native"], (t["source"], t["name"]))

    out_path = "%s_enrichment.json" % out_prefix
    json.dump(out, open(out_path, "w"), indent=1)

    # ---- recurrence across models -----------------------------------
    print("")
    print("=== term recurrence across %d models ===" % n_models)
    if not counter:
        print("  no significant terms in any model")
    for nat, c in counter.most_common(args.top_report):
        s, n = names[nat]
        print("  %2d/%d  %-6s %s" % (c, n_models, s, n[:60]))

    zero = sum(1 for i in range(n_models) if not out["model_%d" % (i + 1)])
    print("")
    print("models with zero significant terms: %d/%d" % (zero, n_models))

    # ---- does the panel contain anything no model supports? ---------
    panel_terms = {t["native"] for t in out["panel"]}
    model_terms = set(counter)
    orphans = panel_terms - model_terms
    print("panel terms: %d | union of model terms: %d"
          % (len(panel_terms), len(model_terms)))
    print("panel terms NOT found in any individual model: %d" % len(orphans))
    for nat in sorted(orphans):
        s, n = names.get(nat, ("?", nat))
        print("   %-6s %s" % (s, n))

    print("")
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
