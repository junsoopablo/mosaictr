#!/usr/bin/env python3
"""Allele size delta analysis for HaploTR.

Stratifies genotyping accuracy by the magnitude of the allele size
difference from reference (|truth allele - ref|). This reveals how
well tools handle small vs large expansions/contractions.

Usage:
  python scripts/analyze_allele_delta.py \
    --haplotr output/genome_wide/v4_hg002_genome_wide.bed \
    --output output/genome_wide/allele_delta_report.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from haplotr.benchmark import load_predictions, _motif_period_bin
from haplotr.utils import load_adotto_catalog, load_tier1_bed, match_tier1_to_catalog
from scripts.benchmark_genome_wide import (
    match_preds_to_truth,
    match_tool_to_truth,
    parse_longtr_vcf,
    parse_trgt_vcf,
    compute_haplotr_metrics,
    compute_tool_metrics,
    TRUTH_BED,
    CATALOG_BED,
    LONGTR_VCF,
    TRGT_VCF,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Delta bins (absolute bp difference from reference per allele)
DELTA_BINS = [
    ("0bp", 0.0, 0.5),
    ("1-5bp", 0.5, 5.5),
    ("6-20bp", 5.5, 20.5),
    ("21-50bp", 20.5, 50.5),
    ("51-100bp", 50.5, 100.5),
    (">100bp", 100.5, float("inf")),
]


def _delta_bin(delta: float) -> str:
    """Assign a delta value to its bin."""
    ad = abs(delta)
    for name, lo, hi in DELTA_BINS:
        if lo <= ad < hi:
            return name
    return ">100bp"


def _max_allele_delta(truth) -> float:
    """Max absolute allele delta from ref for a truth locus."""
    return max(abs(truth.true_allele1_diff), abs(truth.true_allele2_diff))


def main():
    parser = argparse.ArgumentParser(description="Allele delta analysis")
    parser.add_argument("--haplotr", required=True, help="HaploTR v4 BED")
    parser.add_argument("--longtr-vcf", default=LONGTR_VCF)
    parser.add_argument("--trgt-vcf", default=TRGT_VCF)
    parser.add_argument("--truth", default=TRUTH_BED)
    parser.add_argument("--catalog", default=CATALOG_BED)
    parser.add_argument("--output", required=True, help="Output report")
    parser.add_argument("--chroms", default=None, help="Comma-separated chromosomes")
    args = parser.parse_args()

    chrom_set = set(args.chroms.split(",")) if args.chroms else None

    # Load data
    tier1 = load_tier1_bed(args.truth, chroms=chrom_set)
    catalog = load_adotto_catalog(args.catalog, chroms=chrom_set)

    ht_preds = load_predictions(args.haplotr)
    longtr = parse_longtr_vcf(args.longtr_vcf, chroms=chrom_set)
    trgt = parse_trgt_vcf(args.trgt_vcf, chroms=chrom_set)

    ht_pairs = match_preds_to_truth(ht_preds, tier1, catalog)
    lt_pairs = match_tool_to_truth(longtr, tier1, catalog)
    trgt_pairs = match_tool_to_truth(trgt, tier1, catalog)

    # Stratify by max allele delta
    lines = []
    w = lines.append

    w("=" * 90)
    w("  ALLELE SIZE DELTA ANALYSIS")
    w("  Accuracy stratified by |truth allele - reference| magnitude")
    w("=" * 90)

    # HaploTR
    ht_by_delta = defaultdict(list)
    for p, t in ht_pairs:
        ht_by_delta[_delta_bin(_max_allele_delta(t))].append((p, t))

    # LongTR
    lt_by_delta = defaultdict(list)
    for entry in lt_pairs:
        lt_by_delta[_delta_bin(_max_allele_delta(entry[2]))].append(entry)

    # TRGT
    trgt_by_delta = defaultdict(list)
    for entry in trgt_pairs:
        trgt_by_delta[_delta_bin(_max_allele_delta(entry[2]))].append(entry)

    header = f"  {'Delta':<12s} {'Tool':<15s} {'n':>6s} {'Exact':>7s} {'<=1bp':>7s} {'MAE':>7s} {'ZygAcc':>7s} {'GenConc':>7s}"
    w(f"\n{header}")
    w("  " + "-" * 65)

    for bin_name, _, _ in DELTA_BINS:
        ht_sub = ht_by_delta.get(bin_name, [])
        lt_sub = lt_by_delta.get(bin_name, [])
        trgt_sub = trgt_by_delta.get(bin_name, [])

        for tool_name, sub, compute_fn, is_ht in [
            ("HaploTR v4", ht_sub, compute_haplotr_metrics, True),
            ("LongTR", lt_sub, compute_tool_metrics, False),
            ("TRGT", trgt_sub, compute_tool_metrics, False),
        ]:
            m = compute_fn(sub)
            if m is None:
                w(f"  {bin_name:<12s} {tool_name:<15s} {'(no data)':>6s}")
                continue
            w(f"  {bin_name:<12s} {tool_name:<15s} {m['n']:>6d} "
              f"{m['exact']:>6.1%} {m['w1bp']:>6.1%} {m['mae']:>6.2f} "
              f"{m['zyg_acc']:>6.1%} {m['geno_conc']:>6.1%}")
        w("")

    # TP-only analysis
    w(f"\n{'='*90}")
    w("  TP (VARIANT) LOCI ONLY — BY ALLELE DELTA")
    w(f"{'='*90}")

    ht_tp_by_delta = defaultdict(list)
    for p, t in ht_pairs:
        if t.is_variant:
            ht_tp_by_delta[_delta_bin(_max_allele_delta(t))].append((p, t))

    lt_tp_by_delta = defaultdict(list)
    for entry in lt_pairs:
        if entry[2].is_variant:
            lt_tp_by_delta[_delta_bin(_max_allele_delta(entry[2]))].append(entry)

    trgt_tp_by_delta = defaultdict(list)
    for entry in trgt_pairs:
        if entry[2].is_variant:
            trgt_tp_by_delta[_delta_bin(_max_allele_delta(entry[2]))].append(entry)

    w(f"\n{header}")
    w("  " + "-" * 65)
    for bin_name, _, _ in DELTA_BINS:
        for tool_name, sub_dict, compute_fn in [
            ("HaploTR v4", ht_tp_by_delta, compute_haplotr_metrics),
            ("LongTR", lt_tp_by_delta, compute_tool_metrics),
            ("TRGT", trgt_tp_by_delta, compute_tool_metrics),
        ]:
            sub = sub_dict.get(bin_name, [])
            m = compute_fn(sub)
            if m is None:
                continue
            w(f"  {bin_name:<12s} {tool_name:<15s} {m['n']:>6d} "
              f"{m['exact']:>6.1%} {m['w1bp']:>6.1%} {m['mae']:>6.2f} "
              f"{m['zyg_acc']:>6.1%} {m['geno_conc']:>6.1%}")
        w("")

    report = "\n".join(lines)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", args.output)
    print(report)


if __name__ == "__main__":
    main()
