#!/usr/bin/env python3
"""Mendelian inheritance analysis for MosaicTR.

Checks whether child (HG002) genotypes are consistent with parental
genotypes (HG003=father, HG004=mother) at shared loci.

A child's two alleles should be: one from father + one from mother.
For a diploid locus with father alleles (f1,f2) and mother (m1,m2),
the child should have one allele matching f1 or f2 AND one matching m1 or m2.

Reports strict, ±1bp, and ±1 motif unit concordance, stratified by
motif period and zygosity status.

Usage:
  python scripts/analyze_mendelian.py \
    --child output/genome_wide/v4_hg002_genome_wide.bed \
    --father output/genome_wide/v4_hg003_genome_wide.bed \
    --mother output/genome_wide/v4_hg004_genome_wide.bed \
    --catalog /vault/.../adotto_v1.2_longtr_v1.2_format.bed \
    --output output/genome_wide/mendelian_report.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.benchmark import LocusPrediction, load_predictions, _motif_period_bin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Mendelian consistency check
# ---------------------------------------------------------------------------

def _alleles_from_pred(pred: LocusPrediction) -> tuple[float, float]:
    """Extract allele diffs (allele - ref) from prediction."""
    ref_size = pred.end - pred.start
    d1 = pred.allele1_size - ref_size
    d2 = pred.allele2_size - ref_size
    return sorted([d1, d2])


def is_mendelian_consistent(
    child_alleles: tuple[float, float],
    father_alleles: tuple[float, float],
    mother_alleles: tuple[float, float],
    tolerance: float = 0.0,
) -> bool:
    """Check Mendelian consistency: child should inherit one allele from each parent.

    Child genotype (c1, c2) is Mendelian-consistent if there exists an assignment
    where one child allele came from father and one from mother.

    Two possible assignments:
      1. c1 from father, c2 from mother
      2. c1 from mother, c2 from father

    Each "from parent" means the child allele matches one of the parent's two alleles
    within the given tolerance.
    """
    c1, c2 = child_alleles
    f1, f2 = father_alleles
    m1, m2 = mother_alleles

    def _matches(child_a, parent_alleles, tol):
        return any(abs(child_a - pa) <= tol for pa in parent_alleles)

    # Assignment 1: c1 from father, c2 from mother
    if _matches(c1, (f1, f2), tolerance) and _matches(c2, (m1, m2), tolerance):
        return True
    # Assignment 2: c1 from mother, c2 from father
    if _matches(c1, (m1, m2), tolerance) and _matches(c2, (f1, f2), tolerance):
        return True
    return False


# ---------------------------------------------------------------------------
# Build lookup from predictions
# ---------------------------------------------------------------------------

def build_locus_lookup(
    preds: list[LocusPrediction],
) -> dict[tuple[str, int], LocusPrediction]:
    """Build (chrom, start) -> LocusPrediction lookup."""
    lookup = {}
    for p in preds:
        if p.allele1_size == 0 and p.allele2_size == 0:
            continue
        key = (p.chrom, p.start)
        lookup[key] = p
    return lookup


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_mendelian(
    child_preds: list[LocusPrediction],
    father_preds: list[LocusPrediction],
    mother_preds: list[LocusPrediction],
) -> dict:
    """Run Mendelian inheritance analysis on trio data.

    Returns dict with counts and stratified results.
    """
    child_lk = build_locus_lookup(child_preds)
    father_lk = build_locus_lookup(father_preds)
    mother_lk = build_locus_lookup(mother_preds)

    # Find common loci (present in all 3 samples)
    common_keys = set(child_lk.keys()) & set(father_lk.keys()) & set(mother_lk.keys())
    logger.info("Common loci across trio: %d", len(common_keys))
    logger.info("  Child: %d, Father: %d, Mother: %d",
                len(child_lk), len(father_lk), len(mother_lk))

    # Tolerance levels
    tolerances = {
        "strict": 0.0,
        "within_1bp": 1.0,
    }

    # Per-locus results
    results_per_locus = []
    for key in sorted(common_keys):
        cp = child_lk[key]
        fp = father_lk[key]
        mp = mother_lk[key]

        c_alleles = _alleles_from_pred(cp)
        f_alleles = _alleles_from_pred(fp)
        m_alleles = _alleles_from_pred(mp)
        motif_len = len(cp.motif)
        motif_bin = _motif_period_bin(motif_len)

        entry = {
            "chrom": cp.chrom,
            "start": cp.start,
            "motif": cp.motif,
            "motif_len": motif_len,
            "motif_bin": motif_bin,
            "child_alleles": c_alleles,
            "father_alleles": f_alleles,
            "mother_alleles": m_alleles,
            "child_zyg": cp.zygosity,
            "father_zyg": fp.zygosity,
            "mother_zyg": mp.zygosity,
        }

        for name, tol in tolerances.items():
            entry[f"mi_{name}"] = is_mendelian_consistent(c_alleles, f_alleles, m_alleles, tol)

        # ±1 motif unit
        entry["mi_within_1motif"] = is_mendelian_consistent(
            c_alleles, f_alleles, m_alleles, float(motif_len),
        )

        results_per_locus.append(entry)

    # Aggregate
    n_total = len(results_per_locus)
    summary = {"n_total": n_total}

    for metric in ["mi_strict", "mi_within_1bp", "mi_within_1motif"]:
        n_pass = sum(1 for r in results_per_locus if r[metric])
        summary[metric] = n_pass
        summary[f"{metric}_rate"] = n_pass / n_total if n_total > 0 else 0.0

    # By motif period
    by_motif = defaultdict(list)
    for r in results_per_locus:
        by_motif[r["motif_bin"]].append(r)

    summary["by_motif"] = {}
    for mb, entries in sorted(by_motif.items()):
        n = len(entries)
        sub = {}
        for metric in ["mi_strict", "mi_within_1bp", "mi_within_1motif"]:
            n_pass = sum(1 for e in entries if e[metric])
            sub[metric] = n_pass
            sub[f"{metric}_rate"] = n_pass / n if n > 0 else 0.0
        sub["n"] = n
        summary["by_motif"][mb] = sub

    # By child zygosity
    by_zyg = defaultdict(list)
    for r in results_per_locus:
        by_zyg[r["child_zyg"]].append(r)

    summary["by_zyg"] = {}
    for zyg, entries in sorted(by_zyg.items()):
        n = len(entries)
        sub = {}
        for metric in ["mi_strict", "mi_within_1bp", "mi_within_1motif"]:
            n_pass = sum(1 for e in entries if e[metric])
            sub[metric] = n_pass
            sub[f"{metric}_rate"] = n_pass / n if n > 0 else 0.0
        sub["n"] = n
        summary["by_zyg"][zyg] = sub

    # Violation examples (strict)
    violations = [r for r in results_per_locus if not r["mi_strict"]]
    summary["n_violations_strict"] = len(violations)
    summary["violation_examples"] = violations[:20]

    return summary


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(summary: dict) -> str:
    lines = []
    w = lines.append

    w("=" * 80)
    w("  MENDELIAN INHERITANCE ANALYSIS")
    w("  MosaicTR v4 — HG002 (child) / HG003 (father) / HG004 (mother)")
    w("=" * 80)

    w(f"\n  Common loci across trio: {summary['n_total']:,}")

    w(f"\n  {'Metric':<25s} {'Pass':>8s} {'Total':>8s} {'Rate':>8s}")
    w("  " + "-" * 50)
    for metric, label in [
        ("mi_strict", "Strict (exact)"),
        ("mi_within_1bp", "Within 1bp"),
        ("mi_within_1motif", "Within 1 motif unit"),
    ]:
        n = summary[metric]
        rate = summary[f"{metric}_rate"]
        w(f"  {label:<25s} {n:>8,d} {summary['n_total']:>8,d} {rate:>7.2%}")

    # Published comparisons
    w(f"\n{'='*80}")
    w("  COMPARISON WITH PUBLISHED TOOLS")
    w(f"{'='*80}")
    w(f"\n  {'Tool':<20s} {'Metric':<30s} {'Rate':>8s}")
    w("  " + "-" * 60)
    w(f"  {'LongTR':<20s} {'Strict Mendelian':<30s} {'86.0%':>8s}")
    w(f"  {'TRGT':<20s} {'Off-by-one-unit Mendelian':<30s} {'98.38%':>8s}")
    w(f"  {'STRkit':<20s} {'Strict Mendelian':<30s} {'97.85%':>8s}")
    w(f"  {'STRkit':<20s} {'Within 1bp Mendelian':<30s} {'99.08%':>8s}")
    w(f"  {'MosaicTR v4':<20s} {'Strict Mendelian':<30s} {summary['mi_strict_rate']:>7.2%}")
    w(f"  {'MosaicTR v4':<20s} {'Within 1bp Mendelian':<30s} {summary['mi_within_1bp_rate']:>7.2%}")
    w(f"  {'MosaicTR v4':<20s} {'Within 1 motif unit':<30s} {summary['mi_within_1motif_rate']:>7.2%}")

    # By motif period
    w(f"\n{'='*80}")
    w("  BY MOTIF PERIOD")
    w(f"{'='*80}")
    motif_order = ["homopolymer", "dinucleotide", "STR_3bp", "STR_4bp",
                   "STR_5bp", "STR_6bp", "VNTR_7+"]
    w(f"\n  {'Motif':<15s} {'n':>8s} {'Strict':>8s} {'±1bp':>8s} {'±1motif':>8s}")
    w("  " + "-" * 50)
    for mb in motif_order:
        if mb not in summary["by_motif"]:
            continue
        sub = summary["by_motif"][mb]
        w(f"  {mb:<15s} {sub['n']:>8,d} "
          f"{sub['mi_strict_rate']:>7.2%} "
          f"{sub['mi_within_1bp_rate']:>7.2%} "
          f"{sub['mi_within_1motif_rate']:>7.2%}")

    # By zygosity
    w(f"\n{'='*80}")
    w("  BY CHILD ZYGOSITY")
    w(f"{'='*80}")
    w(f"\n  {'Zygosity':<10s} {'n':>8s} {'Strict':>8s} {'±1bp':>8s} {'±1motif':>8s}")
    w("  " + "-" * 45)
    for zyg in ["HOM", "HET"]:
        if zyg not in summary["by_zyg"]:
            continue
        sub = summary["by_zyg"][zyg]
        w(f"  {zyg:<10s} {sub['n']:>8,d} "
          f"{sub['mi_strict_rate']:>7.2%} "
          f"{sub['mi_within_1bp_rate']:>7.2%} "
          f"{sub['mi_within_1motif_rate']:>7.2%}")

    # Violation examples
    w(f"\n{'='*80}")
    w(f"  STRICT MENDELIAN VIOLATIONS — TOP 20 EXAMPLES")
    w(f"{'='*80}")
    w(f"\n  Violations: {summary['n_violations_strict']:,} / {summary['n_total']:,} "
      f"({summary['n_violations_strict']/summary['n_total']:.2%})")
    w(f"\n  {'Chrom':<8s} {'Start':>12s} {'Motif':<8s} {'Child':>15s} {'Father':>15s} {'Mother':>15s}")
    w("  " + "-" * 70)
    for v in summary.get("violation_examples", [])[:20]:
        ca = v["child_alleles"]
        fa = v["father_alleles"]
        ma = v["mother_alleles"]
        w(f"  {v['chrom']:<8s} {v['start']:>12,d} {v['motif']:<8s} "
          f"({ca[0]:>5.0f},{ca[1]:>5.0f}) "
          f"({fa[0]:>5.0f},{fa[1]:>5.0f}) "
          f"({ma[0]:>5.0f},{ma[1]:>5.0f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mendelian inheritance analysis for MosaicTR trio")
    parser.add_argument("--child", required=True, help="HG002 (child) MosaicTR BED")
    parser.add_argument("--father", required=True, help="HG003 (father) MosaicTR BED")
    parser.add_argument("--mother", required=True, help="HG004 (mother) MosaicTR BED")
    parser.add_argument("--output", required=True, help="Output report file")
    args = parser.parse_args()

    logger.info("Loading child (HG002): %s", args.child)
    child = load_predictions(args.child)
    logger.info("Loading father (HG003): %s", args.father)
    father = load_predictions(args.father)
    logger.info("Loading mother (HG004): %s", args.mother)
    mother = load_predictions(args.mother)

    logger.info("Child: %d, Father: %d, Mother: %d", len(child), len(father), len(mother))

    summary = analyze_mendelian(child, father, mother)
    report = format_report(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    logger.info("Report saved to %s", output_path)
    print(report)


if __name__ == "__main__":
    main()
