#!/usr/bin/env python3
"""Confidence calibration analysis for MosaicTR.

Analyzes how MosaicTR's confidence score relates to actual accuracy:
1. Accuracy vs confidence threshold curve
2. Calibration: binned confidence vs observed accuracy
3. Filtering gains: accuracy after removing low-confidence calls

This is a MosaicTR-unique feature — LongTR and TRGT do not provide
per-locus confidence scores.

Usage:
  python scripts/analyze_confidence.py \
    --mosaictr output/genome_wide/v4_hg002_genome_wide.bed \
    --output output/genome_wide/confidence_report.txt \
    --output-tsv output/genome_wide/confidence_curve.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.benchmark import load_predictions, _motif_period_bin
from mosaictr.utils import load_adotto_catalog, load_tier1_bed
from scripts.benchmark_genome_wide import (
    match_preds_to_truth,
    compute_mosaictr_metrics,
    TRUTH_BED,
    CATALOG_BED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Confidence calibration analysis")
    parser.add_argument("--mosaictr", required=True, help="MosaicTR v4 BED")
    parser.add_argument("--truth", default=TRUTH_BED)
    parser.add_argument("--catalog", default=CATALOG_BED)
    parser.add_argument("--output", required=True, help="Output report")
    parser.add_argument("--output-tsv", default=None, help="Output TSV for plotting")
    args = parser.parse_args()

    tier1 = load_tier1_bed(args.truth)
    catalog = load_adotto_catalog(args.catalog)
    ht_preds = load_predictions(args.mosaictr)
    ht_pairs = match_preds_to_truth(ht_preds, tier1, catalog)

    logger.info("Matched %d loci for confidence analysis", len(ht_pairs))

    lines = []
    w = lines.append

    w("=" * 90)
    w("  CONFIDENCE CALIBRATION ANALYSIS")
    w("  MosaicTR v4 — GIAB Tier1 benchmark")
    w("=" * 90)
    w(f"\n  Total matched loci: {len(ht_pairs):,}")

    # ── 1. Accuracy at different confidence thresholds ────────────────
    w(f"\n{'='*90}")
    w("  ACCURACY VS CONFIDENCE THRESHOLD")
    w("  (Loci with confidence >= threshold)")
    w(f"{'='*90}")

    thresholds = np.arange(0.0, 1.01, 0.05)
    threshold_results = []

    w(f"\n  {'Thresh':>7s} {'n':>8s} {'%Kept':>7s} {'Exact':>7s} {'<=1bp':>7s} "
      f"{'MAE':>7s} {'ZygAcc':>7s} {'GenConc':>7s}")
    w("  " + "-" * 60)

    for thresh in thresholds:
        sub = [(p, t) for p, t in ht_pairs if p.confidence >= thresh]
        if len(sub) < 10:
            continue
        m = compute_mosaictr_metrics(sub)
        if m is None:
            continue
        pct_kept = len(sub) / len(ht_pairs) * 100
        w(f"  {thresh:>7.2f} {m['n']:>8,d} {pct_kept:>6.1f}% "
          f"{m['exact']:>6.1%} {m['w1bp']:>6.1%} "
          f"{m['mae']:>6.2f} {m['zyg_acc']:>6.1%} {m['geno_conc']:>6.1%}")
        threshold_results.append({
            "threshold": float(thresh),
            "n": m["n"],
            "pct_kept": pct_kept,
            "exact": m["exact"],
            "w1bp": m["w1bp"],
            "mae": m["mae"],
            "zyg_acc": m["zyg_acc"],
            "geno_conc": m["geno_conc"],
        })

    # ── 2. Calibration: binned confidence vs observed accuracy ─────────
    w(f"\n{'='*90}")
    w("  CALIBRATION: BINNED CONFIDENCE vs OBSERVED ACCURACY")
    w(f"{'='*90}")

    conf_bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    w(f"\n  {'Conf Range':>12s} {'n':>8s} {'Exact':>7s} {'<=1bp':>7s} "
      f"{'ZygAcc':>7s} {'GenConc':>7s}")
    w("  " + "-" * 50)

    for lo, hi in conf_bins:
        sub = [(p, t) for p, t in ht_pairs if lo <= p.confidence < hi]
        if len(sub) < 10:
            continue
        m = compute_mosaictr_metrics(sub)
        if m is None:
            continue
        w(f"  [{lo:.1f}-{hi:.1f}) {m['n']:>8,d} "
          f"{m['exact']:>6.1%} {m['w1bp']:>6.1%} "
          f"{m['zyg_acc']:>6.1%} {m['geno_conc']:>6.1%}")

    # ── 3. TP-only analysis ───────────────────────────────────────────
    w(f"\n{'='*90}")
    w("  TP (VARIANT) LOCI — CONFIDENCE THRESHOLD EFFECT")
    w(f"{'='*90}")

    tp_pairs = [(p, t) for p, t in ht_pairs if t.is_variant]
    w(f"\n  TP loci: {len(tp_pairs):,}")

    w(f"\n  {'Thresh':>7s} {'n':>8s} {'%Kept':>7s} {'Exact':>7s} {'<=1bp':>7s} "
      f"{'MAE':>7s} {'ZygAcc':>7s} {'GenConc':>7s}")
    w("  " + "-" * 60)

    for thresh in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        sub = [(p, t) for p, t in tp_pairs if p.confidence >= thresh]
        if len(sub) < 10:
            continue
        m = compute_mosaictr_metrics(sub)
        if m is None:
            continue
        pct_kept = len(sub) / len(tp_pairs) * 100
        w(f"  {thresh:>7.2f} {m['n']:>8,d} {pct_kept:>6.1f}% "
          f"{m['exact']:>6.1%} {m['w1bp']:>6.1%} "
          f"{m['mae']:>6.2f} {m['zyg_acc']:>6.1%} {m['geno_conc']:>6.1%}")

    # ── 4. Zygosity-specific confidence ───────────────────────────────
    w(f"\n{'='*90}")
    w("  ZYGOSITY-SPECIFIC CONFIDENCE DISTRIBUTION")
    w(f"{'='*90}")

    for zyg_label, zyg_val in [("HOM", "HOM"), ("HET", "HET")]:
        sub = [(p, t) for p, t in ht_pairs if p.zygosity == zyg_val]
        if not sub:
            continue
        confs = [p.confidence for p, _ in sub]
        w(f"\n  {zyg_label} calls (n={len(sub):,d}):")
        w(f"    Mean confidence: {np.mean(confs):.3f}")
        w(f"    Median confidence: {np.median(confs):.3f}")
        w(f"    <0.5 confidence: {sum(1 for c in confs if c < 0.5):,d} ({sum(1 for c in confs if c < 0.5)/len(confs):.1%})")

        # Correct vs incorrect zygosity
        correct = [(p, t) for p, t in sub
                    if (p.zygosity == "HET") == (abs(t.true_allele1_diff - t.true_allele2_diff) > t.motif_length)]
        incorrect = [(p, t) for p, t in sub
                     if (p.zygosity == "HET") != (abs(t.true_allele1_diff - t.true_allele2_diff) > t.motif_length)]
        if correct:
            w(f"    Correct zygosity: n={len(correct):,d}, mean_conf={np.mean([p.confidence for p, _ in correct]):.3f}")
        if incorrect:
            w(f"    Incorrect zygosity: n={len(incorrect):,d}, mean_conf={np.mean([p.confidence for p, _ in incorrect]):.3f}")

    report = "\n".join(lines)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", args.output)

    # Write TSV for plotting
    if args.output_tsv and threshold_results:
        tsv_path = Path(args.output_tsv)
        with open(tsv_path, "w") as f:
            f.write("threshold\tn\tpct_kept\texact\tw1bp\tmae\tzyg_acc\tgeno_conc\n")
            for r in threshold_results:
                f.write(f"{r['threshold']:.2f}\t{r['n']}\t{r['pct_kept']:.1f}\t"
                        f"{r['exact']:.4f}\t{r['w1bp']:.4f}\t{r['mae']:.4f}\t"
                        f"{r['zyg_acc']:.4f}\t{r['geno_conc']:.4f}\n")
        logger.info("TSV saved to %s", tsv_path)

    print(report)


if __name__ == "__main__":
    main()
