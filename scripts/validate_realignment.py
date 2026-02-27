#!/usr/bin/env python3
"""Validate re-alignment quality: compare raw CIGAR vs re-aligned allele sizes against GIAB truth.

For each locus:
  1. Extract per-read features WITHOUT re-alignment (raw minimap2 CIGAR)
  2. Extract per-read features WITH re-alignment (parasail, motif-aware gap penalties)
  3. Compute median allele size from reads for each method
  4. Compare against GIAB truth (hap1_diff + hap2_diff → true allele sizes)

Output: per-locus comparison table + stratified summary statistics.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pysam

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deeptr.features import extract_locus_reads
from deeptr.utils import (
    load_tier1_bed,
    load_adotto_catalog,
    match_tier1_to_catalog,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_realignment")


def median_allele_sizes(read_features: np.ndarray) -> tuple[float, float]:
    """Estimate diploid allele sizes from per-read allele_size_bp using simple median split.

    Sorts reads by allele_size, splits at median, returns (smaller_median, larger_median).
    For homozygous loci both values will be similar.
    """
    sizes = np.sort(read_features[:, 0])  # allele_size_bp is feature index 0
    n = len(sizes)
    if n == 1:
        return sizes[0], sizes[0]
    mid = n // 2
    a1 = np.median(sizes[:mid])
    a2 = np.median(sizes[mid:])
    return float(min(a1, a2)), float(max(a1, a2))


def per_allele_error(pred_a1, pred_a2, true_a1, true_a2):
    """Compute per-allele absolute error (sorted assignment)."""
    pred_sorted = sorted([pred_a1, pred_a2])
    true_sorted = sorted([true_a1, true_a2])
    return abs(pred_sorted[0] - true_sorted[0]), abs(pred_sorted[1] - true_sorted[1])


def main():
    parser = argparse.ArgumentParser(description="Validate re-alignment quality")
    parser.add_argument("--bam", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--truth-bed", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--chrom", default="chr21", help="Chromosome to validate on")
    parser.add_argument("--max-loci", type=int, default=2000, help="Max loci to test")
    parser.add_argument("--catalog-tolerance", type=int, default=10)
    parser.add_argument("--output", default=None, help="Output report path")
    args = parser.parse_args()

    chroms = {args.chrom}

    # Load truth data
    logger.info("Loading GIAB Tier1 truth for %s...", args.chrom)
    tier1_loci = load_tier1_bed(args.truth_bed, chroms=chroms)
    catalog = load_adotto_catalog(args.catalog, chroms=chroms)
    matched = match_tier1_to_catalog(tier1_loci, catalog, tolerance=args.catalog_tolerance)
    logger.info("Matched %d loci on %s", len(matched), args.chrom)

    # Subsample if needed
    if len(matched) > args.max_loci:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(matched), args.max_loci, replace=False)
        matched = [matched[i] for i in sorted(indices)]
        logger.info("Subsampled to %d loci", len(matched))

    # Open BAM and reference
    bam = pysam.AlignmentFile(args.bam, "rb", reference_filename=args.ref)
    ref_fasta = pysam.FastaFile(args.ref)

    # Per-locus comparison
    results = []
    t0 = time.time()

    for i, (locus, motif) in enumerate(matched):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d loci (%.1f loci/sec)", i + 1, len(matched), (i + 1) / elapsed)

        chrom, start, end = locus.chrom, locus.start, locus.end
        ref_size = end - start
        motif_len = len(motif)

        # True allele sizes (absolute, not diff)
        true_a1 = ref_size + locus.hap1_diff_bp
        true_a2 = ref_size + locus.hap2_diff_bp

        # --- Raw CIGAR features (no re-alignment) ---
        raw_reads, _ = extract_locus_reads(
            bam, chrom, start, end, motif,
            ref_fasta=None, realign=False,
        )

        # --- Re-aligned features ---
        realign_reads, _ = extract_locus_reads(
            bam, chrom, start, end, motif,
            ref_fasta=ref_fasta, realign=True,
        )

        if raw_reads is None or realign_reads is None:
            continue

        # Estimate allele sizes from reads
        raw_a1, raw_a2 = median_allele_sizes(raw_reads)
        realign_a1, realign_a2 = median_allele_sizes(realign_reads)

        # Per-allele errors
        raw_err = per_allele_error(raw_a1, raw_a2, true_a1, true_a2)
        realign_err = per_allele_error(realign_a1, realign_a2, true_a1, true_a2)

        results.append({
            "chrom": chrom,
            "start": start,
            "end": end,
            "motif": motif,
            "motif_len": motif_len,
            "ref_size": ref_size,
            "is_variant": locus.is_variant,
            "true_a1": true_a1,
            "true_a2": true_a2,
            "n_reads_raw": raw_reads.shape[0],
            "n_reads_realign": realign_reads.shape[0],
            "raw_a1": raw_a1,
            "raw_a2": raw_a2,
            "realign_a1": realign_a1,
            "realign_a2": realign_a2,
            "raw_err_a1": raw_err[0],
            "raw_err_a2": raw_err[1],
            "realign_err_a1": realign_err[0],
            "realign_err_a2": realign_err[1],
            "raw_mae": (raw_err[0] + raw_err[1]) / 2,
            "realign_mae": (realign_err[0] + realign_err[1]) / 2,
            # Per-read allele size distributions
            "raw_allele_sizes_std": float(np.std(raw_reads[:, 0])),
            "realign_allele_sizes_std": float(np.std(realign_reads[:, 0])),
            # All 7 CIGAR features: compare mean values
            "raw_mean_n_ins": float(np.mean(raw_reads[:, 1])),
            "realign_mean_n_ins": float(np.mean(realign_reads[:, 1])),
            "raw_mean_n_del": float(np.mean(raw_reads[:, 2])),
            "realign_mean_n_del": float(np.mean(realign_reads[:, 2])),
            "raw_mean_total_ins": float(np.mean(raw_reads[:, 3])),
            "realign_mean_total_ins": float(np.mean(realign_reads[:, 3])),
            "raw_mean_total_del": float(np.mean(raw_reads[:, 4])),
            "realign_mean_total_del": float(np.mean(realign_reads[:, 4])),
        })

    bam.close()
    ref_fasta.close()

    elapsed = time.time() - t0
    logger.info("Processed %d loci in %.1fs (%.1f loci/sec)", len(results), elapsed, len(results) / elapsed)

    # === Analysis ===
    if not results:
        logger.error("No results to analyze!")
        return

    lines = []

    def pr(s=""):
        lines.append(s)
        print(s)

    pr("=" * 70)
    pr(f"  Re-alignment Validation Report — {args.chrom}")
    pr(f"  {len(results)} loci, processed in {elapsed:.1f}s")
    pr("=" * 70)

    raw_maes = np.array([r["raw_mae"] for r in results])
    realign_maes = np.array([r["realign_mae"] for r in results])
    raw_stds = np.array([r["raw_allele_sizes_std"] for r in results])
    realign_stds = np.array([r["realign_allele_sizes_std"] for r in results])

    def report_section(name, mask):
        sub_raw = raw_maes[mask]
        sub_realign = realign_maes[mask]
        n = mask.sum()
        if n == 0:
            return

        pr(f"\n{'=' * 60}")
        pr(f"  {name}  (n={n:,})")
        pr(f"{'=' * 60}")

        # MAE
        pr(f"  Mean MAE   — raw: {sub_raw.mean():.2f} bp  |  realign: {sub_realign.mean():.2f} bp  |  diff: {sub_realign.mean() - sub_raw.mean():+.2f} bp")
        pr(f"  Median MAE — raw: {np.median(sub_raw):.2f} bp  |  realign: {np.median(sub_realign):.2f} bp  |  diff: {np.median(sub_realign) - np.median(sub_raw):+.2f} bp")

        # Per-allele accuracy thresholds
        for thresh_name, thresh in [("Exact (<0.5bp)", 0.5), ("Within 1bp", 1.0), ("Within 5bp", 5.0)]:
            raw_pct = (sub_raw < thresh).mean() * 100
            realign_pct = (sub_realign < thresh).mean() * 100
            pr(f"  {thresh_name:18s} — raw: {raw_pct:.1f}%  |  realign: {realign_pct:.1f}%  |  diff: {realign_pct - raw_pct:+.1f}%p")

        # How many loci improved vs degraded
        improved = ((sub_realign < sub_raw) & (sub_raw - sub_realign > 0.1)).sum()
        degraded = ((sub_realign > sub_raw) & (sub_realign - sub_raw > 0.1)).sum()
        unchanged = n - improved - degraded
        pr(f"  Improved: {improved} ({100*improved/n:.1f}%)  |  Degraded: {degraded} ({100*degraded/n:.1f}%)  |  Unchanged: {unchanged} ({100*unchanged/n:.1f}%)")

        # Read-level spread (lower = more consistent = better)
        pr(f"  Read allele_size std — raw: {raw_stds[mask].mean():.2f} bp  |  realign: {realign_stds[mask].mean():.2f} bp")

    # Overall
    all_mask = np.ones(len(results), dtype=bool)
    report_section("OVERALL", all_mask)

    # By variant status
    is_variant = np.array([r["is_variant"] for r in results])
    report_section("Variant (TP)", is_variant)
    report_section("Reference (TN)", ~is_variant)

    # By motif period
    motif_lens = np.array([r["motif_len"] for r in results])
    report_section("Homopolymer (motif=1)", motif_lens == 1)
    report_section("Dinucleotide (motif=2)", motif_lens == 2)
    report_section("STR (motif 3-6)", (motif_lens >= 3) & (motif_lens <= 6))
    report_section("VNTR (motif 7+)", motif_lens >= 7)

    # By repeat length
    ref_sizes = np.array([r["ref_size"] for r in results])
    report_section("Short (<100bp)", ref_sizes < 100)
    report_section("Medium (100-500bp)", (ref_sizes >= 100) & (ref_sizes < 500))
    report_section("Long (500-1000bp)", (ref_sizes >= 500) & (ref_sizes < 1000))
    report_section("Very long (>1000bp)", ref_sizes >= 1000)

    # Worst degraded loci (for debugging)
    pr(f"\n{'=' * 60}")
    pr(f"  TOP 20 DEGRADED LOCI (re-alignment made worse)")
    pr(f"{'=' * 60}")
    diffs = realign_maes - raw_maes
    worst_idx = np.argsort(diffs)[-20:][::-1]
    pr(f"  {'Locus':30s} {'Motif':8s} {'RefSize':>8s} {'TP':>3s} {'RawMAE':>8s} {'RealnMAE':>8s} {'Diff':>8s}")
    for idx in worst_idx:
        r = results[idx]
        if diffs[idx] <= 0.1:
            break
        loc = f"{r['chrom']}:{r['start']}-{r['end']}"
        pr(f"  {loc:30s} {r['motif'][:6]:8s} {r['ref_size']:>8d} {'Y' if r['is_variant'] else 'N':>3s} {r['raw_mae']:>8.1f} {r['realign_mae']:>8.1f} {diffs[idx]:>+8.1f}")

    # Top improved loci
    pr(f"\n{'=' * 60}")
    pr(f"  TOP 20 IMPROVED LOCI (re-alignment helped most)")
    pr(f"{'=' * 60}")
    best_idx = np.argsort(diffs)[:20]
    pr(f"  {'Locus':30s} {'Motif':8s} {'RefSize':>8s} {'TP':>3s} {'RawMAE':>8s} {'RealnMAE':>8s} {'Diff':>8s}")
    for idx in best_idx:
        r = results[idx]
        if diffs[idx] >= -0.1:
            break
        loc = f"{r['chrom']}:{r['start']}-{r['end']}"
        pr(f"  {loc:30s} {r['motif'][:6]:8s} {r['ref_size']:>8d} {'Y' if r['is_variant'] else 'N':>3s} {r['raw_mae']:>8.1f} {r['realign_mae']:>8.1f} {diffs[idx]:>+8.1f}")

    # CIGAR feature comparison
    pr(f"\n{'=' * 60}")
    pr(f"  CIGAR FEATURE COMPARISON (mean across all loci)")
    pr(f"{'=' * 60}")
    for feat_name, raw_key, realign_key in [
        ("n_insertions", "raw_mean_n_ins", "realign_mean_n_ins"),
        ("n_deletions", "raw_mean_n_del", "realign_mean_n_del"),
        ("total_ins_bp", "raw_mean_total_ins", "realign_mean_total_ins"),
        ("total_del_bp", "raw_mean_total_del", "realign_mean_total_del"),
    ]:
        raw_vals = np.array([r[raw_key] for r in results])
        realign_vals = np.array([r[realign_key] for r in results])
        pr(f"  {feat_name:18s} — raw: {raw_vals.mean():.2f}  |  realign: {realign_vals.mean():.2f}  |  diff: {realign_vals.mean() - raw_vals.mean():+.2f}")

    # Save report
    out_path = args.output or str(Path(__file__).parent.parent / "output" / "realignment_validation_report.txt")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
