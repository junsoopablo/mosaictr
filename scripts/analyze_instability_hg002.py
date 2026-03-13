#!/usr/bin/env python3
"""HG002 instability validation: genome-wide analysis, hp-vs-pooled, trio.

Subcommands:
  genomewide     Analyze genome-wide instability TSV and produce statistics
  hp-vs-pooled   Per-locus hp-tagged vs forced-pooled instability comparison
  trio           Compare instability consistency across HG002/HG003/HG004

Usage examples:
  # Task 2: Analyze genome-wide instability TSV
  python scripts/analyze_instability_hg002.py genomewide \
    --tsv output/instability/hg002_genomewide.tsv \
    --output output/instability/genomewide_report.txt

  # Task 1: HP-tagged vs pooled comparison
  python scripts/analyze_instability_hg002.py hp-vs-pooled \
    --bam /vault/.../HG002.bam \
    --loci output/v3_comparison/test_loci.bed \
    --output output/instability/hp_vs_pooled_report.txt \
    --max-loci 10000

  # Task 3: Trio instability consistency
  python scripts/analyze_instability_hg002.py trio \
    --hg002 output/instability/hg002_genomewide.tsv \
    --hg003 output/instability/hg003_genomewide.tsv \
    --hg004 output/instability/hg004_genomewide.tsv \
    --output output/instability/trio_report.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.genotype import ReadInfo, extract_reads_enhanced
from mosaictr.instability import (
    _instability_pooled_fallback,
    compute_instability,
)
from mosaictr.utils import load_loci_bed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try matplotlib (headless)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

BAM_HG002 = (
    "/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio-48x_BAM_GRCh38/"
    "HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.bam"
)
BAM_HG003 = (
    "/vault/external-datasets/2026/HG003_PacBio-HiFi-Revio_BAM_GRCh38/"
    "GRCh38.m84039_241002_000337_s3.hifi_reads.bc2020.bam"
)
BAM_HG004 = (
    "/vault/external-datasets/2026/HG004_PacBio-HiFi-Revio_BAM_GRCh38/"
    "GRCh38.m84039_241002_020632_s4.hifi_reads.bc2021.bam"
)
TEST_LOCI = "output/v3_comparison/test_loci.bed"


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

def load_instability_tsv(path: str) -> list[dict]:
    """Load instability TSV into list of dicts."""
    rows = []
    with open(path) as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                header = line.lstrip("#").split("\t")
                continue
            if not header:
                continue
            fields = line.split("\t")
            row = {}
            for i, col in enumerate(header):
                row[col] = fields[i] if i < len(fields) else "."
            rows.append(row)
    return rows


def parse_float(s: str, default: float = 0.0) -> float:
    """Parse float from TSV field, handling '.' as missing."""
    if s == "." or s == "":
        return default
    return float(s)


# ===========================================================================
# Subcommand: genomewide
# ===========================================================================

def cmd_genomewide(args):
    """Analyze genome-wide instability TSV output."""
    rows = load_instability_tsv(args.tsv)
    n_total = len(rows)
    logger.info("Loaded %d rows from %s", n_total, args.tsv)

    # Parse results
    hiis = []
    ais_vals = []
    ser_vals = []
    paths = {"hp-tagged": 0, "gap-split": 0, "pooled": 0, "failed": 0}

    for row in rows:
        path = row.get("analysis_path", "failed")
        paths[path] = paths.get(path, 0) + 1

        if path == "failed":
            continue

        hii_h1 = parse_float(row.get("hii_h1", "."))
        hii_h2 = parse_float(row.get("hii_h2", "."))
        hiis.append(max(hii_h1, hii_h2))

        ais_vals.append(parse_float(row.get("ais", ".")))
        ser_h1 = parse_float(row.get("ser_h1", "."))
        ser_h2 = parse_float(row.get("ser_h2", "."))
        ser_vals.append(max(ser_h1, ser_h2))

    hiis = np.array(hiis)
    ais_vals = np.array(ais_vals)
    ser_vals = np.array(ser_vals)
    n_analyzed = len(hiis)
    n_failed = paths["failed"]

    # Build report
    lines = []
    w = lines.append

    w("=" * 80)
    w("  HG002 GENOME-WIDE INSTABILITY ANALYSIS")
    w("=" * 80)

    w(f"\n  OVERVIEW")
    w(f"  Total loci:     {n_total:,}")
    w(f"  Analyzed:       {n_analyzed:,} ({n_analyzed/n_total*100:.1f}%)")
    w(f"  Failed:         {n_failed:,} ({n_failed/n_total*100:.1f}%)")

    w(f"\n  ANALYSIS PATH")
    for pname in ["hp-tagged", "gap-split", "pooled", "failed"]:
        n = paths[pname]
        pct = n / n_total * 100 if n_total > 0 else 0
        w(f"  {pname:<12s}  {n:>8,} ({pct:>5.1f}%)")

    # HII distribution
    w(f"\n  HII DISTRIBUTION (max(HII_h1, HII_h2))")
    w(f"  Mean:           {np.mean(hiis):.4f}")
    w(f"  Std:            {np.std(hiis):.4f}")
    w(f"  Median:         {np.median(hiis):.4f}")
    w(f"  P95:            {np.percentile(hiis, 95):.4f}")
    w(f"  P99:            {np.percentile(hiis, 99):.4f}")
    w(f"  Max:            {np.max(hiis):.4f}")

    # Threshold counts
    w("")
    n_zero = int(np.sum(hiis == 0))
    w(f"  HII = 0 (zero instability)   {n_zero:>8,} ({n_zero/n_analyzed*100:.1f}%)")
    for thr, label in [(0.01, "< 0.01 negligible"),
                        (0.1, "< 0.1  minimal"),
                        (0.45, "< 0.45 below 3-sigma")]:
        n = int(np.sum(hiis < thr))
        w(f"  HII {label:<25s} {n:>8,} ({n/n_analyzed*100:.1f}%)")
    n_outlier = int(np.sum(hiis >= 0.45))
    n_sig = int(np.sum(hiis >= 1.0))
    w(f"  HII >= 0.45 outlier          {n_outlier:>8,} ({n_outlier/n_analyzed*100:.2f}%)")
    w(f"  HII >= 1.0  significant      {n_sig:>8,} ({n_sig/n_analyzed*100:.2f}%)")

    # AIS distribution
    w(f"\n  AIS DISTRIBUTION")
    w(f"  Mean:           {np.mean(ais_vals):.4f}")
    w(f"  Std:            {np.std(ais_vals):.4f}")
    w(f"  Median:         {np.median(ais_vals):.4f}")
    w(f"  P99:            {np.percentile(ais_vals, 99):.4f}")
    w(f"  Max:            {np.max(ais_vals):.4f}")

    # SER distribution
    w(f"\n  SER DISTRIBUTION (max(SER_h1, SER_h2))")
    w(f"  Mean:           {np.mean(ser_vals):.4f}")
    n_nonzero_ser = int(np.sum(ser_vals > 0))
    w(f"  Non-zero:       {n_nonzero_ser:,} ({n_nonzero_ser/n_analyzed*100:.1f}%)")

    # Noise floor summary
    mean_hii = float(np.mean(hiis))
    std_hii = float(np.std(hiis))
    threshold_3sigma = mean_hii + 3 * std_hii

    w(f"\n  NOISE FLOOR SUMMARY")
    w(f"  Mean HII:       {mean_hii:.4f}")
    w(f"  Std HII:        {std_hii:.4f}")
    w(f"  3-sigma:        {threshold_3sigma:.4f}")
    w(f"  Loci > 3-sigma: {int(np.sum(hiis > threshold_3sigma)):,}")

    # Outlier loci details (up to 50)
    if 0 < n_outlier <= 50:
        w(f"\n  OUTLIER LOCI (HII >= 0.45)")
        w(f"  {'chrom':<8s} {'start':>12s} {'end':>12s} {'motif':<12s} "
          f"{'HII_h1':>8s} {'HII_h2':>8s} {'AIS':>8s} {'path':<10s}")
        w("  " + "-" * 80)
        for row in rows:
            if row.get("analysis_path") == "failed":
                continue
            h1 = parse_float(row.get("hii_h1", "."))
            h2 = parse_float(row.get("hii_h2", "."))
            if max(h1, h2) >= 0.45:
                w(f"  {row['chrom']:<8s} {row['start']:>12s} {row['end']:>12s} "
                  f"{row['motif'][:12]:<12s} {h1:>8.4f} {h2:>8.4f} "
                  f"{parse_float(row.get('ais', '.')):>8.4f} "
                  f"{row['analysis_path']:<10s}")
    elif n_outlier > 50:
        w(f"\n  OUTLIER LOCI: {n_outlier} loci (too many to list; see TSV)")

    w(f"\n{'=' * 80}")

    report = "\n".join(lines)

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(report)
    logger.info("Report saved to %s", output_path)

    # Figures
    if HAS_MPL:
        fig_dir = output_path.parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        # HII distribution histogram
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(hiis, bins=100, edgecolor="none", alpha=0.7, color="#4C72B0")
        axes[0].axvline(0.45, color="red", ls="--", lw=1, label="3-sigma threshold")
        axes[0].set_xlabel("max(HII_h1, HII_h2)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"HII Distribution (n={n_analyzed:,})")
        axes[0].legend()

        hiis_pos = hiis[hiis > 0]
        if len(hiis_pos) > 0:
            axes[1].hist(np.log10(hiis_pos + 1e-6), bins=100,
                         edgecolor="none", alpha=0.7, color="#4C72B0")
            axes[1].axvline(np.log10(0.45), color="red", ls="--", lw=1,
                            label="3-sigma threshold")
            axes[1].set_xlabel("log10(max HII)")
            axes[1].set_ylabel("Count")
            axes[1].set_title("HII Distribution (log scale, non-zero only)")
            axes[1].legend()

        plt.tight_layout()
        fig_path = fig_dir / "hii_distribution.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info("Figure saved: %s", fig_path)

        # Analysis path pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        labels = [k for k in ["hp-tagged", "gap-split", "pooled", "failed"]
                  if paths[k] > 0]
        sizes = [paths[k] for k in labels]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(labels)]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
               colors=colors)
        ax.set_title("Analysis Path Distribution")
        fig_path = fig_dir / "analysis_path.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info("Figure saved: %s", fig_path)


# ===========================================================================
# Subcommand: hp-vs-pooled
# ===========================================================================

def cmd_hp_vs_pooled(args):
    """Compare hp-tagged vs forced-pooled instability.

    For each locus:
    - compute_instability() with real HP tags → hp-tagged HII
    - _instability_pooled_fallback() ignoring HP → pooled HII
    HET loci: pooled HII inflated (alleles mixed); hp-tagged HII ≈ 0 (correct).
    HOM loci: both similar.
    """
    import pysam

    loci = load_loci_bed(args.loci)
    logger.info("Loaded %d loci from %s", len(loci), args.loci)

    # Subsample
    if args.max_loci and 0 < args.max_loci < len(loci):
        np.random.seed(42)
        indices = sorted(np.random.choice(len(loci), args.max_loci, replace=False))
        loci = [loci[i] for i in indices]
        logger.info("Subsampled to %d loci", len(loci))

    bam = pysam.AlignmentFile(args.bam, "rb")

    comparisons: list[dict] = []
    n_skip = 0
    t0 = time.time()

    for chrom, start, end, motif in tqdm(loci, desc="HP vs Pooled"):
        ref_size = end - start
        motif_len = len(motif)

        reads = extract_reads_enhanced(
            bam, chrom, start, end,
            min_mapq=5, min_flank=50, max_reads=200,
            motif_len=motif_len,
        )
        if not reads or len(reads) < 3:
            n_skip += 1
            continue

        # HP-tagged result (full pipeline with haplotype separation)
        hp_result = compute_instability(reads, ref_size, motif_len)
        if hp_result is None:
            n_skip += 1
            continue

        # Forced pooled result (all reads treated as single pool)
        pooled_result = _instability_pooled_fallback(reads, ref_size, motif_len)

        # Classify zygosity from hp-tagged modal alleles
        allele_diff = abs(hp_result["modal_h1"] - hp_result["modal_h2"])
        is_het = allele_diff > motif_len

        hp_max_hii = max(hp_result["hii_h1"], hp_result["hii_h2"])
        pooled_hii = pooled_result["hii_h1"]  # pooled puts everything in h1

        comparisons.append({
            "chrom": chrom,
            "start": start,
            "end": end,
            "motif": motif,
            "motif_len": motif_len,
            "is_het": is_het,
            "allele_diff": allele_diff,
            "hp_hii_h1": hp_result["hii_h1"],
            "hp_hii_h2": hp_result["hii_h2"],
            "hp_max_hii": hp_max_hii,
            "pooled_hii": pooled_hii,
            "hp_path": hp_result["analysis_path"],
            "n_total": hp_result["n_total"],
        })

    bam.close()
    elapsed = time.time() - t0
    logger.info("Processed %d loci in %.1fs, skipped %d",
                len(comparisons), elapsed, n_skip)

    # Save comparison TSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tsv_path = output_path.with_suffix(".tsv")
    with open(tsv_path, "w") as f:
        f.write("#chrom\tstart\tend\tmotif\tis_het\tallele_diff\t"
                "hp_max_hii\tpooled_hii\thp_path\tn_total\n")
        for c in comparisons:
            f.write(
                f"{c['chrom']}\t{c['start']}\t{c['end']}\t{c['motif']}\t"
                f"{'HET' if c['is_het'] else 'HOM'}\t{c['allele_diff']:.1f}\t"
                f"{c['hp_max_hii']:.4f}\t{c['pooled_hii']:.4f}\t"
                f"{c['hp_path']}\t{c['n_total']}\n"
            )
    logger.info("Comparison TSV: %s", tsv_path)

    # Stratify
    het_comps = [c for c in comparisons if c["is_het"]]
    hom_comps = [c for c in comparisons if not c["is_het"]]

    # Build report
    lines = []
    w = lines.append

    w("=" * 80)
    w("  HP-TAGGED vs POOLED INSTABILITY COMPARISON")
    w("=" * 80)

    w(f"\n  OVERVIEW")
    w(f"  Total loci compared:    {len(comparisons):,}")
    w(f"  HET loci:               {len(het_comps):,} "
      f"({len(het_comps)/len(comparisons)*100:.1f}%)")
    w(f"  HOM loci:               {len(hom_comps):,} "
      f"({len(hom_comps)/len(comparisons)*100:.1f}%)")
    w(f"  Elapsed:                {elapsed:.1f}s")

    # HET loci comparison
    if het_comps:
        het_hp = np.array([c["hp_max_hii"] for c in het_comps])
        het_pooled = np.array([c["pooled_hii"] for c in het_comps])

        w(f"\n  HET LOCI (n={len(het_comps):,})")
        w(f"  {'Metric':<20s} {'HP-tagged':>12s} {'Pooled':>12s} {'Ratio':>8s}")
        w("  " + "-" * 55)
        hp_mean = float(np.mean(het_hp))
        po_mean = float(np.mean(het_pooled))
        ratio_mean = po_mean / (hp_mean + 1e-9)
        w(f"  {'Mean HII':<20s} {hp_mean:>12.4f} {po_mean:>12.4f} "
          f"{ratio_mean:>7.1f}x")
        hp_med = float(np.median(het_hp))
        po_med = float(np.median(het_pooled))
        if hp_med > 0.001:
            ratio_med_str = f"{po_med / hp_med:>7.1f}x"
        elif po_med > 0.001:
            ratio_med_str = "    inf"
        else:
            ratio_med_str = "   1.0x"
        w(f"  {'Median HII':<20s} {hp_med:>12.4f} {po_med:>12.4f} "
          f"{ratio_med_str}")
        w(f"  {'P95 HII':<20s} {np.percentile(het_hp, 95):>12.4f} "
          f"{np.percentile(het_pooled, 95):>12.4f}")

        # False positive analysis
        threshold = 0.45
        hp_fp = int(np.sum(het_hp >= threshold))
        po_fp = int(np.sum(het_pooled >= threshold))
        w(f"\n  FALSE POSITIVE ANALYSIS (threshold = {threshold})")
        w(f"  HP-tagged FP:   {hp_fp:>6,} / {len(het_comps):,} "
          f"({hp_fp/len(het_comps)*100:.2f}%)")
        w(f"  Pooled FP:      {po_fp:>6,} / {len(het_comps):,} "
          f"({po_fp/len(het_comps)*100:.2f}%)")
        if po_fp > hp_fp:
            reduction = (po_fp - hp_fp) / po_fp * 100
            w(f"  FP reduction:   {reduction:.1f}% "
              f"({po_fp - hp_fp:,} false calls eliminated by haplotype separation)")

    # HOM loci comparison
    if hom_comps:
        hom_hp = np.array([c["hp_max_hii"] for c in hom_comps])
        hom_pooled = np.array([c["pooled_hii"] for c in hom_comps])

        w(f"\n  HOM LOCI (n={len(hom_comps):,})")
        w(f"  {'Metric':<20s} {'HP-tagged':>12s} {'Pooled':>12s}")
        w("  " + "-" * 45)
        w(f"  {'Mean HII':<20s} {np.mean(hom_hp):>12.4f} "
          f"{np.mean(hom_pooled):>12.4f}")
        w(f"  {'Median HII':<20s} {np.median(hom_hp):>12.4f} "
          f"{np.median(hom_pooled):>12.4f}")
        if len(hom_hp) > 1:
            corr = float(np.corrcoef(hom_hp, hom_pooled)[0, 1])
            w(f"  {'Correlation':<20s} {corr:>12.4f}")

    # Key finding
    w(f"\n  KEY FINDING")
    if het_comps:
        w(f"  Per-haplotype decomposition reduces mean HII at HET loci by "
          f"{ratio_mean:.1f}x.")
        w(f"  Pooled analysis produces {po_fp:,} false-positive instability "
          f"calls at HET loci")
        w(f"  that are correctly identified as stable by per-haplotype analysis.")
        w(f"  This demonstrates that haplotype separation is essential for")
        w(f"  accurate somatic instability measurement at heterozygous loci.")

    w(f"\n{'=' * 80}")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(report)
    logger.info("Report saved to %s", output_path)

    # Figures
    if HAS_MPL and comparisons:
        fig_dir = output_path.parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel A: Scatter plot (pooled x-axis, hp-tagged y-axis)
        if hom_comps:
            axes[0].scatter(
                [c["pooled_hii"] for c in hom_comps],
                [c["hp_max_hii"] for c in hom_comps],
                s=1, alpha=0.2, c="#4C72B0", label=f"HOM (n={len(hom_comps):,})",
                rasterized=True,
            )
        if het_comps:
            axes[0].scatter(
                [c["pooled_hii"] for c in het_comps],
                [c["hp_max_hii"] for c in het_comps],
                s=1, alpha=0.2, c="#C44E52", label=f"HET (n={len(het_comps):,})",
                rasterized=True,
            )
        all_vals = [c["pooled_hii"] for c in comparisons] + \
                   [c["hp_max_hii"] for c in comparisons]
        max_val = np.percentile(all_vals, 99.5) if all_vals else 1.0
        axes[0].plot([0, max_val], [0, max_val], "k--", alpha=0.4, lw=0.8,
                     label="y=x")
        axes[0].set_xlim(0, max_val)
        axes[0].set_ylim(0, max_val)
        axes[0].set_xlabel("Pooled HII (no haplotype separation)")
        axes[0].set_ylabel("HP-tagged max(HII) (per-haplotype)")
        axes[0].set_title("Per-haplotype vs Pooled Instability")
        axes[0].legend(markerscale=5, fontsize=9)

        # Panel B: HII distribution comparison at HET loci
        if het_comps:
            het_hp_vals = [c["hp_max_hii"] for c in het_comps]
            het_po_vals = [c["pooled_hii"] for c in het_comps]
            bins = np.linspace(0, np.percentile(het_po_vals, 99), 60)
            axes[1].hist(het_hp_vals, bins=bins, alpha=0.6,
                         label="HP-tagged", density=True, color="#4C72B0")
            axes[1].hist(het_po_vals, bins=bins, alpha=0.6,
                         label="Pooled", density=True, color="#C44E52")
            axes[1].axvline(0.45, color="black", ls="--", lw=0.8,
                            label="Threshold (0.45)")
            axes[1].set_xlabel("HII")
            axes[1].set_ylabel("Density")
            axes[1].set_title(f"HII at HET Loci (n={len(het_comps):,})")
            axes[1].legend(fontsize=9)

        plt.tight_layout()
        fig_path = fig_dir / "hp_vs_pooled_scatter.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info("Figure saved: %s", fig_path)


# ===========================================================================
# Subcommand: trio
# ===========================================================================

def cmd_trio(args):
    """Compare instability consistency across HG002/HG003/HG004 trio."""

    def load_hii_by_locus(path: str) -> dict:
        rows = load_instability_tsv(path)
        result = {}
        for row in rows:
            if row.get("analysis_path") == "failed":
                continue
            key = (row["chrom"], row["start"], row["end"])
            hii_h1 = parse_float(row.get("hii_h1", "."))
            hii_h2 = parse_float(row.get("hii_h2", "."))
            result[key] = {
                "max_hii": max(hii_h1, hii_h2),
                "hii_h1": hii_h1,
                "hii_h2": hii_h2,
                "ais": parse_float(row.get("ais", ".")),
                "path": row.get("analysis_path", "unknown"),
            }
        return result

    hg002 = load_hii_by_locus(args.hg002)
    hg003 = load_hii_by_locus(args.hg003)
    hg004 = load_hii_by_locus(args.hg004)

    logger.info("HG002: %d loci, HG003: %d loci, HG004: %d loci",
                len(hg002), len(hg003), len(hg004))

    common = sorted(set(hg002.keys()) & set(hg003.keys()) & set(hg004.keys()))
    logger.info("Common loci: %d", len(common))

    if not common:
        logger.error("No common loci found")
        return

    hiis_002 = np.array([hg002[k]["max_hii"] for k in common])
    hiis_003 = np.array([hg003[k]["max_hii"] for k in common])
    hiis_004 = np.array([hg004[k]["max_hii"] for k in common])

    # Report
    lines = []
    w = lines.append

    w("=" * 80)
    w("  TRIO INSTABILITY CONSISTENCY (HG002 / HG003 / HG004)")
    w("=" * 80)

    w(f"\n  OVERVIEW")
    w(f"  Common loci:     {len(common):,}")
    w(f"  HG002 total:     {len(hg002):,}")
    w(f"  HG003 total:     {len(hg003):,}")
    w(f"  HG004 total:     {len(hg004):,}")

    w(f"\n  NOTE: HG003/HG004 BAMs lack HP tags.")
    w(f"  HG002 uses hp-tagged analysis; HG003/HG004 use gap-split/pooled fallback.")
    w(f"  This comparison tests measurement reproducibility across samples,")
    w(f"  not per-haplotype decomposition quality.")

    # Per-sample stats
    w(f"\n  PER-SAMPLE HII STATISTICS (at common loci)")
    w(f"  {'Sample':<8s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} "
      f"{'P95':>8s} {'P99':>8s} {'Max':>8s}")
    w("  " + "-" * 60)
    for name, hiis in [("HG002", hiis_002), ("HG003", hiis_003),
                        ("HG004", hiis_004)]:
        w(f"  {name:<8s} {np.mean(hiis):>8.4f} {np.median(hiis):>8.4f} "
          f"{np.std(hiis):>8.4f} {np.percentile(hiis, 95):>8.4f} "
          f"{np.percentile(hiis, 99):>8.4f} {np.max(hiis):>8.4f}")

    # Pairwise correlations
    corr_02_03 = float(np.corrcoef(hiis_002, hiis_003)[0, 1])
    corr_02_04 = float(np.corrcoef(hiis_002, hiis_004)[0, 1])
    corr_03_04 = float(np.corrcoef(hiis_003, hiis_004)[0, 1])

    w(f"\n  PAIRWISE HII CORRELATIONS")
    w(f"  HG002-HG003:    {corr_02_03:.4f}")
    w(f"  HG002-HG004:    {corr_02_04:.4f}")
    w(f"  HG003-HG004:    {corr_03_04:.4f}")

    # Consistency
    threshold = 0.45
    all_low = int(np.sum(
        (hiis_002 < threshold) & (hiis_003 < threshold) & (hiis_004 < threshold)
    ))
    any_high = int(np.sum(
        (hiis_002 >= threshold) | (hiis_003 >= threshold) | (hiis_004 >= threshold)
    ))
    all_high = int(np.sum(
        (hiis_002 >= threshold) & (hiis_003 >= threshold) & (hiis_004 >= threshold)
    ))

    w(f"\n  CONSISTENCY AT THRESHOLD = {threshold}")
    w(f"  All 3 low:       {all_low:>8,} ({all_low/len(common)*100:.1f}%)")
    w(f"  Any high:        {any_high:>8,} ({any_high/len(common)*100:.2f}%)")
    w(f"  All 3 high:      {all_high:>8,} ({all_high/len(common)*100:.2f}%)")

    # Shared outliers (up to 30)
    if any_high > 0:
        w(f"\n  OUTLIER LOCI (HII >= {threshold} in any sample, up to 30)")
        w(f"  {'chrom':<8s} {'start':>12s} {'end':>12s} "
          f"{'HG002':>8s} {'HG003':>8s} {'HG004':>8s} {'shared':<8s}")
        w("  " + "-" * 65)
        shown = 0
        for k in common:
            h002 = hg002[k]["max_hii"]
            h003 = hg003[k]["max_hii"]
            h004 = hg004[k]["max_hii"]
            if max(h002, h003, h004) >= threshold:
                n_hi = sum(1 for x in [h002, h003, h004] if x >= threshold)
                shared = "all" if n_hi == 3 else f"{n_hi}/3"
                w(f"  {k[0]:<8s} {k[1]:>12s} {k[2]:>12s} "
                  f"{h002:>8.4f} {h003:>8.4f} {h004:>8.4f} {shared:<8s}")
                shown += 1
                if shown >= 30:
                    remaining = any_high - shown
                    if remaining > 0:
                        w(f"  ... ({remaining} more)")
                    break

    # Analysis path breakdown
    w(f"\n  ANALYSIS PATH BREAKDOWN")
    for name, data in [("HG002", hg002), ("HG003", hg003), ("HG004", hg004)]:
        paths_local: dict[str, int] = {}
        for k in common:
            p = data[k]["path"]
            paths_local[p] = paths_local.get(p, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(paths_local.items()))
        w(f"  {name}: {parts}")

    w(f"\n{'=' * 80}")

    report = "\n".join(lines)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(report)
    logger.info("Report saved to %s", output_path)

    # Figures
    if HAS_MPL:
        fig_dir = output_path.parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        pairs = [
            (hiis_002, hiis_003, "HG002", "HG003", corr_02_03),
            (hiis_002, hiis_004, "HG002", "HG004", corr_02_04),
            (hiis_003, hiis_004, "HG003", "HG004", corr_03_04),
        ]
        for ax, (x, y, xlab, ylab, corr) in zip(axes, pairs):
            ax.scatter(x, y, s=0.5, alpha=0.15, c="#4C72B0", rasterized=True)
            lim = max(np.percentile(x, 99.5), np.percentile(y, 99.5), 0.1)
            ax.plot([0, lim], [0, lim], "r--", alpha=0.5, lw=0.8)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel(f"{xlab} max(HII)")
            ax.set_ylabel(f"{ylab} max(HII)")
            ax.set_title(f"r = {corr:.4f}")

        plt.suptitle("Trio HII Consistency", fontsize=13)
        plt.tight_layout()
        fig_path = fig_dir / "trio_consistency.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info("Figure saved: %s", fig_path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HG002 instability validation: genome-wide, hp-vs-pooled, trio",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # genomewide
    p_gw = subparsers.add_parser(
        "genomewide", help="Analyze genome-wide instability TSV")
    p_gw.add_argument("--tsv", required=True, help="Genome-wide instability TSV")
    p_gw.add_argument("--output", required=True, help="Output report file")

    # hp-vs-pooled
    p_hp = subparsers.add_parser(
        "hp-vs-pooled", help="HP-tagged vs pooled comparison")
    p_hp.add_argument("--bam", default=BAM_HG002, help="HG002 BAM path")
    p_hp.add_argument("--loci", default=TEST_LOCI, help="Loci BED file")
    p_hp.add_argument("--output", required=True, help="Output report file")
    p_hp.add_argument("--max-loci", type=int, default=0,
                      help="Max loci to process (0=all)")

    # trio
    p_trio = subparsers.add_parser(
        "trio", help="Trio instability consistency")
    p_trio.add_argument("--hg002", required=True, help="HG002 instability TSV")
    p_trio.add_argument("--hg003", required=True, help="HG003 instability TSV")
    p_trio.add_argument("--hg004", required=True, help="HG004 instability TSV")
    p_trio.add_argument("--output", required=True, help="Output report file")

    args = parser.parse_args()

    if args.command == "genomewide":
        cmd_genomewide(args)
    elif args.command == "hp-vs-pooled":
        cmd_hp_vs_pooled(args)
    elif args.command == "trio":
        cmd_trio(args)


if __name__ == "__main__":
    main()
