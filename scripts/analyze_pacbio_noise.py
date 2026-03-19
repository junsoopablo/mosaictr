#!/usr/bin/env python3
"""Analyze read-level allele size distributions at TR loci.

Characterize PacBio HiFi noise at stable and unstable loci to inform
optimal dispersion metric choice.

Outputs:
  - Per-read deviation distributions at stable loci
  - Noise vs motif length, repeat length
  - Comparison of metric sensitivity (AAD, MAD, Qn, SD, CV)
  - Carrier locus distributions for visual inspection

Usage:
  python scripts/analyze_pacbio_noise.py \
    --bam HG002.bam \
    --loci test_loci.bed \
    --output output/instability/noise_analysis/
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.genotype import ReadInfo, extract_reads_enhanced
from mosaictr.utils import load_loci_bed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Dispersion metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(sizes: np.ndarray, motif_len: int) -> dict:
    """Compute multiple dispersion metrics for comparison."""
    n = len(sizes)
    if n < 2 or motif_len < 1:
        return {k: 0.0 for k in ["mad", "qn", "aad", "sd", "cv", "iqr",
                                   "p90_dev", "frac_deviant",
                                   "mad_norm", "qn_norm", "aad_norm", "sd_norm"]}

    med = np.median(sizes)
    abs_devs = np.abs(sizes - med)

    # MAD
    mad = float(np.median(abs_devs))

    # AAD (average absolute deviation from median)
    aad = float(np.mean(abs_devs))

    # Qn (first quartile of pairwise distances)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(abs(sizes[i] - sizes[j]))
    diffs.sort()
    k = max(0, int(math.ceil(0.25 * len(diffs))) - 1)
    qn = diffs[k]

    # SD
    sd = float(np.std(sizes, ddof=1)) if n > 1 else 0.0

    # CV
    mean_val = float(np.mean(sizes))
    cv = sd / abs(mean_val) if abs(mean_val) > 0.001 else 0.0

    # IQR
    q75, q25 = np.percentile(sizes, [75, 25])
    iqr = float(q75 - q25)

    # 90th percentile of |deviation|
    p90_dev = float(np.percentile(abs_devs, 90))

    # Fraction of reads deviating by >= 1 motif unit
    frac_deviant = float(np.mean(abs_devs >= motif_len))

    return {
        "mad": mad,
        "qn": qn,
        "aad": aad,
        "sd": sd,
        "cv": cv,
        "iqr": iqr,
        "p90_dev": p90_dev,
        "frac_deviant": frac_deviant,
        # Normalized by motif_len
        "mad_norm": mad / motif_len,
        "qn_norm": qn / motif_len,
        "aad_norm": aad / motif_len,
        "sd_norm": sd / motif_len,
    }


# ---------------------------------------------------------------------------
# Sampling analysis
# ---------------------------------------------------------------------------

def analyze_noise(bam_path: str, loci: list, n_sample: int = 5000,
                  seed: int = 42) -> dict:
    """Sample loci and characterize read-level noise."""
    import pysam

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(loci), min(n_sample, len(loci)), replace=False)
    sampled_loci = [loci[i] for i in sorted(indices)]

    bam = pysam.AlignmentFile(bam_path, "rb")

    per_locus_metrics = []
    per_read_devs = []

    try:
        for i, (chrom, start, end, motif) in enumerate(sampled_loci):
            if i % 500 == 0:
                logger.info("Processing locus %d/%d ...", i, len(sampled_loci))

            motif_len = len(motif)
            ref_size = end - start

            reads = extract_reads_enhanced(
                bam, chrom, start, end,
                min_mapq=5, min_flank=50, max_reads=200,
                ref_fasta=None, motif_len=motif_len,
            )

            if len(reads) < 5:
                continue

            sizes = np.array([r.allele_size for r in reads], dtype=float)
            med = np.median(sizes)
            abs_devs = np.abs(sizes - med)

            for d in abs_devs:
                per_read_devs.append({
                    "dev_bp": float(d),
                    "motif_len": motif_len,
                    "ref_size": ref_size,
                })

            metrics = compute_all_metrics(sizes, motif_len)
            metrics["chrom"] = chrom
            metrics["start"] = start
            metrics["end"] = end
            metrics["motif"] = motif
            metrics["motif_len"] = motif_len
            metrics["ref_size"] = ref_size
            metrics["n_reads"] = len(reads)
            metrics["median"] = float(med)
            per_locus_metrics.append(metrics)

    finally:
        bam.close()

    return {
        "per_locus": per_locus_metrics,
        "per_read_devs": per_read_devs,
        "n_loci_analyzed": len(per_locus_metrics),
    }


def analyze_carriers(carrier_bams: list, carrier_loci: list) -> list:
    """Analyze read-level distributions at known carrier loci."""
    import pysam

    results = []
    for bam_path, locus_info in zip(carrier_bams, carrier_loci):
        chrom, start, end, motif, label = locus_info
        motif_len = len(motif)

        bam = pysam.AlignmentFile(bam_path, "rb")
        try:
            reads = extract_reads_enhanced(
                bam, chrom, start, end,
                min_mapq=5, min_flank=50, max_reads=500,
                ref_fasta=None, motif_len=motif_len,
            )
        finally:
            bam.close()

        if not reads:
            continue

        hp_sizes = defaultdict(list)
        for r in reads:
            hp_sizes[r.hp].append(r.allele_size)

        all_sizes = np.array([r.allele_size for r in reads], dtype=float)

        result = {
            "label": label,
            "bam": bam_path,
            "motif": motif,
            "motif_len": motif_len,
            "n_reads": len(reads),
            "all_sizes": all_sizes,
            "hp_sizes": dict(hp_sizes),
        }

        for hp_label, hp_key in [("hp1", 1), ("hp2", 2), ("hp0", 0)]:
            if hp_key in hp_sizes and len(hp_sizes[hp_key]) >= 3:
                s = np.array(hp_sizes[hp_key], dtype=float)
                m = compute_all_metrics(s, motif_len)
                result[f"{hp_label}_metrics"] = m
                result[f"{hp_label}_sizes"] = s
            else:
                result[f"{hp_label}_metrics"] = None

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_noise_report(data: dict, output_dir: str):
    """Write noise analysis report."""
    loci = data["per_locus"]
    devs = data["per_read_devs"]

    lines = []
    w = lines.append

    w("=" * 90)
    w("  PacBio HiFi TR Sizing Noise Analysis")
    w("=" * 90)
    w(f"\n  Loci analyzed: {data['n_loci_analyzed']:,}")
    w(f"  Total per-read deviations: {len(devs):,}")

    all_devs = np.array([d["dev_bp"] for d in devs])
    w(f"\n  Overall per-read |deviation from median| (bp):")
    w(f"    Mean:   {np.mean(all_devs):.3f}")
    w(f"    Median: {np.median(all_devs):.3f}")
    w(f"    P90:    {np.percentile(all_devs, 90):.3f}")
    w(f"    P95:    {np.percentile(all_devs, 95):.3f}")
    w(f"    P99:    {np.percentile(all_devs, 99):.3f}")
    w(f"    Max:    {np.max(all_devs):.3f}")
    w(f"    % exact (dev=0): {np.mean(all_devs == 0) * 100:.1f}%")
    w(f"    % within 1bp:    {np.mean(all_devs <= 1) * 100:.1f}%")
    w(f"    % within 3bp:    {np.mean(all_devs <= 3) * 100:.1f}%")

    # Per motif length
    w(f"\n  Per-read noise by motif length:")
    w(f"  {'Motif':>6s} {'N_reads':>8s} {'Mean|dev|':>10s} {'Med|dev|':>10s} "
      f"{'P90':>8s} {'%exact':>8s} {'%≤1bp':>8s}")
    w("  " + "-" * 66)

    motif_groups = defaultdict(list)
    for d in devs:
        motif_groups[d["motif_len"]].append(d["dev_bp"])

    for ml in sorted(motif_groups.keys()):
        ds = np.array(motif_groups[ml])
        w(f"  {ml:>6d} {len(ds):>8d} {np.mean(ds):>10.3f} {np.median(ds):>10.3f} "
          f"{np.percentile(ds, 90):>8.3f} {np.mean(ds == 0) * 100:>7.1f}% "
          f"{np.mean(ds <= 1) * 100:>7.1f}%")

    # Locus classification
    stable_loci = [l for l in loci if l["mad"] == 0 and l["sd"] < 1.0]
    noisy_loci = [l for l in loci if l["mad"] > 0 and l["sd"] < 5.0]
    unstable_loci = [l for l in loci if l["sd"] >= 5.0]

    w(f"\n  Locus classification:")
    w(f"    Perfectly stable (MAD=0, SD<1): {len(stable_loci):,} ({len(stable_loci)/len(loci)*100:.1f}%)")
    w(f"    Noisy but stable (MAD>0, SD<5): {len(noisy_loci):,} ({len(noisy_loci)/len(loci)*100:.1f}%)")
    w(f"    Potentially unstable (SD>=5):   {len(unstable_loci):,} ({len(unstable_loci)/len(loci)*100:.1f}%)")

    # Metric comparison at noisy loci
    if noisy_loci:
        w(f"\n  Metric values at noisy-but-stable loci (n={len(noisy_loci)}):")
        for metric in ["mad_norm", "qn_norm", "aad_norm", "sd_norm"]:
            vals = [l[metric] for l in noisy_loci]
            w(f"    {metric:<10s}: mean={np.mean(vals):.4f}, "
              f"median={np.median(vals):.4f}, P95={np.percentile(vals, 95):.4f}")

    # Noise vs repeat size
    w(f"\n  Noise vs reference repeat size:")
    w(f"  {'RefSize':>10s} {'N_loci':>8s} {'Mean AAD':>10s} {'Mean SD':>10s} "
      f"{'Mean MAD':>10s} {'Mean Qn':>10s}")
    w("  " + "-" * 58)

    size_bins = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 10000)]
    for lo, hi in size_bins:
        subset = [l for l in loci if lo <= l["ref_size"] < hi]
        if not subset:
            continue
        w(f"  {lo:>4d}-{hi:<5d} {len(subset):>8d} "
          f"{np.mean([l['aad'] for l in subset]):>10.3f} "
          f"{np.mean([l['sd'] for l in subset]):>10.3f} "
          f"{np.mean([l['mad'] for l in subset]):>10.3f} "
          f"{np.mean([l['qn'] for l in subset]):>10.3f}")

    w(f"\n{'=' * 90}")

    report_path = os.path.join(output_dir, "noise_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", report_path)
    print("\n".join(lines))

    return {
        "all_devs": all_devs,
        "motif_groups": motif_groups,
        "stable_loci": stable_loci,
        "noisy_loci": noisy_loci,
        "unstable_loci": unstable_loci,
    }


def plot_noise(report_data: dict, carrier_results: list, output_dir: str):
    """Generate noise analysis figures."""
    if not HAS_MPL:
        logger.warning("matplotlib not available")
        return

    all_devs = report_data["all_devs"]
    motif_groups = report_data["motif_groups"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Overall per-read deviation histogram
    ax = axes[0, 0]
    nonzero = all_devs[all_devs > 0]
    if len(nonzero) > 0:
        bins = np.arange(0, min(20, np.percentile(nonzero, 99)) + 1, 0.5)
        ax.hist(nonzero, bins=bins, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.set_xlabel("|Deviation from median| (bp)")
    ax.set_ylabel("Count")
    ax.set_title(f"(a) Per-read noise distribution\n"
                 f"({np.mean(all_devs == 0)*100:.0f}% exact, "
                 f"mean={np.mean(all_devs):.2f}bp)")
    ax.axvline(1, color="red", linestyle="--", linewidth=0.8, label="1 bp")
    ax.axvline(3, color="orange", linestyle="--", linewidth=0.8, label="3 bp")
    ax.legend(fontsize=8)

    # (b) Noise by motif length
    ax = axes[0, 1]
    mls = sorted(motif_groups.keys())
    positions = range(len(mls))
    bp_data = [motif_groups[ml] for ml in mls]
    bp = ax.boxplot(bp_data, positions=list(positions), widths=0.6,
                    showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#90CAF9')
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{ml}bp" for ml in mls])
    ax.set_xlabel("Motif length")
    ax.set_ylabel("|Deviation| (bp)")
    ax.set_title("(b) Noise by motif length")
    ax.set_ylim(0, min(15, ax.get_ylim()[1]))

    # (c) Metric sensitivity comparison: simulate increasing expansion fraction
    ax = axes[0, 2]
    fractions = np.arange(0, 0.55, 0.02)
    n_reads = 30
    n_sims = 50
    rng = np.random.default_rng(42)

    metric_traces = {"MAD/l": [], "Qn/l": [], "AAD/l": [], "SD/l": []}
    motif_len_sim = 3
    expansion_bp = 30

    for frac in fractions:
        n_exp = int(round(frac * n_reads))
        n_stable = n_reads - n_exp

        mad_vals, qn_vals, aad_vals, sd_vals = [], [], [], []
        for _ in range(n_sims):
            stable = rng.normal(100, 0.5, n_stable)
            if n_exp > 0:
                expanded = rng.normal(100 + expansion_bp, 1.0, n_exp)
                sizes = np.concatenate([stable, expanded])
            else:
                sizes = stable

            m = compute_all_metrics(sizes, motif_len_sim)
            mad_vals.append(m["mad_norm"])
            qn_vals.append(m["qn_norm"])
            aad_vals.append(m["aad_norm"])
            sd_vals.append(m["sd_norm"])

        metric_traces["MAD/l"].append(np.mean(mad_vals))
        metric_traces["Qn/l"].append(np.mean(qn_vals))
        metric_traces["AAD/l"].append(np.mean(aad_vals))
        metric_traces["SD/l"].append(np.mean(sd_vals))

    colors = {"MAD/l": "#E53935", "Qn/l": "#FF9800", "AAD/l": "#2196F3", "SD/l": "#4CAF50"}
    for name, trace in metric_traces.items():
        ax.plot(fractions * 100, trace, label=name, color=colors[name], linewidth=2)
    ax.set_xlabel("Expansion fraction (%)")
    ax.set_ylabel("Metric value (normalized by motif len)")
    ax.set_title(f"(c) Metric sensitivity\n(n={n_reads}, expansion={expansion_bp}bp)")
    ax.legend(fontsize=8)
    ax.axhline(0.45, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(17, color="gray", linestyle=":", linewidth=0.5)
    ax.text(18, 0.1, "17%", fontsize=7, color="gray")

    # (d) AAD sensitivity vs noise level
    ax = axes[1, 0]
    noise_levels = [0.0, 0.3, 0.5, 1.0, 2.0, 3.0]
    frac_fixed = 0.15
    n_exp_fixed = int(round(frac_fixed * n_reads))
    n_stable_fixed = n_reads - n_exp_fixed

    for noise_std in noise_levels:
        aad_by_exp = []
        exp_sizes = np.arange(0, 61, 3)
        for exp_bp in exp_sizes:
            vals = []
            for _ in range(n_sims):
                stable = rng.normal(100, noise_std, n_stable_fixed)
                if n_exp_fixed > 0 and exp_bp > 0:
                    expanded = rng.normal(100 + exp_bp, noise_std, n_exp_fixed)
                    sizes = np.concatenate([stable, expanded])
                else:
                    sizes = rng.normal(100, noise_std, n_reads)
                m = compute_all_metrics(sizes, motif_len_sim)
                vals.append(m["aad_norm"])
            aad_by_exp.append(np.mean(vals))
        ax.plot(exp_sizes, aad_by_exp, label=f"noise s={noise_std}bp", linewidth=1.5)

    ax.set_xlabel("Expansion amount (bp)")
    ax.set_ylabel("AAD/motif_len")
    ax.set_title(f"(d) AAD sensitivity vs noise level\n(15% expansion fraction)")
    ax.legend(fontsize=7)
    ax.axhline(0.45, color="gray", linestyle=":", linewidth=0.8)

    # (e) Carrier locus read distributions
    ax = axes[1, 1]
    if carrier_results:
        for cr in carrier_results[:3]:
            label = cr["label"]
            for hp_key, color, hp_name in [(1, "#2196F3", "HP1"), (2, "#E53935", "HP2")]:
                if hp_key in cr["hp_sizes"] and len(cr["hp_sizes"][hp_key]) >= 3:
                    sizes = np.array(cr["hp_sizes"][hp_key])
                    sizes_ru = sizes / cr["motif_len"]
                    ax.hist(sizes_ru, bins=30, alpha=0.4,
                            label=f"{label} {hp_name}", density=True)
        ax.set_xlabel("Allele size (repeat units)")
        ax.set_ylabel("Density")
        ax.set_title("(e) Carrier read distributions")
        ax.legend(fontsize=6, ncol=2)
    else:
        ax.text(0.5, 0.5, "No carrier data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(e) Carrier read distributions")

    # (f) Metric comparison at carrier loci
    ax = axes[1, 2]
    if carrier_results:
        labels = []
        metric_vals = {"MAD/l": [], "Qn/l": [], "AAD/l": [], "SD/l": []}
        for cr in carrier_results:
            for hp_label in ["hp2", "hp1"]:
                m = cr.get(f"{hp_label}_metrics")
                if m and m.get("aad_norm", 0) > 0.1:
                    labels.append(f"{cr['label']}\n{hp_label}")
                    metric_vals["MAD/l"].append(m["mad_norm"])
                    metric_vals["Qn/l"].append(m["qn_norm"])
                    metric_vals["AAD/l"].append(m["aad_norm"])
                    metric_vals["SD/l"].append(m["sd_norm"])
                    break

        if labels:
            x = np.arange(len(labels))
            width = 0.2
            for i, (name, vals) in enumerate(metric_vals.items()):
                ax.bar(x + i * width, vals, width, label=name,
                       color=list(colors.values())[i], alpha=0.8)
            ax.set_xticks(x + 1.5 * width)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel("Metric value")
            ax.legend(fontsize=7)
    ax.set_title("(f) Metrics at carrier loci")

    fig.suptitle("PacBio HiFi TR Sizing Noise & Metric Sensitivity Analysis",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = os.path.join(output_dir, "noise_analysis.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", fig_path)


def main():
    parser = argparse.ArgumentParser(description="PacBio HiFi TR noise analysis")
    parser.add_argument("--bam", required=True, help="HG002 BAM file")
    parser.add_argument("--loci", required=True, help="Loci BED file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-sample", type=int, default=5000,
                        help="Number of loci to sample (default: 5000)")
    parser.add_argument("--carriers", action="store_true",
                        help="Also analyze ATXN10 carriers")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading loci from %s ...", args.loci)
    loci = load_loci_bed(args.loci)
    logger.info("Loaded %d loci", len(loci))

    logger.info("Analyzing noise on %d sampled loci ...", args.n_sample)
    data = analyze_noise(args.bam, loci, n_sample=args.n_sample)

    report_data = write_noise_report(data, str(output_dir))

    carrier_results = []
    if args.carriers:
        carrier_bams = [
            "output/instability/1000g/HG01122_ATXN10_region.bam",
            "output/instability/1000g/HG02252_ATXN10_region.bam",
            "output/instability/1000g/HG02345_ATXN10_region.bam",
        ]
        carrier_loci = [
            ("chr22", 45795354, 45795424, "ATTCT", "HG01122 (1041 RU)"),
            ("chr22", 45795354, 45795424, "ATTCT", "HG02252 (bilateral)"),
            ("chr22", 45795354, 45795424, "ATTCT", "HG02345 (320 RU)"),
        ]
        existing = [(b, l) for b, l in zip(carrier_bams, carrier_loci)
                     if os.path.exists(b)]
        if existing:
            bams, loci_c = zip(*existing)
            logger.info("Analyzing %d carrier loci ...", len(bams))
            carrier_results = analyze_carriers(list(bams), list(loci_c))

    plot_noise(report_data, carrier_results, str(output_dir))
    logger.info("Done!")


if __name__ == "__main__":
    main()
