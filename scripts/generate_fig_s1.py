#!/usr/bin/env python3
"""Generate Supplementary Figure S1: Exponential distribution fit for ATXN10 carrier HG01122.

Extracts read-level allele sizes from the expanded haplotype (HP=2) of HG01122
at the ATXN10 locus, fits an exponential distribution to the expansion sizes
(allele size - minimum observed allele), and produces a two-panel figure:
  (a) Histogram of expansion sizes with exponential fit overlay
  (b) Q-Q plot of observed vs theoretical exponential quantiles

The biological model: the smallest observed allele represents the "base" expanded
allele; additional expansion follows an exponential distribution characteristic of
somatic repeat expansion in disease carriers.

Reference: manuscript line 98 -- KS test p = 0.67
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosaictr.genotype import extract_reads_enhanced

import pysam

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BAM_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "output/instability/1000g/HG01122_ATXN10_region.bam",
)
CHROM = "chr22"
START = 45795354
END = 45795424
MOTIF = "ATTCT"
MOTIF_LEN = len(MOTIF)
REF_SIZE = END - START  # 70 bp = 14.0 RU

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")
OUT_PATH = os.path.join(OUT_DIR, "fig_s1_exponential_fit.png")


def main():
    # ------------------------------------------------------------------
    # 1. Extract reads
    # ------------------------------------------------------------------
    bam = pysam.AlignmentFile(BAM_PATH, "rb")
    try:
        reads = extract_reads_enhanced(
            bam, CHROM, START, END,
            min_mapq=5, min_flank=50, max_reads=200, motif_len=MOTIF_LEN,
        )
    finally:
        bam.close()

    print(f"Total reads extracted: {len(reads)}")

    # Separate by HP tag
    hp1 = [r for r in reads if r.hp == 1]
    hp2 = [r for r in reads if r.hp == 2]
    hp0 = [r for r in reads if r.hp == 0]
    print(f"HP=1: {len(hp1)}, HP=2: {len(hp2)}, HP=0: {len(hp0)}")

    # Identify expanded haplotype (higher median)
    hp1_sizes = np.array([r.allele_size for r in hp1])
    hp2_sizes = np.array([r.allele_size for r in hp2])

    if np.median(hp1_sizes) > np.median(hp2_sizes):
        exp_sizes_bp = hp1_sizes
        exp_label = "HP=1"
    else:
        exp_sizes_bp = hp2_sizes
        exp_label = "HP=2"

    exp_ru = exp_sizes_bp / MOTIF_LEN
    median_ru = np.median(exp_ru)
    print(f"\nExpanded haplotype ({exp_label}):")
    print(f"  n = {len(exp_ru)}")
    print(f"  Median = {median_ru:.1f} RU")
    print(f"  Range = {exp_ru.min():.1f} -- {exp_ru.max():.1f} RU")

    # ------------------------------------------------------------------
    # 2. Compute expansion sizes (allele - minimum observed allele)
    # ------------------------------------------------------------------
    # The minimum allele represents the base expanded allele; deviations
    # above it reflect somatic expansion events.
    min_bp = exp_sizes_bp.min()
    expansions = exp_sizes_bp - min_bp
    expansions_nz = expansions[expansions > 0]

    print(f"\nExpansion sizes from base allele ({min_bp:.0f} bp = {min_bp/MOTIF_LEN:.1f} RU):")
    print(f"  n (nonzero) = {len(expansions_nz)}")
    print(f"  Mean = {np.mean(expansions_nz):.1f} bp")
    print(f"  Values: {np.sort(expansions_nz).astype(int)}")

    # ------------------------------------------------------------------
    # 3. Fit exponential distribution
    # ------------------------------------------------------------------
    _, scale_fit = stats.expon.fit(expansions_nz, floc=0)
    ks_stat, ks_p = stats.kstest(expansions_nz, "expon", args=(0, scale_fit))

    print(f"\nExponential fit: scale = {scale_fit:.1f} bp")
    print(f"KS test: D = {ks_stat:.4f}, p = {ks_p:.4f}")

    # ------------------------------------------------------------------
    # 4. Create figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Color palette
    hist_color = "#3274A1"
    fit_color = "#E1812C"
    qq_color = "#3A923A"

    # --- Panel (a): Histogram with exponential fit ---
    ax = axes[0]
    n_bins = 8
    counts, bin_edges, patches = ax.hist(
        expansions_nz, bins=n_bins, density=True,
        color=hist_color, alpha=0.7, edgecolor="white", linewidth=0.8,
        label="Observed reads",
    )
    x_fit = np.linspace(0, expansions_nz.max() * 1.15, 300)
    y_fit = stats.expon.pdf(x_fit, loc=0, scale=scale_fit)
    ax.plot(x_fit, y_fit, color=fit_color, linewidth=2.2,
            label=f"Exp. fit ($\\lambda^{{-1}}$ = {scale_fit:.0f} bp)")

    ax.set_xlabel("Expansion beyond base allele (bp)", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title("(a)", fontsize=13, fontweight="bold", loc="left")

    ax.legend(fontsize=9, frameon=True, fancybox=True, framealpha=0.8,
              edgecolor="#cccccc")

    # Annotation
    ax.text(
        0.95, 0.82,
        f"KS $D$ = {ks_stat:.3f}\n$p$ = {ks_p:.2f}\n$n$ = {len(expansions_nz)}",
        transform=ax.transAxes, fontsize=9.5,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0",
                  edgecolor="#cccccc", alpha=0.9),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel (b): Q-Q plot ---
    ax = axes[1]
    sorted_obs = np.sort(expansions_nz)
    n = len(sorted_obs)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.expon.ppf(probs, loc=0, scale=scale_fit)

    ax.scatter(theoretical, sorted_obs, s=55, c=qq_color, edgecolors="white",
               linewidth=0.8, zorder=3)
    # Reference line y=x
    lim_max = max(theoretical.max(), sorted_obs.max()) * 1.08
    ax.plot([0, lim_max], [0, lim_max], color="#888888", linestyle="--",
            linewidth=1.2, alpha=0.7, label="$y = x$", zorder=2)

    ax.set_xlabel("Theoretical exponential quantiles (bp)", fontsize=11)
    ax.set_ylabel("Observed quantiles (bp)", fontsize=11)
    ax.set_title("(b)", fontsize=13, fontweight="bold", loc="left")
    ax.set_xlim(-30, lim_max)
    ax.set_ylim(-30, lim_max)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, frameon=True, fancybox=True, framealpha=0.8,
              edgecolor="#cccccc", loc="lower right")

    # Correlation annotation
    r_val = np.corrcoef(theoretical, sorted_obs)[0, 1]
    ax.text(
        0.05, 0.95,
        f"$r$ = {r_val:.3f}",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                  edgecolor="#cccccc", alpha=0.9),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Suptitle
    fig.suptitle(
        "HG01122 ATXN10 expanded allele (1,041 RU): exponential distribution fit",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to: {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
