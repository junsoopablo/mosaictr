#!/usr/bin/env python3
"""Generate publication-quality passage drift figure.

Side-by-side waterfall plots showing per-haplotype TR instability change
between p21 and p41 (20 passages apart) at selected loci.
Uses haplotagged BAMs for fair comparison.

Usage:
    python scripts/plot_passage_drift_figure.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pysam
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from mosaictr.genotype import extract_reads_enhanced, ReadInfo, _weighted_median

# --- Config ---
REF = "/vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
BAMS = {
    "p21": "/vault/external-datasets/2026/HG008_haplotagged/HG008_p21_haplotagged.bam",
    "p41": "/vault/external-datasets/2026/HG008_haplotagged/HG008_p41_haplotagged.bam",
}
OUTDIR = "output/passage_drift_hp/figures"

# Colors (colorblind-friendly Tol bright)
HP1_COLOR = "#4477AA"
HP2_COLOR = "#CC6677"
HP0_COLOR = "#BBBBBB"
REF_COLOR = "#333333"
MED_HP1 = "#224488"
MED_HP2 = "#993355"

# Instability values from TSV (pre-computed)
LOCI_DATA = {
    "chr22:11031587-11032148": {
        "chrom": "chr22", "start": 11031587, "end": 11032148,
        "motif": "TC", "motif_len": 2, "ref_size": 561,
        "p21": {"hii_h1": 0.50, "hii_h2": 1.50, "med_h1": 604, "med_h2": 525,
                "n_h1": 22, "n_h2": 23, "ias": 0.67},
        "p41": {"hii_h1": 8.00, "hii_h2": 3.00, "med_h1": 593, "med_h2": 523,
                "n_h1": 15, "n_h2": 16, "ias": 0.63},
    },
    "chr21:10378605-10379219": {
        "chrom": "chr21", "start": 10378605, "end": 10379219,
        "motif": "AG", "motif_len": 2, "ref_size": 614,
        "p21": {"hii_h1": 0.50, "hii_h2": 0.50, "med_h1": 618, "med_h2": 587,
                "n_h1": 13, "n_h2": 18, "ias": 0.00},
        "p41": {"hii_h1": 0.50, "hii_h2": 4.00, "med_h1": 617, "med_h2": 586,
                "n_h1": 15, "n_h2": 21, "ias": 0.88},
    },
    "chr21:9021081-9021291": {
        "chrom": "chr21", "start": 9021081, "end": 9021291,
        "motif": "TG", "motif_len": 2, "ref_size": 210,
        "p21": {"hii_h1": 0.00, "hii_h2": 0.00, "med_h1": 202, "med_h2": 184,
                "n_h1": 23, "n_h2": 58, "ias": 0.00},
        "p41": {"hii_h1": 2.00, "hii_h2": 3.50, "med_h1": 198, "med_h2": 191,
                "n_h1": 32, "n_h2": 67, "ias": 0.43},
    },
}


def get_reads(bam_path, chrom, start, end, motif_len):
    """Extract reads at a locus."""
    bam = pysam.AlignmentFile(bam_path, "rb")
    ref_fasta = pysam.FastaFile(REF)
    try:
        reads = extract_reads_enhanced(
            bam, chrom, start, end,
            motif_len=motif_len,
            ref_fasta=ref_fasta if motif_len < 7 else None,
        )
    finally:
        bam.close()
        ref_fasta.close()
    return reads


def draw_waterfall(ax, reads_info, ref_size, med_h1, med_h2, hii_h1, hii_h2):
    """Draw waterfall on axis with median lines and HII annotations."""
    groups = {0: [], 1: [], 2: []}
    for r in reads_info:
        hp = r.hp if r.hp in (1, 2) else 0
        groups[hp].append(r)
    for hp in groups:
        groups[hp].sort(key=lambda r: r.allele_size)

    ordered = []
    boundaries = []
    for hp in [1, 0, 2]:
        boundaries.append(len(ordered))
        ordered.extend(groups[hp])

    n_reads = len(ordered)
    if n_reads == 0:
        ax.text(0.5, 0.5, "No reads", transform=ax.transAxes, ha="center")
        return

    all_sizes = [r.allele_size for r in ordered]
    size_range = max(all_sizes) - min(all_sizes) if n_reads > 1 else 1.0
    min_bar_half = max(size_range * 0.005, 0.8)

    colors_map = {0: HP0_COLOR, 1: HP1_COLOR, 2: HP2_COLOR}
    for y_idx, read in enumerate(ordered):
        hp = read.hp if read.hp in (1, 2) else 0
        ax.barh(y_idx, width=min_bar_half * 2, left=read.allele_size - min_bar_half,
                height=0.8, color=colors_map[hp], edgecolor="none", alpha=0.85)

    # Reference line
    ax.axvline(ref_size, color=REF_COLOR, linestyle="--", linewidth=1.0, alpha=0.5,
               label=f"Reference ({ref_size} bp)")

    # Section dividers
    for b in boundaries[1:]:
        if 0 < b < n_reads:
            ax.axhline(b - 0.5, color="#cccccc", linestyle="-", linewidth=0.6)

    # Per-haplotype median lines
    if med_h1 is not None:
        ax.axvline(med_h1, color=MED_HP1, linestyle=":", linewidth=2.0, alpha=0.9)
    if med_h2 is not None:
        ax.axvline(med_h2, color=MED_HP2, linestyle=":", linewidth=2.0, alpha=0.9)

    # Legend with HII
    handles = []
    for hp, label_name, color, hii in [(1, "HP1", HP1_COLOR, hii_h1),
                                        (2, "HP2", HP2_COLOR, hii_h2),
                                        (0, "HP0", HP0_COLOR, None)]:
        n = len(groups[hp])
        if n > 0:
            lbl = f"{label_name} (n={n}"
            if hii is not None:
                lbl += f", HII={hii:.1f}"
            lbl += ")"
            handles.append(mpatches.Patch(color=color, label=lbl))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9,
              edgecolor="#cccccc")

    ax.set_ylim(-0.5, n_reads - 0.5)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("Allele size (bp)", fontsize=10)


def plot_single_locus(locus_key, data):
    """Generate a publication-quality 1x2 figure for a single locus."""
    chrom, start, end = data["chrom"], data["start"], data["end"]
    motif, motif_len, ref_size = data["motif"], data["motif_len"], data["ref_size"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for col, passage in enumerate(["p21", "p41"]):
        reads = get_reads(BAMS[passage], chrom, start, end, motif_len)
        pdata = data[passage]

        draw_waterfall(
            axes[col], reads, ref_size,
            med_h1=pdata["med_h1"], med_h2=pdata["med_h2"],
            hii_h1=pdata["hii_h1"], hii_h2=pdata["hii_h2"],
        )

        passage_label = f"Passage {passage[1:]}"
        n_total = pdata["n_h1"] + pdata["n_h2"]
        axes[col].set_title(f"{passage_label}  (n={n_total})", fontsize=12, fontweight="bold")

        # Add IAS annotation
        ias = pdata.get("ias", None)
        if ias is not None:
            axes[col].text(0.98, 0.02, f"IAS = {ias:.2f}", transform=axes[col].transAxes,
                          fontsize=8, ha="right", va="bottom", color="#555555",
                          style="italic")

    # Suptitle
    fig.suptitle(
        f"{locus_key}\n{motif} repeat, ref = {ref_size} bp (long-read only)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Arrow annotation between panels showing drift direction
    fig.text(0.5, 0.01,
             f"20 passages (~40-60 cell divisions)",
             ha="center", fontsize=9, color="#666666", style="italic")

    fig.tight_layout()
    safe = locus_key.replace(":", "_").replace("-", "_")
    out = os.path.join(OUTDIR, f"drift_{safe}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_combined_figure():
    """Generate a combined 2x2 figure with the two best loci."""
    best_loci = [
        "chr22:11031587-11032148",  # HP1 HII 0.5→8.0
        "chr21:10378605-10379219",  # HP2 HII 0.5→4.0
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panel_labels = ["A", "B", "C", "D"]

    for row, locus_key in enumerate(best_loci):
        data = LOCI_DATA[locus_key]
        chrom, start, end = data["chrom"], data["start"], data["end"]
        motif, motif_len, ref_size = data["motif"], data["motif_len"], data["ref_size"]

        for col, passage in enumerate(["p21", "p41"]):
            ax = axes[row, col]
            reads = get_reads(BAMS[passage], chrom, start, end, motif_len)
            pdata = data[passage]

            draw_waterfall(
                ax, reads, ref_size,
                med_h1=pdata["med_h1"], med_h2=pdata["med_h2"],
                hii_h1=pdata["hii_h1"], hii_h2=pdata["hii_h2"],
            )

            label_idx = row * 2 + col
            passage_num = passage[1:]
            title = f"{panel_labels[label_idx]}. Passage {passage_num}"
            if col == 0:
                title += f"  —  {locus_key} ({motif}, {ref_size}bp)"
            ax.set_title(title, fontsize=10, fontweight="bold", loc="left")

            ias = pdata.get("ias", None)
            if ias is not None and ias > 0:
                ax.text(0.98, 0.02, f"IAS = {ias:.2f}", transform=ax.transAxes,
                       fontsize=8, ha="right", va="bottom", color="#555555", style="italic")

    fig.suptitle(
        "Per-Haplotype Tandem Repeat Instability Across Cell Line Passages\n"
        "HG008 Pancreatic Tumor, Haplotagged HiFi Reads",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Arrow-like annotation between columns
    for row in range(2):
        fig.text(0.50, 0.73 - row * 0.47,
                 "20 passages\n(~40-60 divisions)",
                 ha="center", va="center", fontsize=8, color="#888888",
                 style="italic",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#cccccc", alpha=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTDIR, "fig_passage_drift_combined.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("=== Individual locus figures ===")
    for key, data in LOCI_DATA.items():
        print(f"\n--- {key} ---")
        try:
            plot_single_locus(key, data)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n=== Combined figure ===")
    try:
        plot_combined_figure()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
