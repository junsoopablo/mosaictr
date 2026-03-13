#!/usr/bin/env python3
"""Generate publication figures for MosaicTR Application Note.

Figure 1 (main, multi-panel):
  (a) Tool accuracy bar chart (Exact, <=1bp, GenConc) — 3 tools
  (b) Motif period vs MAE grouped bar
  (c) Confidence threshold vs accuracy curve (MosaicTR only)

Figure 2 (supplementary):
  (a) Mendelian inheritance rate comparison
  (b) Allele delta vs accuracy
  (c) Repeat length vs accuracy line chart

Usage:
  python scripts/generate_figures.py \
    --mosaictr output/genome_wide/v4_hg002_genome_wide.bed \
    --output-dir output/genome_wide/figures/
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.benchmark import load_predictions, _motif_period_bin, _repeat_length_bin
from mosaictr.utils import load_adotto_catalog, load_tier1_bed
from scripts.benchmark_genome_wide import (
    match_preds_to_truth,
    match_tool_to_truth,
    parse_longtr_vcf,
    parse_trgt_vcf,
    compute_mosaictr_metrics,
    compute_tool_metrics,
    mosaictr_motif_bin,
    mosaictr_length_bin,
    tool_motif_bin,
    tool_length_bin,
    stratify_pairs,
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

# Publication style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
rcParams["font.size"] = 9
rcParams["axes.linewidth"] = 0.8
rcParams["xtick.major.width"] = 0.8
rcParams["ytick.major.width"] = 0.8

COLORS = {
    "MosaicTR": "#2196F3",
    "LongTR": "#FF9800",
    "TRGT": "#4CAF50",
}


def fig1_accuracy_bars(ax, ht_m, lt_m, trgt_m=None):
    """Panel (a): Overall accuracy bar chart."""
    metrics = ["exact", "w1bp", "geno_conc", "zyg_acc"]
    labels = ["Exact\nmatch", "Within\n1bp", "Geno.\nconcord.", "Zygosity\naccuracy"]
    x = np.arange(len(metrics))

    tools = [
        ("MosaicTR", ht_m, COLORS["MosaicTR"]),
        ("LongTR", lt_m, COLORS["LongTR"]),
    ]
    n_tools = len(tools)
    width = 0.3

    for i, (name, m, color) in enumerate(tools):
        vals = [m[k] * 100 for k in metrics]
        offset = (i - (n_tools - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(80, 101)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(a) Overall accuracy", fontsize=10, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig1_motif_mae(ax, ht_pairs, lt_pairs, trgt_pairs=None):
    """Panel (b): MAE by motif period grouped bar."""
    motif_order = ["homopolymer", "dinucleotide", "STR_3bp", "STR_4bp",
                   "STR_5bp", "STR_6bp", "VNTR_7+"]
    short_labels = ["Homo", "Di", "3bp", "4bp", "5bp", "6bp", "VNTR"]

    ht_by = stratify_pairs(ht_pairs, mosaictr_motif_bin)
    lt_by = stratify_pairs(lt_pairs, tool_motif_bin)

    x = np.arange(len(motif_order))
    tool_list = [
        ("MosaicTR", ht_by, compute_mosaictr_metrics, COLORS["MosaicTR"]),
        ("LongTR", lt_by, compute_tool_metrics, COLORS["LongTR"]),
    ]
    n_tools = len(tool_list)
    width = 0.3

    for i, (name, by_dict, compute_fn, color) in enumerate(tool_list):
        vals = []
        for mb in motif_order:
            sub = by_dict.get(mb, [])
            m = compute_fn(sub) if sub else None
            vals.append(m["mae"] if m else 0)
        offset = (i - (n_tools - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=name, color=color,
               edgecolor="white", linewidth=0.5)

    ax.set_ylabel("MAE (bp)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8, rotation=0)
    ax.legend(fontsize=7)
    ax.set_title("(b) MAE by motif period", fontsize=10, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)


def fig1_confidence_curve(ax, ht_pairs):
    """Panel (c): Confidence threshold vs accuracy curve."""
    thresholds = np.arange(0.0, 1.01, 0.05)
    geno_concs = []
    zyg_accs = []
    exacts = []
    pct_kept = []

    for thresh in thresholds:
        sub = [(p, t) for p, t in ht_pairs if p.confidence >= thresh]
        if len(sub) < 10:
            geno_concs.append(np.nan)
            zyg_accs.append(np.nan)
            exacts.append(np.nan)
            pct_kept.append(0)
            continue
        m = compute_mosaictr_metrics(sub)
        geno_concs.append(m["geno_conc"] * 100)
        zyg_accs.append(m["zyg_acc"] * 100)
        exacts.append(m["exact"] * 100)
        pct_kept.append(len(sub) / len(ht_pairs) * 100)

    ax.plot(thresholds, geno_concs, "o-", color=COLORS["MosaicTR"],
            markersize=3, linewidth=1.5, label="Genotype conc.")
    ax.plot(thresholds, zyg_accs, "s-", color="#E91E63",
            markersize=3, linewidth=1.5, label="Zygosity acc.")
    ax.plot(thresholds, exacts, "^-", color="#9C27B0",
            markersize=3, linewidth=1.5, label="Exact match")

    # Secondary axis for % loci retained
    ax2 = ax.twinx()
    ax2.fill_between(thresholds, pct_kept, alpha=0.15, color="gray")
    ax2.set_ylabel("Loci retained (%)", fontsize=8, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
    ax2.set_ylim(0, 105)

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(85, 101)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(c) Confidence filtering", fontsize=10, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)


def fig2_mendelian(ax, mendelian_data=None):
    """Panel (a): Mendelian inheritance rate comparison."""
    # Published values + placeholder for MosaicTR
    tools = ["LongTR", "TRGT", "STRkit", "MosaicTR"]
    strict = [86.0, None, 97.85, None]
    off_by_one = [None, 98.38, 99.08, None]

    if mendelian_data:
        strict[3] = mendelian_data.get("strict", None)
        off_by_one[3] = mendelian_data.get("within_1motif", None)

    x = np.arange(len(tools))
    width = 0.35

    # Strict
    strict_vals = [v if v is not None else 0 for v in strict]
    strict_mask = [v is not None for v in strict]
    bars1 = ax.bar(x[strict_mask] - width / 2, [strict_vals[i] for i in range(len(tools)) if strict_mask[i]],
                   width, label="Strict", color="#2196F3", edgecolor="white")

    # Off-by-one
    obo_vals = [v if v is not None else 0 for v in off_by_one]
    obo_mask = [v is not None for v in off_by_one]
    bars2 = ax.bar(x[obo_mask] + width / 2, [obo_vals[i] for i in range(len(tools)) if obo_mask[i]],
                   width, label="Off-by-one-unit", color="#FF9800", edgecolor="white")

    ax.set_ylabel("Mendelian consistency (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(tools, fontsize=8)
    ax.set_ylim(80, 101)
    ax.legend(fontsize=7)
    ax.set_title("(a) Mendelian inheritance", fontsize=10, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig2_delta_accuracy(ax, ht_pairs, lt_pairs, trgt_pairs=None):
    """Panel (b): Allele delta vs genotype concordance."""
    delta_bins = [
        ("0bp", 0.0, 0.5),
        ("1-5bp", 0.5, 5.5),
        ("6-20bp", 5.5, 20.5),
        ("21-50bp", 20.5, 50.5),
        ("51-100bp", 50.5, 100.5),
        (">100bp", 100.5, float("inf")),
    ]
    bin_labels = [b[0] for b in delta_bins]

    def _max_delta(truth):
        return max(abs(truth.true_allele1_diff), abs(truth.true_allele2_diff))

    def _bin_delta(d):
        ad = abs(d)
        for name, lo, hi in delta_bins:
            if lo <= ad < hi:
                return name
        return ">100bp"

    for tool_name, pairs, compute_fn, color, is_ht in [
        ("MosaicTR", ht_pairs, compute_mosaictr_metrics, COLORS["MosaicTR"], True),
        ("LongTR", lt_pairs, compute_tool_metrics, COLORS["LongTR"], False),
    ]:
        vals = []
        for bl in bin_labels:
            if is_ht:
                sub = [(p, t) for p, t in pairs if _bin_delta(_max_delta(t)) == bl]
            else:
                sub = [entry for entry in pairs if _bin_delta(_max_delta(entry[2])) == bl]
            m = compute_fn(sub) if sub else None
            vals.append(m["geno_conc"] * 100 if m else np.nan)
        ax.plot(range(len(bin_labels)), vals, "o-", color=color,
                label=tool_name, markersize=4, linewidth=1.5)

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=7, rotation=30)
    ax.set_ylabel("Genotype concordance (%)")
    ax.set_xlabel("Max |allele - ref|")
    ax.legend(fontsize=7)
    ax.set_title("(b) Accuracy by allele delta", fontsize=10, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig2_length_accuracy(ax, ht_pairs, lt_pairs, trgt_pairs=None):
    """Panel (c): Repeat length vs genotype concordance."""
    len_bins = ["<100bp", "100-500bp", "500-1000bp", ">1000bp"]

    for tool_name, pairs, compute_fn, color, bin_fn in [
        ("MosaicTR", ht_pairs, compute_mosaictr_metrics, COLORS["MosaicTR"], mosaictr_length_bin),
        ("LongTR", lt_pairs, compute_tool_metrics, COLORS["LongTR"], tool_length_bin),
    ]:
        by_len = stratify_pairs(pairs, bin_fn)
        vals = []
        for lb in len_bins:
            sub = by_len.get(lb, [])
            m = compute_fn(sub) if sub else None
            vals.append(m["geno_conc"] * 100 if m else np.nan)
        ax.plot(range(len(len_bins)), vals, "o-", color=color,
                label=tool_name, markersize=4, linewidth=1.5)

    ax.set_xticks(range(len(len_bins)))
    ax.set_xticklabels(len_bins, fontsize=8)
    ax.set_ylabel("Genotype concordance (%)")
    ax.set_xlabel("Repeat length")
    ax.legend(fontsize=7)
    ax.set_title("(c) Accuracy by repeat length", fontsize=10, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--mosaictr", required=True, help="MosaicTR v4 BED")
    parser.add_argument("--longtr-vcf", default=LONGTR_VCF)
    parser.add_argument("--trgt-vcf", default=TRGT_VCF)
    parser.add_argument("--truth", default=TRUTH_BED)
    parser.add_argument("--catalog", default=CATALOG_BED)
    parser.add_argument("--output-dir", required=True, help="Output directory for figures")
    parser.add_argument("--chroms", default=None, help="Comma-separated chromosomes")
    parser.add_argument("--mendelian-strict", type=float, default=None,
                        help="MosaicTR strict Mendelian rate (%%)")
    parser.add_argument("--mendelian-1motif", type=float, default=None,
                        help="MosaicTR ±1motif Mendelian rate (%%)")
    args = parser.parse_args()

    chrom_set = set(args.chroms.split(",")) if args.chroms else None

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    tier1 = load_tier1_bed(args.truth, chroms=chrom_set)
    catalog = load_adotto_catalog(args.catalog, chroms=chrom_set)
    ht_preds = load_predictions(args.mosaictr)
    longtr = parse_longtr_vcf(args.longtr_vcf, chroms=chrom_set)
    trgt = parse_trgt_vcf(args.trgt_vcf, chroms=chrom_set)

    ht_pairs = match_preds_to_truth(ht_preds, tier1, catalog)
    lt_pairs = match_tool_to_truth(longtr, tier1, catalog)
    trgt_pairs = match_tool_to_truth(trgt, tier1, catalog)

    ht_m = compute_mosaictr_metrics(ht_pairs)
    lt_m = compute_tool_metrics(lt_pairs)
    trgt_m = compute_tool_metrics(trgt_pairs)

    logger.info("Matched: MosaicTR=%d, LongTR=%d, TRGT=%d",
                len(ht_pairs), len(lt_pairs), len(trgt_pairs))

    # ── Figure 1 (main) ─────────────────────────────────────────────
    logger.info("Generating Figure 1...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.subplots_adjust(wspace=0.4, left=0.06, right=0.94, top=0.88, bottom=0.15)

    fig1_accuracy_bars(axes[0], ht_m, lt_m, trgt_m)
    fig1_motif_mae(axes[1], ht_pairs, lt_pairs, trgt_pairs)
    fig1_confidence_curve(axes[2], ht_pairs)

    fig.savefig(outdir / "figure1.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "figure1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 1 saved.")

    # ── Figure 2 (supplementary) ──────────────────────────────────
    logger.info("Generating Figure 2...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.subplots_adjust(wspace=0.4, left=0.06, right=0.94, top=0.88, bottom=0.15)

    mendelian_data = None
    if args.mendelian_strict is not None:
        mendelian_data = {"strict": args.mendelian_strict}
        if args.mendelian_1motif is not None:
            mendelian_data["within_1motif"] = args.mendelian_1motif

    fig2_mendelian(axes[0], mendelian_data)
    fig2_delta_accuracy(axes[1], ht_pairs, lt_pairs, trgt_pairs)
    fig2_length_accuracy(axes[2], ht_pairs, lt_pairs, trgt_pairs)

    fig.savefig(outdir / "figure2.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "figure2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 2 saved.")

    # ── Summary table as figure ──────────────────────────────────
    logger.info("Generating summary table figure...")
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")

    cols = ["Tool", "n", "Exact%", "≤1bp", "MAE", "Zyg%", "GenConc"]
    data = []
    for name, m in [("MosaicTR v4", ht_m), ("LongTR", lt_m), ("TRGT", trgt_m)]:
        data.append([
            name, f"{m['n']:,d}",
            f"{m['exact']*100:.1f}%", f"{m['w1bp']*100:.1f}%",
            f"{m['mae']:.2f}", f"{m['zyg_acc']*100:.1f}%",
            f"{m['geno_conc']*100:.1f}%",
        ])

    table = ax.table(cellText=data, colLabels=cols, loc="center",
                     cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(cols)):
        table[0, j].set_facecolor("#E3F2FD")
        table[0, j].set_text_props(fontweight="bold")

    fig.savefig(outdir / "table_summary.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "table_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("All figures saved to %s", outdir)


if __name__ == "__main__":
    main()
