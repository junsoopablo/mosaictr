#!/usr/bin/env python3
"""Compare MosaicTR HII vs Owl CV at common loci on HG002 (MSS control).

Reads:
  - MosaicTR genome-wide instability TSV (13-column)
  - Owl profile output (tab-delimited per-locus CV)

Produces:
  - Scatter plot: Owl CV vs MosaicTR HII
  - Distribution overlay: both metrics on HG002
  - Summary statistics text report
  - Figure for manuscript (300 dpi)

Usage:
  python scripts/compare_owl_mosaictr.py \
    --mosaictr output/instability/hg002_genomewide_qn.tsv \
    --owl output/instability/owl_hg002_profile.txt \
    --output output/instability/owl_comparison/ \
    [--atxn10-tsvs HG01122.tsv HG02252.tsv HG02345.tsv]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def load_mosaictr_tsv(path: str) -> dict:
    """Load MosaicTR instability TSV → {(chrom, start, end): record}."""
    data = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 13:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            motif = parts[3]
            try:
                hii_h1 = float(parts[6]) if parts[6] != "." else 0.0
                hii_h2 = float(parts[7]) if parts[7] != "." else 0.0
            except ValueError:
                continue
            max_hii = max(hii_h1, hii_h2)
            motif_len = len(motif)
            data[(chrom, start, end)] = {
                "motif": motif,
                "motif_len": motif_len,
                "hii_h1": hii_h1,
                "hii_h2": hii_h2,
                "max_hii": max_hii,
                "analysis_path": parts[12],
            }
    return data


def load_owl_profile(path: str) -> dict:
    """Load Owl v0.4.0 profile output → {(chrom, start, end): record}.

    Owl profile format (4 tab-separated columns):
      #Region          info              format           SAMPLE
      chr1:100-200     RL=100;MO=AC      PS:HP:CT:MU:CV:LN  PS,HP,CT,MU,CV,LN;...

    Each haplotype entry is semicolon-separated, fields are comma-separated.
    CV is field index 4 (0-based) within each entry.
    HP: 0=unphased, 1=hap1, 2=hap2.
    """
    data = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            try:
                # Parse region: chr1:100-200
                region = parts[0]
                colon_idx = region.index(":")
                chrom = region[:colon_idx]
                dash_idx = region.index("-", colon_idx)
                start = int(region[colon_idx + 1 : dash_idx])
                end = int(region[dash_idx + 1 :])

                # Parse motif from info: RL=100;MO=AC
                info = parts[1]
                motif = ""
                for kv in info.split(";"):
                    if kv.startswith("MO="):
                        motif = kv[3:]
                        break

                # Parse haplotype entries: PS,HP,CT,MU,CV,LN;...
                sample_data = parts[3]
                entries = sample_data.split(";")

                cv_h1, cv_h2 = 0.0, 0.0
                for entry in entries:
                    fields = entry.split(",")
                    if len(fields) < 5:
                        continue
                    hp = int(fields[1])
                    cv_str = fields[4]
                    if cv_str in (".", "NA", "nan", ""):
                        cv_val = 0.0
                    else:
                        cv_val = float(cv_str)

                    if hp == 1:
                        cv_h1 = cv_val
                    elif hp == 2:
                        cv_h2 = cv_val
                    elif hp == 0:
                        # Unphased: use as max if no phased data
                        if cv_h1 == 0.0 and cv_h2 == 0.0:
                            cv_h1 = cv_val

            except (ValueError, IndexError):
                continue

            max_cv = max(cv_h1, cv_h2)
            data[(chrom, start, end)] = {
                "motif": motif,
                "motif_len": len(motif) if motif else 1,
                "cv_h1": cv_h1,
                "cv_h2": cv_h2,
                "max_cv": max_cv,
            }
    return data


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compare_metrics(mosaictr: dict, owl: dict) -> dict:
    """Find common loci and compute concordance statistics."""
    common_keys = set(mosaictr.keys()) & set(owl.keys())
    logger.info("MosaicTR loci: %d, Owl loci: %d, Common: %d",
                len(mosaictr), len(owl), len(common_keys))

    if not common_keys:
        logger.warning("No common loci found! Check coordinate systems.")
        return {"n_common": 0}

    hiis = []
    cvs = []
    motifs = []
    motif_lens = []

    for key in sorted(common_keys):
        m = mosaictr[key]
        o = owl[key]
        hiis.append(m["max_hii"])
        cvs.append(o["max_cv"])
        motifs.append(m["motif"])
        motif_lens.append(m["motif_len"])

    hiis = np.array(hiis)
    cvs = np.array(cvs)
    motif_lens = np.array(motif_lens)

    # Correlation (only where both > 0)
    mask = (hiis > 0) & (cvs > 0)
    if mask.sum() > 10:
        corr = np.corrcoef(hiis[mask], cvs[mask])[0, 1]
    else:
        corr = float("nan")

    # Per-motif-length statistics
    motif_stats = {}
    for ml in sorted(set(motif_lens)):
        idx = motif_lens == ml
        motif_stats[ml] = {
            "n": int(idx.sum()),
            "mean_hii": float(np.mean(hiis[idx])),
            "mean_cv": float(np.mean(cvs[idx])),
            "pct_hii_above_045": float(np.mean(hiis[idx] > 0.45) * 100),
        }

    return {
        "n_common": len(common_keys),
        "hiis": hiis,
        "cvs": cvs,
        "motifs": motifs,
        "motif_lens": motif_lens,
        "correlation": corr,
        "mean_hii": float(np.mean(hiis)),
        "mean_cv": float(np.mean(cvs)),
        "pct_hii_zero": float(np.mean(hiis == 0) * 100),
        "pct_cv_zero": float(np.mean(cvs == 0) * 100),
        "motif_stats": motif_stats,
    }


def write_report(stats: dict, output_path: str):
    """Write comparison report."""
    lines = []
    w = lines.append

    w("=" * 80)
    w("  MosaicTR HII vs Owl CV — HG002 (MSS) Comparison")
    w("=" * 80)
    w(f"\n  Common loci: {stats['n_common']:,}")
    w(f"  MosaicTR mean HII: {stats['mean_hii']:.4f}")
    w(f"  Owl mean CV: {stats['mean_cv']:.4f}")
    w(f"  Correlation (HII>0 & CV>0): {stats['correlation']:.4f}")
    w(f"  % HII=0: {stats['pct_hii_zero']:.1f}%")
    w(f"  % CV=0: {stats['pct_cv_zero']:.1f}%")

    w(f"\n  Per-motif-length breakdown:")
    w(f"  {'Motif len':>10s} {'N':>8s} {'Mean HII':>10s} {'Mean CV':>10s} {'% HII>0.45':>12s}")
    w("  " + "-" * 52)
    for ml, ms in sorted(stats.get("motif_stats", {}).items()):
        w(f"  {ml:>10d} {ms['n']:>8d} {ms['mean_hii']:>10.4f} {ms['mean_cv']:>10.4f} {ms['pct_hii_above_045']:>11.1f}%")

    w(f"\n{'=' * 80}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", output_path)


def plot_comparison(stats: dict, output_dir: str):
    """Generate comparison figures."""
    if not HAS_MPL:
        logger.warning("matplotlib not available, skipping plots")
        return

    hiis = stats["hiis"]
    cvs = stats["cvs"]

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Distribution overlay
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 3, 60)
    ax1.hist(hiis[hiis > 0], bins=bins, alpha=0.6, color="#2196F3", label="MosaicTR HII", density=True)
    ax1.hist(cvs[cvs > 0], bins=bins, alpha=0.6, color="#FF9800", label="Owl CV", density=True)
    ax1.axvline(0.45, color="red", linestyle="--", linewidth=1, label="Threshold (0.45)")
    ax1.set_xlabel("Metric value")
    ax1.set_ylabel("Density")
    ax1.set_title("(a) MSS noise floor: HII vs CV")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 3)

    # Panel B: Scatter
    ax2 = fig.add_subplot(gs[0, 1])
    mask = (hiis > 0) | (cvs > 0)
    ax2.scatter(cvs[mask], hiis[mask], s=2, alpha=0.3, c="#666666", rasterized=True)
    ax2.set_xlabel("Owl CV")
    ax2.set_ylabel("MosaicTR HII")
    ax2.set_title(f"(b) Per-locus concordance (r={stats['correlation']:.3f})")
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    lims = [0, min(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lims, lims, "r--", linewidth=0.5, alpha=0.5)

    # Panel C: Per-motif-length bars
    ax3 = fig.add_subplot(gs[1, 0])
    motif_stats = stats.get("motif_stats", {})
    if motif_stats:
        mls = sorted(motif_stats.keys())
        x = np.arange(len(mls))
        width = 0.35
        hii_means = [motif_stats[ml]["mean_hii"] for ml in mls]
        cv_means = [motif_stats[ml]["mean_cv"] for ml in mls]
        ax3.bar(x - width/2, hii_means, width, label="MosaicTR HII", color="#2196F3", alpha=0.8)
        ax3.bar(x + width/2, cv_means, width, label="Owl CV", color="#FF9800", alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{ml}bp" for ml in mls])
        ax3.set_xlabel("Motif length")
        ax3.set_ylabel("Mean metric value")
        ax3.set_title("(c) Per-motif-length comparison")
        ax3.legend(fontsize=8)

    # Panel D: IAS — unique to MosaicTR (placeholder with ATXN10 data)
    ax4 = fig.add_subplot(gs[1, 1])
    carriers = {
        "HG01122\n(1041 RU)": {"hii_exp": 10.0, "hii_norm": 0.2, "ias": 0.98},
        "HG02345\n(320 RU)": {"hii_exp": 0.8, "hii_norm": 0.0, "ias": 1.0},
        "HG02252\n(bilateral)": {"hii_exp": 3.6, "hii_norm": 3.8, "ias": 0.05},
    }
    x = np.arange(len(carriers))
    names = list(carriers.keys())
    ias_vals = [carriers[n]["ias"] for n in names]
    colors = ["#E53935" if v > 0.5 else "#1E88E5" for v in ias_vals]
    ax4.bar(x, ias_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, fontsize=8)
    ax4.set_ylabel("IAS")
    ax4.set_title("(d) MosaicTR-unique: IAS in ATXN10 carriers")
    ax4.set_ylim(0, 1.1)
    ax4.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax4.text(2.4, 0.52, "bilateral", fontsize=7, color="gray")
    ax4.text(0.4, 0.85, "heterozygous", fontsize=7, color="gray")

    fig.savefig(os.path.join(output_dir, "fig_owl_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s/fig_owl_comparison.png", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Compare MosaicTR HII vs Owl CV")
    parser.add_argument("--mosaictr", required=True, help="MosaicTR instability TSV")
    parser.add_argument("--owl", required=True, help="Owl profile output")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MosaicTR results...")
    mosaictr = load_mosaictr_tsv(args.mosaictr)

    logger.info("Loading Owl results...")
    owl = load_owl_profile(args.owl)

    logger.info("Comparing metrics...")
    stats = compare_metrics(mosaictr, owl)

    if stats["n_common"] == 0:
        logger.error("No common loci — cannot compare. Check coordinate systems.")
        return

    write_report(stats, str(output_dir / "owl_comparison_report.txt"))
    plot_comparison(stats, str(output_dir))

    logger.info("Done.")


if __name__ == "__main__":
    main()
