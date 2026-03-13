#!/usr/bin/env python3
"""Generate combined simulation figure for MosaicTR manuscript.

4-panel figure (2x2):
  (A) HII dose-response linearity
  (B) ROC curve for stable vs unstable classification
  (C) Detection power vs coverage
  (D) Disease-specific ECB patterns (SER vs SCR)

Usage:
  python scripts/plot_simulation_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Data (from simulation_report.txt, n_reps=30)
# ---------------------------------------------------------------------------

# (A) HII dose-response
hii_targets = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
hii_measured = [0.0635, 0.3958, 0.7917, 1.5833, 3.9583, 7.9165, 15.8331]
hii_std = [0.0171, 0.1345, 0.2690, 0.5380, 1.3450, 2.6899, 5.3799]

# (B) ROC data — recompute from the simulation
# We'll run the ROC analysis inline for the curve data
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mosaictr.genotype import ReadInfo
from mosaictr.instability import compute_instability


def _ri(size, hp, mapq=60):
    return ReadInfo(allele_size=size, hp=hp, mapq=mapq)


def _stable_reads(median, n, hp, rng):
    sizes = rng.normal(median, 0.5, n)
    return [_ri(float(s), hp) for s in sizes]


def _unstable_reads(median, n, hp, motif_len, target_hii, rng):
    b = target_hii * motif_len / np.log(2)
    sizes = rng.laplace(median, b, n)
    return [_ri(float(s), hp) for s in sizes]


def compute_roc_data():
    motif_len = 3
    n_per_hp = 25
    n_stable = 300
    n_unstable = 300

    labels = []
    max_hiis = []

    for i in range(n_stable):
        rng = np.random.default_rng(4000 + i)
        hp1 = _stable_reads(60.0, n_per_hp, 1, rng)
        hp2 = _stable_reads(90.0, n_per_hp, 2, rng)
        r = compute_instability(hp1 + hp2, 60.0, motif_len)
        if r is None:
            continue
        max_hiis.append(max(r["hii_h1"], r["hii_h2"]))
        labels.append(0)

    for i in range(n_unstable):
        rng = np.random.default_rng(5000 + i)
        target = rng.uniform(1.5, 10.0)
        hp1 = _stable_reads(60.0, n_per_hp, 1, rng)
        hp2 = _unstable_reads(120.0, n_per_hp, 2, motif_len, target, rng)
        r = compute_instability(hp1 + hp2, 60.0, motif_len)
        if r is None:
            continue
        max_hiis.append(max(r["hii_h1"], r["hii_h2"]))
        labels.append(1)

    labels = np.array(labels)
    max_hiis = np.array(max_hiis)

    thresholds = np.linspace(0, 5.0, 200)
    fprs, tprs = [], []
    for thr in thresholds:
        pred = (max_hiis > thr).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        tn = np.sum((pred == 0) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    sort_idx = np.argsort(fprs)
    fprs_s = np.array(fprs)[sort_idx]
    tprs_s = np.array(tprs)[sort_idx]
    auc = float(np.trapezoid(tprs_s, fprs_s))

    # Default threshold metrics
    pred_d = (max_hiis > 0.45).astype(int)
    tp_d = int(np.sum((pred_d == 1) & (labels == 1)))
    fp_d = int(np.sum((pred_d == 1) & (labels == 0)))
    tn_d = int(np.sum((pred_d == 0) & (labels == 0)))
    fn_d = int(np.sum((pred_d == 0) & (labels == 1)))
    sens = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0
    spec = tn_d / (tn_d + fp_d) if (tn_d + fp_d) > 0 else 0

    return fprs, tprs, auc, sens, spec


# (C) Coverage sweep
cov_x = [5, 10, 15, 20, 30, 40, 60, 80]
cov_detect = [60.0, 83.3, 100.0, 96.7, 100.0, 96.7, 100.0, 100.0]
cov_hii_mean = [1.2585, 1.1852, 1.2792, 1.3770, 1.4298, 1.2188, 1.2701, 1.3362]
cov_hii_std = [1.3549, 0.6839, 0.5129, 0.6978, 0.5706, 0.3701, 0.2632, 0.1899]

# (D) Disease ECB patterns
disease_names = ["HD-like\n(expansion)", "DM1-like\n(pure exp.)", "Contraction\ndominant", "Balanced\nMSI"]
disease_sers = [0.316, 0.267, 0.250, 0.377]
disease_scrs = [0.261, 0.177, 0.305, 0.342]
disease_ecb_target = [0.80, 1.00, -0.80, 0.00]
disease_ecb_measured = [0.0918, 0.2428, -0.0708, 0.0495]

# ---------------------------------------------------------------------------
# Color palette (professional, colorblind-safe)
# ---------------------------------------------------------------------------
C_BLUE = "#3274A1"
C_GREEN = "#3A923A"
C_RED = "#E1812C"
C_PURPLE = "#9372B2"
C_TEAL = "#3B9B9B"
C_ORANGE = "#C85200"
C_GRAY = "#666666"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def main():
    print("Computing ROC data...")
    roc_fprs, roc_tprs, roc_auc, roc_sens, roc_spec = compute_roc_data()

    fig = plt.figure(figsize=(7.2, 6.5))  # ~183mm wide (Bioinformatics double column)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35,
                           left=0.09, right=0.97, top=0.95, bottom=0.08)

    label_kw = dict(fontsize=14, fontweight="bold", va="top", ha="left")

    # ── (A) HII Dose-Response ──────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.errorbar(hii_targets, hii_measured, yerr=hii_std, fmt="o-",
                  color=C_GREEN, markersize=5, linewidth=1.5, capsize=3,
                  elinewidth=0.8, markeredgecolor="white", markeredgewidth=0.5,
                  zorder=3)
    ax_a.plot([0, 22], [0, 22], "--", color=C_GRAY, alpha=0.5, linewidth=1,
              label="y = x")

    # Linear fit line
    slope = np.polyfit(hii_targets[1:], hii_measured[1:], 1)[0]
    x_fit = np.linspace(0, 21, 100)
    ax_a.plot(x_fit, slope * x_fit, ":", color=C_GREEN, alpha=0.6, linewidth=1,
              label=f"Slope = {slope:.2f}")

    ax_a.set_xlabel("Input HII", fontsize=10)
    ax_a.set_ylabel("Measured HII", fontsize=10)
    ax_a.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_a.set_xlim(-0.5, 22)
    ax_a.set_ylim(-0.5, 22)
    ax_a.tick_params(labelsize=8.5)
    ax_a.grid(True, alpha=0.15, linewidth=0.5)

    # R² inset
    ax_a.text(0.97, 0.15, f"R$^2$ = {1.000:.3f}\nMonotonic",
              transform=ax_a.transAxes, fontsize=8.5, ha="right",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor=C_GREEN, alpha=0.9))

    ax_a.text(-0.15, 1.05, "A", transform=ax_a.transAxes, **label_kw)

    # ── (B) ROC Curve ──────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(roc_fprs, roc_tprs, color=C_BLUE, linewidth=2, zorder=3)
    ax_b.plot([0, 1], [0, 1], "--", color=C_GRAY, alpha=0.5, linewidth=1)

    # Default threshold marker
    ax_b.plot(1 - roc_spec, roc_sens, "*", color=C_RED, markersize=14,
              zorder=4, markeredgecolor="white", markeredgewidth=0.5)
    ax_b.annotate(
        f"$\\tau$=0.45\nSens={roc_sens:.0%}\nSpec={roc_spec:.0%}",
        xy=(1 - roc_spec, roc_sens),
        xytext=(0.35, 0.55), fontsize=8,
        arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=C_RED, alpha=0.9),
    )

    ax_b.set_xlabel("False Positive Rate", fontsize=10)
    ax_b.set_ylabel("True Positive Rate", fontsize=10)
    ax_b.set_xlim(-0.02, 1.02)
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.tick_params(labelsize=8.5)
    ax_b.grid(True, alpha=0.15, linewidth=0.5)

    ax_b.text(0.97, 0.15, f"AUC = {roc_auc:.3f}\nn = 600 loci",
              transform=ax_b.transAxes, fontsize=8.5, ha="right",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor=C_BLUE, alpha=0.9))

    ax_b.text(-0.15, 1.05, "B", transform=ax_b.transAxes, **label_kw)

    # ── (C) Coverage vs Detection ──────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])

    # Detection rate (left y-axis)
    ln1 = ax_c.plot(cov_x, cov_detect, "o-", color=C_RED, markersize=5,
                    linewidth=1.5, markeredgecolor="white", markeredgewidth=0.5,
                    label="Detection rate", zorder=3)
    ax_c.axhline(80, color=C_GRAY, linestyle="--", alpha=0.4, linewidth=0.8)
    ax_c.set_xlabel("Coverage (reads/haplotype)", fontsize=10)
    ax_c.set_ylabel("Detection Rate (%)", fontsize=10, color=C_RED)
    ax_c.set_ylim(0, 108)
    ax_c.tick_params(axis="y", labelcolor=C_RED, labelsize=8.5)
    ax_c.tick_params(axis="x", labelsize=8.5)

    # HII estimation (right y-axis)
    ax_c2 = ax_c.twinx()
    ln2 = ax_c2.errorbar(cov_x, cov_hii_mean, yerr=cov_hii_std, fmt="s--",
                         color=C_PURPLE, markersize=4, linewidth=1, capsize=2,
                         elinewidth=0.6, alpha=0.8, label="HII (mean$\\pm$SD)",
                         zorder=2)
    ax_c2.axhline(0.45, color=C_GRAY, linestyle=":", alpha=0.4, linewidth=0.8)
    ax_c2.set_ylabel("Measured HII", fontsize=10, color=C_PURPLE)
    ax_c2.tick_params(axis="y", labelcolor=C_PURPLE, labelsize=8.5)
    ax_c2.set_ylim(-0.2, 3.0)

    # Combined legend
    lines = ln1 + [ln2]
    labels = [l.get_label() for l in lines]
    ax_c.legend(lines, labels, fontsize=7.5, loc="center right", framealpha=0.9)

    ax_c.grid(True, alpha=0.15, linewidth=0.5)

    ax_c.text(0.97, 0.95, "Input HII = 1.5\n$\\tau$ = 0.45",
              transform=ax_c.transAxes, fontsize=8, ha="right", va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor=C_RED, alpha=0.9))

    ax_c.text(-0.15, 1.05, "C", transform=ax_c.transAxes, **label_kw)

    # ── (D) Disease ECB Patterns ───────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])

    x = np.arange(len(disease_names))
    width = 0.32

    bars_ser = ax_d.bar(x - width / 2, disease_sers, width,
                        color=C_GREEN, edgecolor="white", linewidth=0.5,
                        label="SER (expansion)", zorder=3)
    bars_scr = ax_d.bar(x + width / 2, disease_scrs, width,
                        color=C_ORANGE, edgecolor="white", linewidth=0.5,
                        label="SCR (contraction)", zorder=3)

    # Add arrows showing direction
    for i in range(len(disease_names)):
        diff = disease_sers[i] - disease_scrs[i]
        y_top = max(disease_sers[i], disease_scrs[i]) + 0.02
        if abs(diff) > 0.02:
            direction = "+" if diff > 0 else "-"
            color = C_GREEN if diff > 0 else C_ORANGE
            ax_d.text(x[i], y_top, f"ECB {direction}",
                      ha="center", va="bottom", fontsize=7.5,
                      fontweight="bold", color=color)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(disease_names, fontsize=8)
    ax_d.set_ylabel("Ratio", fontsize=10)
    ax_d.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_d.set_ylim(0, 0.48)
    ax_d.tick_params(labelsize=8.5)
    ax_d.grid(True, alpha=0.15, linewidth=0.5, axis="y")

    ax_d.text(-0.15, 1.05, "D", transform=ax_d.transAxes, **label_kw)

    # ── Save ───────────────────────────────────────────────────────────────
    outdir = Path("output/instability/simulation")
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / "fig_simulation_combined.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Also save PDF for journal submission
    out_pdf = outdir / "fig_simulation_combined.pdf"
    fig2 = plt.figure(figsize=(7.2, 6.5))
    gs2 = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35,
                            left=0.09, right=0.97, top=0.95, bottom=0.08)

    # Re-draw for PDF (matplotlib closes fig, need fresh one)
    # ... just save both formats from same figure
    print(f"(Re-run with PDF backend for {out_pdf} if needed)")


if __name__ == "__main__":
    main()
