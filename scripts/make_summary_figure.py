"""Generate DeepTR project summary figure as PDF."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

C = {  # colour palette
    "bg":       "#FAFAFA",
    "blue":     "#2C73D2",
    "blue_lt":  "#D6E8FF",
    "orange":   "#E07A3A",
    "orange_lt":"#FDE8D8",
    "green":    "#2CA02C",
    "green_lt": "#D5F0D5",
    "red":      "#D62728",
    "red_lt":   "#FADADA",
    "purple":   "#9467BD",
    "purple_lt":"#E8D8F0",
    "gray":     "#7F7F7F",
    "gray_lt":  "#EEEEEE",
    "dark":     "#2C2C2C",
    "white":    "#FFFFFF",
}

fig = plt.figure(figsize=(16, 22), facecolor=C["white"])

# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
fig.text(0.50, 0.975, "DeepTR: Attention-based Tandem Repeat Genotyping\nfrom Long-Read Sequencing",
         ha="center", va="top", fontsize=18, fontweight="bold", color=C["dark"],
         linespacing=1.4)
fig.text(0.50, 0.945, "Project Summary  —  February 2026",
         ha="center", va="top", fontsize=11, color=C["gray"])

# ══════════════════════════════════════════════════════════════════════════
# PANEL 1 — Problem & Motivation  (top-left)
# ══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_axes([0.04, 0.74, 0.44, 0.19])
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
ax1.axis("off")
ax1.set_title("1. Problem: TR Genotyping Discordance", loc="left", pad=8)

# Background box
ax1.add_patch(FancyBboxPatch((0.2, 0.3), 9.5, 9.2, boxstyle="round,pad=0.3",
              facecolor=C["red_lt"], edgecolor=C["red"], linewidth=1.2, alpha=0.5))

problem_text = (
    "Tandem Repeats (TRs) are the most variable regions in the human genome,\n"
    "yet existing long-read genotypers show poor concordance:\n\n"
    "   TRGT vs LongTR vs Straglr → only 47–86% agreement\n\n"
    "All current tools use heuristic / unsupervised methods:\n"
    "   • TRGT:    Hidden Markov Model (HMM)\n"
    "   • LongTR:  Alignment + heuristic repeat counting\n"
    "   • Straglr: Gaussian Mixture Model (GMM) clustering\n\n"
    "No tool uses supervised deep learning for TR genotyping."
)
ax1.text(0.6, 5.0, problem_text, fontsize=8.5, va="center", ha="left",
         color=C["dark"], family="monospace", linespacing=1.5)

# ══════════════════════════════════════════════════════════════════════════
# PANEL 2 — Our Approach  (top-right)
# ══════════════════════════════════════════════════════════════════════════
ax2 = fig.add_axes([0.52, 0.74, 0.44, 0.19])
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_title("2. Our Approach: Supervised Transformer", loc="left", pad=8)

ax2.add_patch(FancyBboxPatch((0.2, 0.3), 9.5, 9.2, boxstyle="round,pad=0.3",
              facecolor=C["green_lt"], edgecolor=C["green"], linewidth=1.2, alpha=0.5))

approach_text = (
    "DeepTR — first attention-based TR genotyper for long reads\n\n"
    "Key innovations:\n"
    "  1. Per-read CIGAR features (15 dims) → captures all alignment signals\n"
    "  2. Transformer encoder with self-attention over reads\n"
    "     → learns to weight informative reads & cluster alleles\n"
    "  3. Trained on GIAB assembly-based truth (1.6M loci)\n"
    "     → supervised learning, not heuristics\n"
    "  4. Tiny model (106K params) → fast CPU/GPU inference\n\n"
    "Advantage: learns from data what TRGT/Straglr hard-code as rules"
)
ax2.text(0.6, 5.0, approach_text, fontsize=8.5, va="center", ha="left",
         color=C["dark"], family="monospace", linespacing=1.5)

# ══════════════════════════════════════════════════════════════════════════
# PANEL 3 — Architecture Diagram  (middle, full width)
# ══════════════════════════════════════════════════════════════════════════
ax3 = fig.add_axes([0.04, 0.44, 0.92, 0.27])
ax3.set_xlim(0, 20); ax3.set_ylim(0, 8)
ax3.axis("off")
ax3.set_title("3. DeepTR Architecture", loc="left", pad=8)

def draw_box(ax, x, y, w, h, label, sublabel="", color=C["blue"], text_color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                 facecolor=color, edgecolor="none", alpha=0.9))
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=text_color)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.3, sublabel,
                ha="center", va="center", fontsize=7, color=text_color, alpha=0.85)

def draw_arrow(ax, x1, y1, x2, y2, color=C["gray"]):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

# Input
draw_box(ax3, 0.3, 4.5, 2.4, 2.5, "BAM + Loci BED", "HG002 HiFi\n+ adotto catalog", C["gray"], C["white"])

# Feature extraction
draw_arrow(ax3, 2.8, 5.75, 3.8, 5.75)
draw_box(ax3, 3.8, 4.0, 3.0, 3.5, "Feature\nExtraction", "CIGAR walk\n15 per-read features\n8 locus features", C["orange"], C["white"])

# Per-read features detail
ax3.add_patch(FancyBboxPatch((3.9, 0.5), 2.8, 3.0, boxstyle="round,pad=0.1",
              facecolor=C["orange_lt"], edgecolor=C["orange"], linewidth=0.8))
feat_text = (
    "Per-read (K=15):\n"
    " allele_size_bp\n"
    " n_ins/del, total_ins/del\n"
    " max_single_ins/del\n"
    " mapq, softclips L/R\n"
    " flank_match L/R\n"
    " read_len, strand, is_supp"
)
ax3.text(5.3, 2.0, feat_text, ha="center", va="center", fontsize=6.5,
         color=C["dark"], family="monospace", linespacing=1.4)

# Transformer
draw_arrow(ax3, 6.9, 5.75, 8.0, 5.75)
draw_box(ax3, 8.0, 3.8, 3.2, 3.7, "Transformer\nEncoder", "2 layers, 4 heads\nd_model=64\nself-attention", C["blue"], C["white"])

# Attention detail
ax3.add_patch(FancyBboxPatch((8.1, 0.5), 3.0, 2.8, boxstyle="round,pad=0.1",
              facecolor=C["blue_lt"], edgecolor=C["blue"], linewidth=0.8))
attn_text = (
    "Self-attention:\n"
    " Read encoding + pos enc\n"
    " Multi-head attention ×2\n"
    " Mean + Max pooling\n"
    " + Locus context (L=8)\n"
    " → Combined repr (192d)"
)
ax3.text(9.6, 1.9, attn_text, ha="center", va="center", fontsize=6.5,
         color=C["dark"], family="monospace", linespacing=1.4)

# Prediction heads
draw_arrow(ax3, 11.3, 5.75, 12.5, 5.75)
draw_box(ax3, 12.5, 4.0, 3.0, 3.5, "Prediction\nHeads", "Allele sizes (2)\nZygosity (1)\nConfidence (1)", C["green"], C["white"])

# Output
draw_arrow(ax3, 15.6, 5.75, 16.7, 5.75)
draw_box(ax3, 16.7, 4.5, 2.8, 2.5, "Output BED", "allele1, allele2\nHOM/HET\nconfidence", C["purple"], C["white"])

# ══════════════════════════════════════════════════════════════════════════
# PANEL 4 — Training Pipeline  (bottom-left)
# ══════════════════════════════════════════════════════════════════════════
ax4 = fig.add_axes([0.04, 0.15, 0.44, 0.26])
ax4.set_xlim(0, 10); ax4.set_ylim(0, 10)
ax4.axis("off")
ax4.set_title("4. Training Data & Pipeline", loc="left", pad=8)

ax4.add_patch(FancyBboxPatch((0.2, 0.2), 9.5, 9.3, boxstyle="round,pad=0.3",
              facecolor=C["gray_lt"], edgecolor=C["gray"], linewidth=1.0, alpha=0.5))

train_text = (
    "Training data:\n"
    "  • GIAB HG002 Tier1 benchmark v1.0.1\n"
    "  • 1,638,106 loci (101K variants + 1.54M reference)\n"
    "  • Assembly-based haplotype-resolved truth labels\n"
    "  • Matched to adotto TR catalog v1.2 for motif info\n\n"
    "Split (by chromosome):\n"
    "  • Train: chr1–18  (~75%, ~1.2M loci)\n"
    "  • Val:   chr19–20 (~10%)\n"
    "  • Test:  chr21–22, chrX (~15%)\n\n"
    "Training config:\n"
    "  • Loss: SmoothL1 (allele) + BCE (zygosity)\n"
    "  • AdamW, lr=1e-3, cosine annealing\n"
    "  • Batch=512, early stopping (patience=5)\n"
    "  • Augmentation: read subsampling + Gaussian noise\n"
    "  • Hardware: NVIDIA L40S (46GB) via SLURM"
)
ax4.text(0.6, 5.1, train_text, fontsize=7.8, va="center", ha="left",
         color=C["dark"], family="monospace", linespacing=1.45)

# ══════════════════════════════════════════════════════════════════════════
# PANEL 5 — Benchmarking Plan  (bottom-right)
# ══════════════════════════════════════════════════════════════════════════
ax5 = fig.add_axes([0.52, 0.15, 0.44, 0.26])
ax5.set_xlim(0, 10); ax5.set_ylim(0, 10)
ax5.axis("off")
ax5.set_title("5. Evaluation & Benchmarking Plan", loc="left", pad=8)

ax5.add_patch(FancyBboxPatch((0.2, 0.2), 9.5, 9.3, boxstyle="round,pad=0.3",
              facecolor=C["purple_lt"], edgecolor=C["purple"], linewidth=1.0, alpha=0.5))

bench_text = (
    "Comparison targets:\n"
    "  • TRGT      — HMM-based (PacBio gold standard)\n"
    "  • LongTR    — alignment + heuristic\n"
    "  • Straglr   — GMM clustering\n"
    "  • FastStraglr — CIGAR + GMM (our baseline)\n\n"
    "Metrics:\n"
    "  • Per-allele: exact, ±1bp, ±1 motif unit, ±5bp\n"
    "  • Genotype concordance (both alleles correct)\n"
    "  • Zygosity accuracy (HOM/HET)\n"
    "  • R² and MAE for allele size prediction\n\n"
    "Stratification:\n"
    "  • By motif period: homo / di / STR / VNTR\n"
    "  • By repeat length: <100 / 100–500 / 500–1K / >1K bp\n"
    "  • By coverage: <15× / 15–30× / >30×\n"
    "  • By variant type: TP (variant) vs TN (reference)"
)
ax5.text(0.6, 5.1, bench_text, fontsize=7.8, va="center", ha="left",
         color=C["dark"], family="monospace", linespacing=1.45)

# ══════════════════════════════════════════════════════════════════════════
# PANEL 6 — DL Landscape & Novelty  (bottom strip)
# ══════════════════════════════════════════════════════════════════════════
ax6 = fig.add_axes([0.04, 0.02, 0.92, 0.11])
ax6.set_xlim(0, 20); ax6.set_ylim(0, 4)
ax6.axis("off")
ax6.set_title("6. Existing DL Landscape → DeepTR Fills the Gap", loc="left", pad=6)

# Table-like comparison
cols = [
    ("DeepRepeat\n(2022)", "CNN on raw\nONT signal", "ONT only\ndisease loci", C["gray"]),
    ("DeepTRs\n(2023 preprint)", "CNN for TR\nboundary detect", "No\ngenotyping", C["gray"]),
    ("DeepVariant\n(Google)", "Pileup CNN\nSNPs/indels", "F1 drops\n0.97→0.54 in TRs", C["gray"]),
    ("DeepTR\n(this work)", "Transformer on\nCIGAR features", "First supervised\nTR genotyper", C["green"]),
]

for i, (name, method, limitation, col) in enumerate(cols):
    x = 0.5 + i * 5.0
    ax6.add_patch(FancyBboxPatch((x, 0.3), 4.2, 3.2, boxstyle="round,pad=0.15",
                  facecolor=col if col == C["green"] else C["gray_lt"],
                  edgecolor=col, linewidth=1.5, alpha=0.3 if col == C["gray"] else 0.5))
    ax6.text(x + 2.1, 2.9, name, ha="center", va="center", fontsize=8,
             fontweight="bold", color=C["dark"])
    ax6.text(x + 2.1, 1.8, method, ha="center", va="center", fontsize=7,
             color=C["dark"])
    ax6.text(x + 2.1, 0.8, limitation, ha="center", va="center", fontsize=7,
             color=C["red"] if col == C["gray"] else C["green"], fontweight="bold")

# ── Save ──────────────────────────────────────────────────────────────────
outpath = "/qbio/junsoopablo/02_Projects/10_internship/deeptr/output/DeepTR_project_summary.pdf"
fig.savefig(outpath, format="pdf", bbox_inches="tight", facecolor=C["white"])
plt.close(fig)
print(f"Saved: {outpath}")
