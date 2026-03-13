"""MosaicTR visualization module.

Publication-quality waterfall plots, allele histograms, and instability
summary panels for haplotype-resolved tandem repeat analysis.

Inspired by TRGT-viz (TRVZ) but designed around MosaicTR's haplotype-aware
instability metrics (HII, IAS).

All plotting functions use matplotlib with lazy imports so that the
mosaictr package does not require matplotlib at import time.
"""

from __future__ import annotations

import logging
from typing import Optional

from .genotype import ReadInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (colorblind-friendly, Tol bright)
# ---------------------------------------------------------------------------

_HP1_COLOR = "#4477AA"  # blue
_HP2_COLOR = "#CC6677"  # rose/red
_HP0_COLOR = "#999999"  # gray
_REF_COLOR = "#333333"  # near-black for reference line
_MEDIAN_HP1 = "#224488"  # darker blue for median line
_MEDIAN_HP2 = "#993355"  # darker red for median line


def _apply_style():
    """Apply a clean publication style. Falls back gracefully."""
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("default")


def _hp_color(hp: int) -> str:
    """Return the color string for a given HP tag value."""
    if hp == 1:
        return _HP1_COLOR
    elif hp == 2:
        return _HP2_COLOR
    return _HP0_COLOR


def _hp_label(hp: int) -> str:
    """Return a human-readable label for a given HP tag value."""
    if hp == 1:
        return "HP1 (haplotype 1)"
    elif hp == 2:
        return "HP2 (haplotype 2)"
    return "HP0 (unphased)"


# ---------------------------------------------------------------------------
# Waterfall plot
# ---------------------------------------------------------------------------

def waterfall_plot(
    reads_info: list[ReadInfo],
    ref_size: float,
    motif_len: int,
    output_path: str,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 6),
    show_units: bool = False,
) -> str:
    """Per-read waterfall plot showing allele sizes colored by haplotype.

    Reads are sorted by allele size within each haplotype group, laid out
    as horizontal bars (HP1 on top, HP0 in the middle, HP2 on the bottom)
    to give an IGV-style view of the repeat length distribution.

    Args:
        reads_info: List of ReadInfo(allele_size, hp, mapq) namedtuples.
            Typically obtained from ``genotype.extract_reads_enhanced()``.
        ref_size: Reference allele size in bp (locus end - start).
        motif_len: Repeat motif length in bp (e.g., 3 for CAG).
        output_path: File path for the saved figure (PNG recommended).
        title: Optional plot title. If None a default is generated.
        figsize: Matplotlib figure size as (width, height) in inches.
        show_units: If True, x-axis is shown in repeat units instead of bp.

    Returns:
        The output_path string (for chaining convenience).

    Raises:
        ValueError: If reads_info is empty.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    _apply_style()

    if not reads_info:
        raise ValueError("reads_info is empty; cannot create waterfall plot.")

    # --- Separate and sort reads by HP group, then by allele size ----------
    groups = {0: [], 1: [], 2: []}
    for r in reads_info:
        hp = r.hp if r.hp in (1, 2) else 0
        groups[hp].append(r)

    for hp in groups:
        groups[hp].sort(key=lambda r: r.allele_size)

    # Layout order: HP1 (top), HP0 (middle), HP2 (bottom)
    ordered_reads: list[ReadInfo] = []
    section_boundaries: list[int] = []  # y-index where each group starts
    group_order = [1, 0, 2]
    for hp in group_order:
        section_boundaries.append(len(ordered_reads))
        ordered_reads.extend(groups[hp])

    n_reads = len(ordered_reads)

    # --- Coordinate transform (optional repeat-unit mode) ------------------
    def _to_x(bp_val: float) -> float:
        if show_units and motif_len >= 1:
            return bp_val / motif_len
        return bp_val

    x_label = "Repeat units" if (show_units and motif_len >= 1) else "Allele size (bp)"

    # --- Build the figure --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Minimum bar width so single-read or identical-size reads are visible
    all_sizes = [r.allele_size for r in ordered_reads]
    size_range = max(all_sizes) - min(all_sizes) if len(all_sizes) > 1 else 1.0
    min_bar_half = max(size_range * 0.003, 0.5 if not show_units else 0.05)

    for y_idx, read in enumerate(ordered_reads):
        x_center = _to_x(read.allele_size)
        color = _hp_color(read.hp if read.hp in (1, 2) else 0)
        ax.barh(
            y_idx,
            width=min_bar_half * 2,
            left=x_center - min_bar_half,
            height=0.8,
            color=color,
            edgecolor="none",
            alpha=0.85,
        )

    # Reference line
    ref_x = _to_x(ref_size)
    ax.axvline(ref_x, color=_REF_COLOR, linestyle="--", linewidth=1.2,
               label=f"Reference ({ref_size:.0f} bp)", zorder=5)

    # Section separators (thin gray lines between HP groups)
    for boundary in section_boundaries[1:]:
        if 0 < boundary < n_reads:
            ax.axhline(boundary - 0.5, color="#cccccc", linestyle="-",
                       linewidth=0.6, zorder=1)

    # --- Legend -------------------------------------------------------------
    legend_handles = []
    for hp in group_order:
        n = len(groups[hp])
        if n > 0:
            patch = mpatches.Patch(
                color=_hp_color(hp),
                label=f"{_hp_label(hp)} (n={n})",
            )
            legend_handles.append(patch)
    # Reference line entry
    legend_handles.append(
        plt.Line2D([0], [0], color=_REF_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Reference ({ref_size:.0f} bp)")
    )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.9)

    # --- Axes and labels ---------------------------------------------------
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Reads (sorted by allele size)", fontsize=11)
    ax.set_ylim(-0.5, n_reads - 0.5)
    ax.set_yticks([])
    ax.invert_yaxis()

    if title is None:
        title = "MosaicTR Waterfall Plot"
    ax.set_title(title, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Waterfall plot saved to %s (%d reads)", output_path, n_reads)
    return output_path


# ---------------------------------------------------------------------------
# Allele histogram
# ---------------------------------------------------------------------------

def allele_histogram(
    reads_info: list[ReadInfo],
    ref_size: float,
    motif_len: int,
    output_path: str,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 5),
    hii_h1: Optional[float] = None,
    hii_h2: Optional[float] = None,
    show_units: bool = False,
) -> str:
    """Stacked allele-size histogram colored by haplotype.

    Bin width is set to ``motif_len`` for STRs (motif_len <= 6 bp) or
    chosen automatically for VNTRs. Per-haplotype weighted medians are
    shown as vertical dashed lines.

    Args:
        reads_info: List of ReadInfo(allele_size, hp, mapq) namedtuples.
        ref_size: Reference allele size in bp.
        motif_len: Repeat motif length in bp.
        output_path: File path for the saved figure.
        title: Optional plot title.
        figsize: Figure size in inches.
        hii_h1: Optional HII value for haplotype 1 (annotated on plot).
        hii_h2: Optional HII value for haplotype 2 (annotated on plot).
        show_units: If True, x-axis is in repeat units.

    Returns:
        The output_path string.

    Raises:
        ValueError: If reads_info is empty.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_style()

    if not reads_info:
        raise ValueError("reads_info is empty; cannot create histogram.")

    # --- Separate by HP ----------------------------------------------------
    groups = {0: [], 1: [], 2: []}
    for r in reads_info:
        hp = r.hp if r.hp in (1, 2) else 0
        groups[hp].append(r.allele_size)

    all_sizes = [r.allele_size for r in reads_info]
    size_min, size_max = min(all_sizes), max(all_sizes)

    # --- Coordinate transform ----------------------------------------------
    def _to_x(bp_val: float) -> float:
        if show_units and motif_len >= 1:
            return bp_val / motif_len
        return bp_val

    x_label = "Repeat units" if (show_units and motif_len >= 1) else "Allele size (bp)"

    # --- Determine bin edges -----------------------------------------------
    if motif_len <= 6:
        # STR: use motif_len as bin width
        bin_width = motif_len if not show_units else 1.0
    else:
        # VNTR: auto bin width
        bin_width = max(1.0, (size_max - size_min) / 30) if size_max > size_min else 1.0
        if show_units and motif_len >= 1:
            bin_width = bin_width / motif_len

    x_min = _to_x(size_min) - bin_width
    x_max = _to_x(size_max) + bin_width
    n_bins = max(int(np.ceil((x_max - x_min) / bin_width)), 1)
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    # --- Build the figure --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Stacked histogram (HP1 bottom, HP0 middle, HP2 top)
    stack_order = [1, 0, 2]
    colors = [_hp_color(hp) for hp in stack_order]
    labels = [f"{_hp_label(hp)} (n={len(groups[hp])})" for hp in stack_order]
    data = [[_to_x(s) for s in groups[hp]] for hp in stack_order]

    # Filter out empty groups for clean legend
    non_empty = [(d, c, l) for d, c, l in zip(data, colors, labels) if len(d) > 0]
    if non_empty:
        stack_data, stack_colors, stack_labels = zip(*non_empty)
        ax.hist(
            list(stack_data),
            bins=bin_edges,
            stacked=True,
            color=list(stack_colors),
            label=list(stack_labels),
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )

    # --- Per-haplotype medians (weighted) ----------------------------------
    _draw_hp_medians(ax, reads_info, motif_len, show_units, hii_h1, hii_h2)

    # --- Reference line ----------------------------------------------------
    ref_x = _to_x(ref_size)
    ax.axvline(ref_x, color=_REF_COLOR, linestyle="--", linewidth=1.2,
               label=f"Reference ({ref_size:.0f} bp)")

    # --- Labels / legend ---------------------------------------------------
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Read count", fontsize=11)

    if title is None:
        title = "MosaicTR Allele Size Distribution"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Allele histogram saved to %s (%d reads)", output_path, len(reads_info))
    return output_path


def _draw_hp_medians(
    ax,
    reads_info: list[ReadInfo],
    motif_len: int,
    show_units: bool,
    hii_h1: Optional[float],
    hii_h2: Optional[float],
) -> None:
    """Draw per-haplotype weighted median lines and optional HII annotations."""
    import numpy as np

    # Lazy import to avoid circular dependency at module level
    from .genotype import _weighted_median

    def _to_x(bp_val: float) -> float:
        if show_units and motif_len >= 1:
            return bp_val / motif_len
        return bp_val

    for hp, color, median_color, hii_val in [
        (1, _HP1_COLOR, _MEDIAN_HP1, hii_h1),
        (2, _HP2_COLOR, _MEDIAN_HP2, hii_h2),
    ]:
        hp_reads = [r for r in reads_info if r.hp == hp]
        if not hp_reads:
            continue
        sizes = np.array([r.allele_size for r in hp_reads])
        weights = np.maximum(np.array([r.mapq for r in hp_reads], dtype=float), 1.0)
        med = _weighted_median(sizes, weights)
        med_x = _to_x(med)

        label = f"HP{hp} median ({med:.1f} bp)"
        if hii_val is not None:
            label += f"  [HII={hii_val:.3f}]"

        ax.axvline(med_x, color=median_color, linestyle=":", linewidth=1.5,
                   alpha=0.9, label=label)


# ---------------------------------------------------------------------------
# Instability summary (3-panel)
# ---------------------------------------------------------------------------

def instability_summary_plot(
    reads_info: list[ReadInfo],
    ref_size: float,
    motif_len: int,
    instability_result: dict,
    output_path: str,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (14, 5),
) -> str:
    """Three-panel instability summary figure.

    Panel A: Waterfall plot (per-read allele sizes by haplotype).
    Panel B: Allele size histogram with per-haplotype medians and HII.
    Panel C: Metric bar chart (HII, IAS per haplotype).

    This is designed as the main visual output for a single locus,
    suitable for supplementary figures or clinical reports.

    Args:
        reads_info: List of ReadInfo namedtuples.
        ref_size: Reference allele size in bp.
        motif_len: Repeat motif length in bp.
        instability_result: Dict returned by
            ``instability.compute_instability()``.  Expected keys include
            ``hii_h1``, ``hii_h2``, ``ias``, ``median_h1``, ``median_h2``, etc.
        output_path: File path for the saved figure.
        title: Optional super-title for the figure.
        figsize: Figure size in inches.

    Returns:
        The output_path string.

    Raises:
        ValueError: If reads_info is empty.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    _apply_style()

    if not reads_info:
        raise ValueError("reads_info is empty; cannot create summary plot.")

    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})

    # -----------------------------------------------------------------------
    # Panel A: Waterfall
    # -----------------------------------------------------------------------
    _waterfall_on_axis(axes[0], reads_info, ref_size, motif_len)
    axes[0].set_title("A. Per-read allele sizes", fontsize=11, fontweight="bold")

    # -----------------------------------------------------------------------
    # Panel B: Histogram
    # -----------------------------------------------------------------------
    hii_h1 = instability_result.get("hii_h1")
    hii_h2 = instability_result.get("hii_h2")
    _histogram_on_axis(axes[1], reads_info, ref_size, motif_len, hii_h1, hii_h2)
    axes[1].set_title("B. Allele size distribution", fontsize=11, fontweight="bold")

    # -----------------------------------------------------------------------
    # Panel C: Metric bar chart
    # -----------------------------------------------------------------------
    _metric_bars_on_axis(axes[2], instability_result)
    axes[2].set_title("C. Instability metrics", fontsize=11, fontweight="bold")

    # -----------------------------------------------------------------------
    # Super title
    # -----------------------------------------------------------------------
    if title is None:
        analysis = instability_result.get("analysis_path", "unknown")
        title = f"MosaicTR Instability Summary (analysis: {analysis})"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Summary plot saved to %s (%d reads)", output_path, len(reads_info))
    return output_path


# ---------------------------------------------------------------------------
# Internal axis-level drawing helpers
# ---------------------------------------------------------------------------

def _waterfall_on_axis(ax, reads_info: list[ReadInfo], ref_size: float,
                       motif_len: int) -> None:
    """Draw a waterfall plot onto an existing matplotlib Axes."""
    import matplotlib.patches as mpatches
    import numpy as np

    groups = {0: [], 1: [], 2: []}
    for r in reads_info:
        hp = r.hp if r.hp in (1, 2) else 0
        groups[hp].append(r)

    for hp in groups:
        groups[hp].sort(key=lambda r: r.allele_size)

    ordered: list[ReadInfo] = []
    boundaries: list[int] = []
    for hp in [1, 0, 2]:
        boundaries.append(len(ordered))
        ordered.extend(groups[hp])

    n_reads = len(ordered)
    if n_reads == 0:
        return

    all_sizes = [r.allele_size for r in ordered]
    size_range = max(all_sizes) - min(all_sizes) if n_reads > 1 else 1.0
    min_bar_half = max(size_range * 0.003, 0.5)

    for y_idx, read in enumerate(ordered):
        color = _hp_color(read.hp if read.hp in (1, 2) else 0)
        ax.barh(
            y_idx,
            width=min_bar_half * 2,
            left=read.allele_size - min_bar_half,
            height=0.8,
            color=color,
            edgecolor="none",
            alpha=0.85,
        )

    ax.axvline(ref_size, color=_REF_COLOR, linestyle="--", linewidth=1.0)

    # Section dividers
    for b in boundaries[1:]:
        if 0 < b < n_reads:
            ax.axhline(b - 0.5, color="#cccccc", linestyle="-", linewidth=0.5)

    # Compact legend
    handles = []
    for hp in [1, 0, 2]:
        n = len(groups[hp])
        if n > 0:
            handles.append(mpatches.Patch(
                color=_hp_color(hp),
                label=f"HP{hp} (n={n})",
            ))
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)

    ax.set_xlabel("Allele size (bp)", fontsize=9)
    ax.set_ylabel("Reads", fontsize=9)
    ax.set_ylim(-0.5, n_reads - 0.5)
    ax.set_yticks([])
    ax.invert_yaxis()


def _histogram_on_axis(ax, reads_info: list[ReadInfo], ref_size: float,
                       motif_len: int, hii_h1: Optional[float],
                       hii_h2: Optional[float]) -> None:
    """Draw a stacked allele histogram onto an existing matplotlib Axes."""
    import numpy as np
    from .genotype import _weighted_median

    groups = {0: [], 1: [], 2: []}
    for r in reads_info:
        hp = r.hp if r.hp in (1, 2) else 0
        groups[hp].append(r.allele_size)

    all_sizes = [r.allele_size for r in reads_info]
    size_min, size_max = min(all_sizes), max(all_sizes)

    if motif_len <= 6:
        bin_width = max(motif_len, 1)
    else:
        bin_width = max(1.0, (size_max - size_min) / 30) if size_max > size_min else 1.0

    x_min = size_min - bin_width
    x_max = size_max + bin_width
    n_bins = max(int(np.ceil((x_max - x_min) / bin_width)), 1)
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    stack_order = [1, 0, 2]
    colors = [_hp_color(hp) for hp in stack_order]
    labels = [f"HP{hp} (n={len(groups[hp])})" for hp in stack_order]
    data = [groups[hp] for hp in stack_order]

    non_empty = [(d, c, l) for d, c, l in zip(data, colors, labels) if len(d) > 0]
    if non_empty:
        stack_data, stack_colors, stack_labels = zip(*non_empty)
        ax.hist(
            list(stack_data),
            bins=bin_edges,
            stacked=True,
            color=list(stack_colors),
            label=list(stack_labels),
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )

    # Per-haplotype median lines
    for hp, median_color, hii_val in [
        (1, _MEDIAN_HP1, hii_h1),
        (2, _MEDIAN_HP2, hii_h2),
    ]:
        hp_reads = [r for r in reads_info if r.hp == hp]
        if not hp_reads:
            continue
        sizes = np.array([r.allele_size for r in hp_reads])
        weights = np.maximum(np.array([r.mapq for r in hp_reads], dtype=float), 1.0)
        med = _weighted_median(sizes, weights)
        label = f"HP{hp} med={med:.1f}"
        if hii_val is not None:
            label += f" [HII={hii_val:.3f}]"
        ax.axvline(med, color=median_color, linestyle=":", linewidth=1.3,
                   alpha=0.9, label=label)

    ax.axvline(ref_size, color=_REF_COLOR, linestyle="--", linewidth=1.0,
               label=f"Ref ({ref_size:.0f} bp)")

    ax.set_xlabel("Allele size (bp)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


def _metric_bars_on_axis(ax, result: dict) -> None:
    """Draw a grouped bar chart of instability metrics onto an Axes.

    Shows pairs of per-haplotype values (h1, h2) for HII, SER, SCR,
    plus the inter-haplotype IAS as a single bar.
    """
    import numpy as np

    # Metric definitions: (display_name, key_h1, key_h2)
    paired_metrics = [
        ("HII", "hii_h1", "hii_h2"),
    ]
    single_metrics = [
        ("IAS", "ias"),
    ]

    # Collect values
    group_labels = []
    vals_h1 = []
    vals_h2 = []
    is_single = []

    for name, k1, k2 in paired_metrics:
        group_labels.append(name)
        vals_h1.append(result.get(k1, 0.0))
        vals_h2.append(result.get(k2, 0.0))
        is_single.append(False)

    for name, key in single_metrics:
        group_labels.append(name)
        vals_h1.append(result.get(key, 0.0))
        vals_h2.append(0.0)
        is_single.append(True)

    n_groups = len(group_labels)
    x = np.arange(n_groups)
    bar_width = 0.35

    # Draw bars
    for i in range(n_groups):
        if is_single[i]:
            ax.bar(x[i], vals_h1[i], bar_width * 2, color="#666666",
                   edgecolor="white", alpha=0.85)
            # Value label
            ax.text(x[i], vals_h1[i] + 0.01, f"{vals_h1[i]:.3f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
        else:
            ax.bar(x[i] - bar_width / 2, vals_h1[i], bar_width,
                   color=_HP1_COLOR, edgecolor="white", alpha=0.85)
            ax.bar(x[i] + bar_width / 2, vals_h2[i], bar_width,
                   color=_HP2_COLOR, edgecolor="white", alpha=0.85)
            # Value labels
            ax.text(x[i] - bar_width / 2, vals_h1[i] + 0.01,
                    f"{vals_h1[i]:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color=_MEDIAN_HP1)
            ax.text(x[i] + bar_width / 2, vals_h2[i] + 0.01,
                    f"{vals_h2[i]:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color=_MEDIAN_HP2)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.set_ylim(bottom=0)

    # Add a small legend
    import matplotlib.patches as mpatches
    ax.legend(
        handles=[
            mpatches.Patch(color=_HP1_COLOR, label="Haplotype 1"),
            mpatches.Patch(color=_HP2_COLOR, label="Haplotype 2"),
            mpatches.Patch(color="#666666", label="Inter-haplotype"),
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
    )

    # Analysis path annotation in lower-left
    analysis = result.get("analysis_path", "")
    if analysis:
        ax.text(0.02, 0.95, f"path: {analysis}",
                transform=ax.transAxes, fontsize=7, va="top",
                color="#555555", style="italic")
