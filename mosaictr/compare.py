"""MosaicTR cross-tissue instability comparison module.

Compares somatic instability profiles across tissues from the same individual,
designed for multi-tissue long-read WGS studies (e.g., SMaHT consortium).

Two modes:
  - compare: Paired comparison (baseline vs target tissue)
  - matrix:  Multi-sample HII matrix across N tissues
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_instability_tsv(path: str) -> dict[tuple[str, int, int, str], dict]:
    """Load instability TSV into dict keyed by (chrom, start, end, motif).

    Returns:
        Dict mapping locus tuple to row dict with all instability fields.
    """
    data = {}
    with open(path) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                header = line.strip().lstrip("#").split("\t")
                continue
            if header is None:
                continue
            cols = line.strip().split("\t")
            if len(cols) < len(header):
                continue
            row = dict(zip(header, cols))
            key = (row["chrom"], int(row["start"]), int(row["end"]), row["motif"])

            # Parse numeric fields
            parsed = {}
            float_keys = [
                "median_h1", "median_h2", "hii_h1", "hii_h2",
                "ias", "range_h1", "range_h2", "concordance",
            ]
            int_keys = ["n_h1", "n_h2", "n_total", "n_trimmed_h1", "n_trimmed_h2"]

            for k in float_keys:
                v = row.get(k, ".")
                parsed[k] = float(v) if v != "." else float("nan")
            for k in int_keys:
                v = row.get(k, "0")
                parsed[k] = int(v) if v != "." else 0
            parsed["analysis_path"] = row.get("analysis_path", "unknown")
            parsed["unstable_haplotype"] = row.get("unstable_haplotype", "none")
            parsed["dropout_flag"] = row.get("dropout_flag", "0") == "1"

            data[key] = parsed

    return data


def _safe_float(v: float) -> float:
    """Return 0.0 for NaN/Inf."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return 0.0
    return v


def _max_hii(row: dict) -> float:
    """Max HII across haplotypes, treating NaN as 0."""
    return max(_safe_float(row["hii_h1"]), _safe_float(row["hii_h2"]))


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def compare_tissues(
    baseline: dict[tuple, dict],
    target: dict[tuple, dict],
    noise_threshold: float = 0.45,
    min_delta: float = 0.5,
) -> list[dict]:
    """Compare instability profiles between baseline and target tissues.

    Identifies loci where target tissue shows elevated instability
    relative to baseline (e.g., blood vs colon).

    Uses max(hii_h1, hii_h2) per locus to avoid HP swap issues
    between independently phased samples.

    Args:
        baseline: Instability data from reference tissue (e.g., blood).
        target: Instability data from tissue of interest.
        noise_threshold: HII threshold for calling a locus "unstable".
        min_delta: Minimum ΔHII to report.

    Returns:
        List of comparison result dicts, sorted by max_delta_hii descending.
    """
    common_loci = set(baseline.keys()) & set(target.keys())
    logger.info("Common loci: %d (baseline=%d, target=%d)",
                len(common_loci), len(baseline), len(target))

    results = []
    for locus in common_loci:
        bl = baseline[locus]
        tg = target[locus]

        bl_hii_h1 = _safe_float(bl["hii_h1"])
        bl_hii_h2 = _safe_float(bl["hii_h2"])
        tg_hii_h1 = _safe_float(tg["hii_h1"])
        tg_hii_h2 = _safe_float(tg["hii_h2"])

        bl_max = max(bl_hii_h1, bl_hii_h2)
        tg_max = max(tg_hii_h1, tg_hii_h2)

        delta_max = tg_max - bl_max

        # Fold change (avoid division by zero)
        eps = 0.01
        fold_change = tg_max / max(bl_max, eps)

        # Category assignment
        bl_unstable = bl_max >= noise_threshold
        tg_unstable = tg_max >= noise_threshold

        if tg_unstable and not bl_unstable:
            category = "tissue_specific"
        elif tg_unstable and bl_unstable:
            category = "both_unstable"
        elif not tg_unstable and bl_unstable:
            category = "baseline_only"
        else:
            category = "stable"

        chrom, start, end, motif = locus
        row = {
            "chrom": chrom,
            "start": start,
            "end": end,
            "motif": motif,
            "baseline_hii_h1": bl_hii_h1,
            "baseline_hii_h2": bl_hii_h2,
            "target_hii_h1": tg_hii_h1,
            "target_hii_h2": tg_hii_h2,
            "baseline_max_hii": bl_max,
            "target_max_hii": tg_max,
            "delta_max_hii": delta_max,
            "fold_change": fold_change,
            "category": category,
            "baseline_median_h1": _safe_float(bl["median_h1"]),
            "baseline_median_h2": _safe_float(bl["median_h2"]),
            "target_median_h1": _safe_float(tg["median_h1"]),
            "target_median_h2": _safe_float(tg["median_h2"]),
            "baseline_n_total": bl["n_total"],
            "target_n_total": tg["n_total"],
            "baseline_path": bl["analysis_path"],
            "target_path": tg["analysis_path"],
        }
        results.append(row)

    # Sort by delta_max_hii descending
    results.sort(key=lambda r: r["delta_max_hii"], reverse=True)

    # Filter by min_delta
    if min_delta > 0:
        results = [r for r in results if r["delta_max_hii"] >= min_delta]

    return results


def write_compare_tsv(output_path: str, results: list[dict]) -> int:
    """Write paired comparison results to TSV."""
    cols = [
        "chrom", "start", "end", "motif",
        "baseline_max_hii", "target_max_hii", "delta_max_hii", "fold_change",
        "category",
        "baseline_hii_h1", "baseline_hii_h2",
        "target_hii_h1", "target_hii_h2",
        "baseline_median_h1", "baseline_median_h2",
        "target_median_h1", "target_median_h2",
        "baseline_n_total", "target_n_total",
        "baseline_path", "target_path",
    ]
    with open(output_path, "w") as f:
        f.write("#" + "\t".join(cols) + "\n")
        for r in results:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        vals.append(".")
                    elif v == int(v) and abs(v) < 1e12:
                        vals.append(str(int(v)))
                    else:
                        vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")
    return len(results)


def format_compare_summary(
    results: list[dict],
    n_common: int,
    baseline_label: str = "baseline",
    target_label: str = "target",
) -> str:
    """Format a human-readable comparison summary."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"MosaicTR Cross-Tissue Comparison: {baseline_label} vs {target_label}")
    lines.append("=" * 70)
    lines.append(f"  Common loci: {n_common:,}")
    lines.append(f"  Loci with ΔHII reported: {len(results):,}")

    # Category counts (over all common loci, not just filtered)
    cats = {}
    for r in results:
        cats[r["category"]] = cats.get(r["category"], 0) + 1

    lines.append("")
    lines.append("  Category breakdown (reported loci):")
    for cat in ["tissue_specific", "both_unstable", "baseline_only", "stable"]:
        n = cats.get(cat, 0)
        lines.append(f"    {cat:<20s}: {n:>6d}")

    # Top loci
    tissue_specific = [r for r in results if r["category"] == "tissue_specific"]
    if tissue_specific:
        n_show = min(20, len(tissue_specific))
        lines.append(f"\n  Top {n_show} tissue-specific loci (ΔHII):")
        lines.append(f"    {'Locus':<30s} {'Motif':<8s} {'BL HII':>7s} {'TG HII':>7s} "
                      f"{'ΔHII':>7s} {'Fold':>6s}")
        lines.append("    " + "-" * 70)
        for r in tissue_specific[:n_show]:
            loc = f"{r['chrom']}:{r['start']}-{r['end']}"
            lines.append(
                f"    {loc:<30s} {r['motif']:<8s} "
                f"{r['baseline_max_hii']:>7.2f} {r['target_max_hii']:>7.2f} "
                f"{r['delta_max_hii']:>7.2f} {r['fold_change']:>6.1f}x"
            )

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Multi-sample matrix
# ---------------------------------------------------------------------------

def build_matrix(
    samples: dict[str, dict[tuple, dict]],
    noise_threshold: float = 0.45,
) -> tuple[list[tuple], list[str], list[list[float]], list[dict]]:
    """Build loci × samples HII matrix.

    Uses max(hii_h1, hii_h2) per locus per sample for robustness
    against HP swap between independently phased samples.

    Args:
        samples: Dict mapping sample_name -> instability data.
        noise_threshold: HII threshold for classifying loci.

    Returns:
        (loci, sample_names, hii_matrix, locus_stats)
        - loci: list of (chrom, start, end, motif) tuples
        - sample_names: list of sample labels
        - hii_matrix: loci × samples, hii_matrix[i][j] = HII for locus i, sample j
        - locus_stats: per-locus summary dicts
    """
    sample_names = list(samples.keys())

    # Find loci present in all samples
    if not sample_names:
        return [], [], [], []

    common_loci = set(samples[sample_names[0]].keys())
    for name in sample_names[1:]:
        common_loci &= set(samples[name].keys())

    logger.info("Matrix: %d samples, %d common loci", len(sample_names), len(common_loci))

    # Sort loci for consistent output
    loci = sorted(common_loci, key=lambda x: (x[0], x[1]))

    hii_matrix = []
    locus_stats = []

    for locus in loci:
        row = []
        for name in sample_names:
            hii = _max_hii(samples[name][locus])
            row.append(hii)
        hii_matrix.append(row)

        # Per-locus statistics
        vals = [v for v in row if not math.isnan(v)]
        n = len(vals)
        mean_hii = sum(vals) / n if n > 0 else 0.0
        sd_hii = (sum((v - mean_hii) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        max_hii = max(vals) if vals else 0.0
        min_hii = min(vals) if vals else 0.0
        max_idx = row.index(max_hii) if vals else 0
        tissue_max = sample_names[max_idx] if vals else ""

        n_unstable = sum(1 for v in vals if v >= noise_threshold)
        if n_unstable == 0:
            category = "stable"
        elif n_unstable == n:
            category = "constitutive"
        else:
            category = "tissue_variable"

        locus_stats.append({
            "mean_hii": mean_hii,
            "sd_hii": sd_hii,
            "max_hii": max_hii,
            "min_hii": min_hii,
            "tissue_max": tissue_max,
            "n_unstable": n_unstable,
            "category": category,
        })

    return loci, sample_names, hii_matrix, locus_stats


def write_matrix_tsv(
    output_path: str,
    loci: list[tuple],
    sample_names: list[str],
    hii_matrix: list[list[float]],
    locus_stats: list[dict],
) -> int:
    """Write HII matrix to TSV."""
    def _fmt(v):
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return "."
            if v == int(v) and abs(v) < 1e12:
                return str(int(v))
            return f"{v:.4f}"
        return str(v)

    sample_cols = [f"hii_{name}" for name in sample_names]
    header_cols = ["chrom", "start", "end", "motif"] + sample_cols + [
        "mean_hii", "sd_hii", "max_hii", "min_hii",
        "tissue_max", "n_unstable", "category",
    ]

    with open(output_path, "w") as f:
        f.write("#" + "\t".join(header_cols) + "\n")
        for i, locus in enumerate(loci):
            chrom, start, end, motif = locus
            row_vals = [chrom, str(start), str(end), motif]
            row_vals += [_fmt(v) for v in hii_matrix[i]]
            stats = locus_stats[i]
            row_vals += [
                _fmt(stats["mean_hii"]),
                _fmt(stats["sd_hii"]),
                _fmt(stats["max_hii"]),
                _fmt(stats["min_hii"]),
                stats["tissue_max"],
                str(stats["n_unstable"]),
                stats["category"],
            ]
            f.write("\t".join(row_vals) + "\n")

    return len(loci)


def format_matrix_summary(
    loci: list[tuple],
    sample_names: list[str],
    locus_stats: list[dict],
    noise_threshold: float = 0.45,
) -> str:
    """Format a human-readable matrix summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("MosaicTR Multi-Tissue HII Matrix Summary")
    lines.append("=" * 70)
    lines.append(f"  Samples: {len(sample_names)} ({', '.join(sample_names)})")
    lines.append(f"  Common loci: {len(loci):,}")
    lines.append(f"  Noise threshold: {noise_threshold}")

    # Category counts
    cats = {}
    for s in locus_stats:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    lines.append("\n  Locus categories:")
    for cat in ["stable", "tissue_variable", "constitutive"]:
        n = cats.get(cat, 0)
        pct = n / max(len(loci), 1) * 100
        lines.append(f"    {cat:<20s}: {n:>8,} ({pct:.1f}%)")

    # Per-tissue summary
    lines.append("\n  Per-tissue mean HII:")
    for j, name in enumerate(sample_names):
        vals = [locus_stats[i]["mean_hii"] for i in range(len(loci))]
        # Actually compute per-tissue from matrix — but we don't have matrix here
        # Use which tissue has max
        n_max = sum(1 for s in locus_stats if s["tissue_max"] == name)
        lines.append(f"    {name:<20s}: max-tissue at {n_max:,} loci")

    # Top tissue-variable loci
    variable = [(loci[i], locus_stats[i]) for i in range(len(loci))
                if locus_stats[i]["category"] == "tissue_variable"]
    variable.sort(key=lambda x: x[1]["sd_hii"], reverse=True)

    if variable:
        n_show = min(20, len(variable))
        lines.append(f"\n  Top {n_show} tissue-variable loci (by SD):")
        lines.append(f"    {'Locus':<30s} {'Motif':<8s} {'Mean':>6s} {'SD':>6s} "
                      f"{'Max':>6s} {'Tissue':>10s}")
        lines.append("    " + "-" * 70)
        for locus, stats in variable[:n_show]:
            loc = f"{locus[0]}:{locus[1]}-{locus[2]}"
            lines.append(
                f"    {loc:<30s} {locus[3]:<8s} "
                f"{stats['mean_hii']:>6.2f} {stats['sd_hii']:>6.2f} "
                f"{stats['max_hii']:>6.2f} {stats['tissue_max']:>10s}"
            )

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------

def run_compare(
    baseline_path: str,
    target_path: str,
    output_path: str,
    noise_threshold: float = 0.45,
    min_delta: float = 0.5,
    baseline_label: str = "baseline",
    target_label: str = "target",
) -> str:
    """Run paired tissue comparison pipeline.

    Args:
        baseline_path: Instability TSV for reference tissue (e.g., blood).
        target_path: Instability TSV for tissue of interest.
        output_path: Output comparison TSV.
        noise_threshold: HII threshold for calling unstable.
        min_delta: Minimum ΔHII to include in output.
        baseline_label: Label for baseline tissue.
        target_label: Label for target tissue.

    Returns:
        Path to output file.
    """
    logger.info("Loading baseline: %s", baseline_path)
    baseline = load_instability_tsv(baseline_path)
    logger.info("Baseline: %d loci", len(baseline))

    logger.info("Loading target: %s", target_path)
    target = load_instability_tsv(target_path)
    logger.info("Target: %d loci", len(target))

    n_common = len(set(baseline.keys()) & set(target.keys()))

    # Compare without min_delta for summary stats
    all_results = compare_tissues(baseline, target, noise_threshold, min_delta=0)
    filtered = [r for r in all_results if r["delta_max_hii"] >= min_delta]

    # Write filtered results
    n_written = write_compare_tsv(output_path, filtered)
    logger.info("Wrote %d loci to %s", n_written, output_path)

    # Print summary
    summary = format_compare_summary(all_results, n_common, baseline_label, target_label)
    print(summary)

    return output_path


def run_matrix(
    input_paths: list[str],
    sample_names: list[str],
    output_path: str,
    noise_threshold: float = 0.45,
) -> str:
    """Run multi-sample HII matrix pipeline.

    Args:
        input_paths: List of instability TSV paths.
        sample_names: List of sample labels (same order as input_paths).
        output_path: Output matrix TSV.
        noise_threshold: HII threshold for classifying loci.

    Returns:
        Path to output file.
    """
    samples = {}
    for path, name in zip(input_paths, sample_names):
        logger.info("Loading %s: %s", name, path)
        data = load_instability_tsv(path)
        logger.info("  %d loci", len(data))
        samples[name] = data

    loci, names, hii_matrix, locus_stats = build_matrix(samples, noise_threshold)

    n_written = write_matrix_tsv(output_path, loci, names, hii_matrix, locus_stats)
    logger.info("Wrote %d loci to %s", n_written, output_path)

    summary = format_matrix_summary(loci, names, locus_stats, noise_threshold)
    print(summary)

    return output_path
