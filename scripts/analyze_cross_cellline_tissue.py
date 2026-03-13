#!/usr/bin/env python3
"""Cross-cell-line and cross-tissue TR instability comparison.

Scenario A: Different cell lines — HG002 (LCL) vs HG008-T (tumor) vs HG003/HG004 (LCL)
Scenario B: Same person, different tissues — HG008 normal pancreas vs normal duodenum vs tumor

Usage:
    python scripts/analyze_cross_cellline_tissue.py \
        --output-dir output/cross_comparison \
        --scenario all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VAULT = "/vault/external-datasets/2026"

# ---------------------------------------------------------------------------
# Sample definitions
# ---------------------------------------------------------------------------

# Scenario A: Cross-cell-line
CELLLINE_SAMPLES = {
    "HG002_LCL": {
        "bam": f"{VAULT}/HG002_PacBio-HiFi-Revio-48x_BAM_GRCh38/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.bam",
        "label": "HG002 LCL (Ashkenazi male, 48x Revio, HP-tagged)",
        "cell_type": "LCL",
        "ethnicity": "Ashkenazi",
        "existing_tsv": "output/instability/hg002_genomewide.tsv",
    },
    "HG003_LCL": {
        "bam": f"{VAULT}/HG003_PacBio-HiFi-Revio_BAM_GRCh38/GRCh38.m84039_241002_000337_s3.hifi_reads.bc2020.bam",
        "label": "HG003 LCL (Ashkenazi father, Revio, no HP)",
        "cell_type": "LCL",
        "ethnicity": "Ashkenazi",
        "existing_tsv": "output/instability/hg003_genomewide.tsv",
    },
    "HG004_LCL": {
        "bam": f"{VAULT}/HG004_PacBio-HiFi-Revio_BAM_GRCh38/GRCh38.m84039_241002_020632_s4.hifi_reads.bc2021.bam",
        "label": "HG004 LCL (Ashkenazi mother, Revio, no HP)",
        "cell_type": "LCL",
        "ethnicity": "Ashkenazi",
        "existing_tsv": "output/instability/hg004_genomewide.tsv",
    },
    "HG008T_tumor": {
        "bam": f"{VAULT}/HG008_PacBio-HiFi-Revio_GRCh38-GIABv3_TumorNormal/GRCh38_GIABv3.HG008-T.Revio.hifi_reads.bam",
        "label": "HG008-T Tumor p23 (pancreatic cancer, 124x SPRQ)",
        "cell_type": "Tumor",
        "ethnicity": "Ashkenazi",
        "existing_tsv": "output/passage_drift/instability_p23.tsv",
    },
}

# Scenario B: Cross-tissue (same person = HG008)
TISSUE_SAMPLES = {
    "HG008_pancreas": {
        "bam": f"{VAULT}/HG008_PacBio-HiFi-Revio_GRCh38-GIABv3_TumorNormal/GRCh38_GIABv3.HG008-N-P.Revio.hifi_reads.bam",
        "label": "HG008 Normal Pancreas (38x SPRQ)",
        "tissue": "pancreas",
        "existing_tsv": "output/passage_drift/instability_normal.tsv",
    },
    "HG008_duodenum": {
        "bam": f"{VAULT}/HG008_PacBio-HiFi-Revio_GRCh38-GIABv3_TumorNormal/HG008-N-D_PacBio-HiFi-Revio_20240313_68x_GRCh38-GIABv3.bam",
        "label": "HG008 Normal Duodenum (68x Revio)",
        "tissue": "duodenum",
    },
    "HG008_tumor": {
        "bam": f"{VAULT}/HG008_PacBio-HiFi-Revio_GRCh38-GIABv3_TumorNormal/GRCh38_GIABv3.HG008-T.Revio.hifi_reads.bam",
        "label": "HG008 Tumor p23 (pancreatic cancer, 124x SPRQ)",
        "tissue": "tumor",
        "existing_tsv": "output/passage_drift/instability_p23.tsv",
    },
}

TEST_LOCI = "output/v3_comparison/test_loci.bed"


def run_instability(sample_key: str, sample_info: dict, loci_path: str,
                    output_dir: str, threads: int = 1) -> str:
    """Run MosaicTR instability for a single sample, reusing existing if available."""
    # Check for existing TSV first
    existing = sample_info.get("existing_tsv")
    if existing and os.path.exists(existing):
        logger.info("Reusing existing: %s -> %s", sample_key, existing)
        return existing

    from mosaictr.instability import run_instability as _run

    out_path = os.path.join(output_dir, f"instability_{sample_key}.tsv")

    if os.path.exists(out_path):
        logger.info("Skipping %s (already exists: %s)", sample_key, out_path)
        return out_path

    logger.info("Running instability: %s (%s)", sample_key, sample_info["label"])
    logger.info("  BAM: %s", sample_info["bam"])

    t0 = time.time()
    _run(
        bam_path=sample_info["bam"],
        loci_bed_path=loci_path,
        output_path=out_path,
        nprocs=threads,
        skip_hp_check=True,
        noise_threshold=0.45,
    )
    elapsed = time.time() - t0
    logger.info("  Done in %.0fs: %s", elapsed, out_path)
    return out_path


def run_scenario_comparisons(output_dir: str, tsv_paths: dict[str, str],
                             scenario_name: str, pairs: list[tuple[str, str]]):
    """Run pairwise comparisons and matrix for a scenario."""
    from mosaictr.compare import run_compare, run_matrix

    for baseline_key, target_key in pairs:
        if baseline_key not in tsv_paths or target_key not in tsv_paths:
            continue
        out = os.path.join(output_dir, f"compare_{baseline_key}_vs_{target_key}.tsv")
        if not os.path.exists(out):
            logger.info("Comparing: %s vs %s", baseline_key, target_key)
            run_compare(
                baseline_path=tsv_paths[baseline_key],
                target_path=tsv_paths[target_key],
                output_path=out,
                baseline_label=baseline_key,
                target_label=target_key,
                min_delta=0.5,
            )

    # Matrix
    available = [(k, v) for k, v in tsv_paths.items() if os.path.exists(v)]
    if len(available) >= 2:
        out = os.path.join(output_dir, f"matrix_{scenario_name}.tsv")
        if not os.path.exists(out):
            logger.info("Building matrix: %s", [k for k, _ in available])
            run_matrix(
                input_paths=[v for _, v in available],
                sample_names=[k for k, _ in available],
                output_path=out,
            )


def generate_summary(output_dir: str, tsv_paths: dict[str, str],
                     scenario_name: str, sample_defs: dict):
    """Generate a text summary."""
    from mosaictr.compare import (
        _max_hii,
        build_matrix,
        compare_tissues,
        load_instability_tsv,
    )

    lines = []
    lines.append("=" * 75)
    lines.append(f"Cross-Comparison Analysis: {scenario_name}")
    lines.append("=" * 75)

    data = {}
    for key, path in tsv_paths.items():
        if os.path.exists(path):
            d = load_instability_tsv(path)
            data[key] = d
            label = sample_defs.get(key, {}).get("label", key)
            lines.append(f"  {key}: {len(d):,} loci ({label})")

    if len(data) < 2:
        lines.append("\n  ERROR: Need at least 2 samples.")
        summary = "\n".join(lines)
        print(summary)
        return

    # Per-sample stats
    lines.append("\n--- Per-Sample Instability Summary ---")
    for key, d in data.items():
        hiis = [_max_hii(row) for row in d.values()]
        mean_hii = sum(hiis) / len(hiis) if hiis else 0
        n_unstable = sum(1 for h in hiis if h >= 0.45)

        # Analysis path distribution
        paths = {}
        for row in d.values():
            p = row.get("analysis_path", "unknown")
            paths[p] = paths.get(p, 0) + 1
        hp_pct = paths.get("hp-tagged", 0) / max(len(d), 1) * 100

        lines.append(f"  {key:<20s}: mean HII={mean_hii:.4f}, "
                      f"unstable={n_unstable:,} ({n_unstable/len(hiis)*100:.1f}%), "
                      f"HP-tagged={hp_pct:.0f}%")

    # All pairwise comparisons
    keys = list(data.keys())
    lines.append("\n--- Pairwise Comparisons ---")
    lines.append(f"  {'Pair':<45s} {'Overlap':>8s} {'TissSpec':>8s} {'Both':>8s} {'BaseOnly':>8s} {'Stable':>8s}")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            results = compare_tissues(data[k1], data[k2], min_delta=0)
            cats = {}
            for r in results:
                cats[r["category"]] = cats.get(r["category"], 0) + 1
            overlap = len(set(data[k1].keys()) & set(data[k2].keys()))
            pair_name = f"{k1} vs {k2}"
            lines.append(f"  {pair_name:<45s} {overlap:>8,} "
                          f"{cats.get('tissue_specific', 0):>8,} "
                          f"{cats.get('both_unstable', 0):>8,} "
                          f"{cats.get('baseline_only', 0):>8,} "
                          f"{cats.get('stable', 0):>8,}")

    # Matrix
    if len(data) >= 3:
        lines.append("\n--- Multi-Sample Matrix ---")
        loci, names, matrix, stats = build_matrix(data)
        cats = {}
        for s in stats:
            cats[s["category"]] = cats.get(s["category"], 0) + 1
        lines.append(f"  Common loci: {len(loci):,}")
        for cat in ["stable", "tissue_variable", "constitutive"]:
            n = cats.get(cat, 0)
            pct = n / max(len(loci), 1) * 100
            lines.append(f"  {cat:<20s}: {n:>8,} ({pct:.1f}%)")

        # Top variable loci
        variable = [(loci[i], stats[i]) for i in range(len(loci))
                     if stats[i]["category"] == "tissue_variable"]
        variable.sort(key=lambda x: x[1]["sd_hii"], reverse=True)
        if variable:
            lines.append(f"\n  Top 10 most variable loci (by SD of max HII):")
            hdr = f"    {'Locus':<30s} {'Motif':<8s} {'MaxTissue':<15s} {'MaxHII':>7s} {'SD':>7s}"
            # Add per-sample HII headers
            for name in names:
                short = name[:10]
                hdr += f" {short:>10s}"
            lines.append(hdr)
            for loc, st in variable[:10]:
                loc_str = f"{loc[0]}:{loc[1]}-{loc[2]}"
                row_str = (f"    {loc_str:<30s} {loc[3]:<8s} "
                            f"{st['tissue_max']:<15s} {st['max_hii']:>7.2f} {st['sd_hii']:>7.2f}")
                idx = loci.index(loc)
                for val in matrix[idx]:
                    row_str += f" {val:>10.2f}"
                lines.append(row_str)

    lines.append("")
    summary = "\n".join(lines)
    print(summary)

    summary_path = os.path.join(output_dir, f"summary_{scenario_name}.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info("Summary saved to: %s", summary_path)


def run_cellline_scenario(output_dir: str, loci_path: str, threads: int):
    """Scenario A: Cross-cell-line comparison."""
    logger.info("=== Scenario A: Cross-Cell-Line Comparison ===")
    subdir = os.path.join(output_dir, "cellline")
    os.makedirs(subdir, exist_ok=True)

    tsv_paths = {}
    for key, info in CELLLINE_SAMPLES.items():
        if not os.path.exists(info["bam"]):
            logger.warning("MISSING BAM: %s -> %s", key, info["bam"])
            # Still check existing TSV
            existing = info.get("existing_tsv")
            if existing and os.path.exists(existing):
                tsv_paths[key] = existing
                logger.info("Using existing TSV: %s -> %s", key, existing)
            continue
        path = run_instability(key, info, loci_path, subdir, threads)
        tsv_paths[key] = path

    if len(tsv_paths) < 2:
        logger.error("Need at least 2 samples. Available: %s", list(tsv_paths.keys()))
        return

    pairs = [
        ("HG002_LCL", "HG008T_tumor"),
        ("HG002_LCL", "HG003_LCL"),
        ("HG002_LCL", "HG004_LCL"),
        ("HG003_LCL", "HG008T_tumor"),
    ]
    run_scenario_comparisons(subdir, tsv_paths, "cellline", pairs)
    generate_summary(subdir, tsv_paths, "cellline", CELLLINE_SAMPLES)


def run_tissue_scenario(output_dir: str, loci_path: str, threads: int):
    """Scenario B: Cross-tissue comparison (HG008)."""
    logger.info("=== Scenario B: Cross-Tissue Comparison (HG008) ===")
    subdir = os.path.join(output_dir, "tissue")
    os.makedirs(subdir, exist_ok=True)

    tsv_paths = {}
    for key, info in TISSUE_SAMPLES.items():
        if not os.path.exists(info["bam"]):
            logger.warning("MISSING BAM: %s -> %s", key, info["bam"])
            existing = info.get("existing_tsv")
            if existing and os.path.exists(existing):
                tsv_paths[key] = existing
                logger.info("Using existing TSV: %s -> %s", key, existing)
            continue
        path = run_instability(key, info, loci_path, subdir, threads)
        tsv_paths[key] = path

    if len(tsv_paths) < 2:
        logger.error("Need at least 2 samples. Available: %s", list(tsv_paths.keys()))
        return

    pairs = [
        ("HG008_pancreas", "HG008_duodenum"),
        ("HG008_pancreas", "HG008_tumor"),
        ("HG008_duodenum", "HG008_tumor"),
    ]
    run_scenario_comparisons(subdir, tsv_paths, "tissue", pairs)
    generate_summary(subdir, tsv_paths, "tissue", TISSUE_SAMPLES)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-cell-line and cross-tissue TR instability comparison"
    )
    parser.add_argument("--output-dir", default="output/cross_comparison")
    parser.add_argument("--loci", default=TEST_LOCI)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--scenario", choices=["cellline", "tissue", "all"],
                        default="all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scenario in ("cellline", "all"):
        run_cellline_scenario(args.output_dir, args.loci, args.threads)

    if args.scenario in ("tissue", "all"):
        run_tissue_scenario(args.output_dir, args.loci, args.threads)


if __name__ == "__main__":
    main()
