#!/usr/bin/env python3
"""HG008 cell line passage TR drift analysis — haplotagged BAMs.

Same analysis as analyze_passage_drift.py but using whatshap-haplotagged BAMs
so all 4 samples go through the hp-tagged analysis path for fair comparison.

Usage:
    python scripts/analyze_passage_drift_haplotagged.py \
        --output-dir output/passage_drift_hp \
        --loci output/v3_comparison/test_loci.bed \
        --threads 1
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

# ---------------------------------------------------------------------------
# Data paths — haplotagged BAMs
# ---------------------------------------------------------------------------

VAULT = "/vault/external-datasets/2026"
HP_DIR = f"{VAULT}/HG008_haplotagged"

SAMPLES = {
    "normal": {
        "bam": f"{HP_DIR}/HG008_normal_haplotagged.bam",
        "label": "Normal (pancreas)",
        "passage": "normal",
    },
    "p21": {
        "bam": f"{HP_DIR}/HG008_p21_haplotagged.bam",
        "label": "Tumor p21",
        "passage": "p21",
    },
    "p23": {
        "bam": f"{HP_DIR}/HG008_p23_haplotagged.bam",
        "label": "Tumor p23 (SPRQ)",
        "passage": "p23",
    },
    "p41": {
        "bam": f"{HP_DIR}/HG008_p41_haplotagged.bam",
        "label": "Tumor p41",
        "passage": "p41",
    },
}

TEST_LOCI = "output/v3_comparison/test_loci.bed"  # 108K subset (chr21/22/X)


def run_instability(sample_key: str, loci_path: str, output_dir: str,
                    threads: int = 1) -> str:
    """Run MosaicTR instability for a single sample."""
    from mosaictr.instability import run_instability as _run

    info = SAMPLES[sample_key]
    out_path = os.path.join(output_dir, f"instability_{sample_key}.tsv")

    if os.path.exists(out_path):
        logger.info("Skipping %s (already exists: %s)", sample_key, out_path)
        return out_path

    logger.info("Running instability: %s (%s)", sample_key, info["label"])
    logger.info("  BAM: %s", info["bam"])

    t0 = time.time()
    _run(
        bam_path=info["bam"],
        loci_bed_path=loci_path,
        output_path=out_path,
        nprocs=threads,
        noise_threshold=0.45,
    )
    elapsed = time.time() - t0
    logger.info("  Done in %.0fs: %s", elapsed, out_path)
    return out_path


def run_comparisons(output_dir: str, tsv_paths: dict[str, str]):
    """Run all pairwise comparisons and matrix."""
    from mosaictr.compare import run_compare, run_matrix

    # Pairwise: normal vs each tumor passage
    for passage in ["p21", "p23", "p41"]:
        if passage not in tsv_paths:
            continue
        out = os.path.join(output_dir, f"compare_normal_vs_{passage}.tsv")
        if not os.path.exists(out):
            logger.info("Comparing: normal vs %s", passage)
            run_compare(
                baseline_path=tsv_paths["normal"],
                target_path=tsv_paths[passage],
                output_path=out,
                baseline_label="normal",
                target_label=passage,
                min_delta=0.5,
            )

    # Pairwise: p21 vs p41 (passage drift)
    if "p21" in tsv_paths and "p41" in tsv_paths:
        out = os.path.join(output_dir, "compare_p21_vs_p41.tsv")
        if not os.path.exists(out):
            logger.info("Comparing: p21 vs p41 (passage drift)")
            run_compare(
                baseline_path=tsv_paths["p21"],
                target_path=tsv_paths["p41"],
                output_path=out,
                baseline_label="p21",
                target_label="p41",
                min_delta=0.5,
            )

    # Matrix: all samples
    available = [(k, v) for k, v in tsv_paths.items() if os.path.exists(v)]
    if len(available) >= 2:
        out = os.path.join(output_dir, "matrix_all_passages.tsv")
        if not os.path.exists(out):
            logger.info("Building matrix: %s", [k for k, _ in available])
            run_matrix(
                input_paths=[v for _, v in available],
                sample_names=[k for k, _ in available],
                output_path=out,
            )


def generate_summary(output_dir: str, tsv_paths: dict[str, str]):
    """Generate a text summary of the analysis."""
    from mosaictr.compare import (
        build_matrix,
        compare_tissues,
        load_instability_tsv,
    )

    lines = []
    lines.append("=" * 75)
    lines.append("HG008 Cell Line Passage TR Drift Analysis (HAPLOTAGGED)")
    lines.append("=" * 75)

    # Load all data
    data = {}
    for key, path in tsv_paths.items():
        if os.path.exists(path):
            d = load_instability_tsv(path)
            data[key] = d
            lines.append(f"  {key}: {len(d):,} loci loaded")

    if len(data) < 2:
        lines.append("\n  ERROR: Need at least 2 samples for comparison.")
        summary = "\n".join(lines)
        print(summary)
        return

    # Per-sample stats
    lines.append("\n--- Per-Sample Instability Summary ---")
    for key, d in data.items():
        from mosaictr.compare import _max_hii
        hiis = [_max_hii(row) for row in d.values()]
        mean_hii = sum(hiis) / len(hiis) if hiis else 0
        n_unstable = sum(1 for h in hiis if h >= 0.45)
        n_zero = sum(1 for h in hiis if h == 0)

        # Count analysis paths
        paths = {}
        for row in d.values():
            p = row.get("analysis_path", "unknown")
            paths[p] = paths.get(p, 0) + 1
        path_str = ", ".join(f"{k}: {v}" for k, v in sorted(paths.items()))

        lines.append(f"  {key:<10s}: mean HII={mean_hii:.4f}, "
                      f"unstable (>=0.45)={n_unstable:,} ({n_unstable/len(hiis)*100:.1f}%), "
                      f"HII=0: {n_zero/len(hiis)*100:.1f}%")
        lines.append(f"             analysis paths: {path_str}")

    # Normal vs Tumor
    if "normal" in data and "p23" in data:
        lines.append("\n--- Normal vs Tumor (p23) ---")
        results = compare_tissues(data["normal"], data["p23"], min_delta=0)
        cats = {}
        for r in results:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        for cat in ["tissue_specific", "both_unstable", "baseline_only", "stable"]:
            n = cats.get(cat, 0)
            lines.append(f"  {cat:<20s}: {n:>8,}")
        # Top tissue-specific
        ts = [r for r in results if r["category"] == "tissue_specific"]
        if ts:
            lines.append(f"\n  Top 10 tumor-specific loci:")
            lines.append(f"    {'Locus':<30s} {'Motif':<8s} {'NL HII':>7s} {'TU HII':>7s} {'dHII':>7s}")
            for r in ts[:10]:
                loc = f"{r['chrom']}:{r['start']}-{r['end']}"
                lines.append(f"    {loc:<30s} {r['motif']:<8s} "
                              f"{r['baseline_max_hii']:>7.2f} {r['target_max_hii']:>7.2f} "
                              f"{r['delta_max_hii']:>7.2f}")

    # Passage drift: p21 vs p41
    if "p21" in data and "p41" in data:
        lines.append("\n--- Passage Drift: p21 vs p41 (20 passages) ---")
        results = compare_tissues(data["p21"], data["p41"], min_delta=0)
        cats = {}
        for r in results:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        for cat in ["tissue_specific", "both_unstable", "baseline_only", "stable"]:
            n = cats.get(cat, 0)
            lines.append(f"  {cat:<20s}: {n:>8,}")

        # Drift loci
        drift = [r for r in results if r["delta_max_hii"] >= 0.5]
        lines.append(f"\n  Loci with dHII >= 0.5: {len(drift):,}")

        if drift:
            # Motif length distribution of drifted loci
            mono = sum(1 for r in drift if len(r["motif"]) == 1)
            di = sum(1 for r in drift if len(r["motif"]) == 2)
            tri = sum(1 for r in drift if len(r["motif"]) == 3)
            longer = sum(1 for r in drift if len(r["motif"]) > 3)
            lines.append(f"  Motif distribution of drifted loci:")
            lines.append(f"    Mono (1bp): {mono}, Di (2bp): {di}, "
                          f"Tri (3bp): {tri}, >3bp: {longer}")

            lines.append(f"\n  Top 10 passage-drifted loci:")
            lines.append(f"    {'Locus':<30s} {'Motif':<8s} {'p21 HII':>7s} {'p41 HII':>7s} {'dHII':>7s}")
            for r in drift[:10]:
                loc = f"{r['chrom']}:{r['start']}-{r['end']}"
                lines.append(f"    {loc:<30s} {r['motif']:<8s} "
                              f"{r['baseline_max_hii']:>7.2f} {r['target_max_hii']:>7.2f} "
                              f"{r['delta_max_hii']:>7.2f}")

    # Matrix summary
    if len(data) >= 3:
        lines.append("\n--- Multi-Passage Matrix ---")
        loci, names, matrix, stats = build_matrix(data)
        cats = {}
        for s in stats:
            cats[s["category"]] = cats.get(s["category"], 0) + 1
        lines.append(f"  Common loci: {len(loci):,}")
        for cat in ["stable", "tissue_variable", "constitutive"]:
            n = cats.get(cat, 0)
            pct = n / max(len(loci), 1) * 100
            lines.append(f"  {cat:<20s}: {n:>8,} ({pct:.1f}%)")

    # Comparison with untagged results
    lines.append("\n--- Comparison with Untagged Analysis ---")
    lines.append("  (compare output/passage_drift/ vs output/passage_drift_hp/)")

    lines.append("")
    summary = "\n".join(lines)
    print(summary)

    # Save
    summary_path = os.path.join(output_dir, "passage_drift_summary_hp.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info("Summary saved to: %s", summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="HG008 cell line passage TR drift analysis (haplotagged BAMs)"
    )
    parser.add_argument("--output-dir", default="output/passage_drift_hp",
                        help="Output directory")
    parser.add_argument("--loci", default=TEST_LOCI,
                        help="TR loci BED file (default: 108K test loci)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads for instability analysis")
    parser.add_argument("--samples", nargs="*", default=None,
                        help="Samples to analyze (default: all available)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which samples to run
    sample_keys = args.samples or list(SAMPLES.keys())

    # Verify BAMs exist
    available_keys = []
    for key in sample_keys:
        bam = SAMPLES[key]["bam"]
        if os.path.exists(bam) and os.path.getsize(bam) > 0:
            available_keys.append(key)
            logger.info("Available: %s -> %s", key, bam)
        else:
            logger.warning("MISSING or empty: %s -> %s", key, bam)

    if not available_keys:
        logger.error("No BAMs available. Exiting.")
        sys.exit(1)

    # Step 1: Run instability on each sample
    tsv_paths = {}
    for key in available_keys:
        path = run_instability(key, args.loci, args.output_dir, args.threads)
        tsv_paths[key] = path

    # Step 2: Comparisons and matrix
    run_comparisons(args.output_dir, tsv_paths)

    # Step 3: Generate summary
    generate_summary(args.output_dir, tsv_paths)


if __name__ == "__main__":
    main()
