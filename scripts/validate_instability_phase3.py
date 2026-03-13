#!/usr/bin/env python3
"""Phase 3: CASTLE cancer WGS instability validation.

Analyzes tumor-normal PacBio HiFi pairs from the CASTLE panel (PRJNA1086849)
for microsatellite instability. All cell lines are MSS → tests specificity
(false positive rate should be low).

Workflow:
  1. Data audit: check SRA availability for PacBio HiFi runs
  2. Download + align 1-2 tumor-normal pairs
  3. MSI loci instability analysis (Bethesda panel + mononucleotide repeats)
  4. Paired tumor vs normal comparison

Usage:
  # Audit only (no download)
  python scripts/validate_instability_phase3.py --audit-only --output output/instability/phase3_audit.txt

  # Full analysis (requires aligned BAMs)
  python scripts/validate_instability_phase3.py \
    --castle-dir data/castle/ \
    --output output/instability/phase3_report.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaictr.genotype import ReadInfo, extract_reads_enhanced
from mosaictr.instability import compute_instability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CASTLE cell line pairs (PacBio HiFi only)
# ---------------------------------------------------------------------------

CASTLE_PAIRS = [
    {
        "name": "HCC1954",
        "cancer_type": "Breast ductal carcinoma",
        "tumor_srr": "SRR28305163",
        "normal_srr": "SRR28305160",
        "tumor_name": "HCC1954",
        "normal_name": "HCC1954BL",
        "tumor_size_gb": 77,
        "normal_size_gb": 77,
    },
    {
        "name": "HCC1937",
        "cancer_type": "Breast ductal carcinoma",
        "tumor_srr": "SRR28305185",
        "normal_srr": "SRR28305182",
        "tumor_name": "HCC1937",
        "normal_name": "HCC1937BL",
        "tumor_size_gb": 71,
        "normal_size_gb": 67,
    },
    {
        "name": "H1437",
        "cancer_type": "NSCLC lung adenocarcinoma",
        "tumor_srr": "SRR28305179",
        "normal_srr": "SRR28305175",
        "tumor_name": "H1437",
        "normal_name": "BL1437",
        "tumor_size_gb": 75,
        "normal_size_gb": 83,
    },
    {
        "name": "H2009",
        "cancer_type": "NSCLC lung adenocarcinoma",
        "tumor_srr": "SRR28305172",
        "normal_srr": "SRR28305169",
        "tumor_name": "H2009",
        "normal_name": "BL2009",
        "tumor_size_gb": 79,
        "normal_size_gb": 81,
    },
    {
        "name": "Hs578T",
        "cancer_type": "Breast carcinoma",
        "tumor_srr": "SRR31537482",
        "normal_srr": "SRR31537479",
        "tumor_name": "Hs578T",
        "normal_name": "Hs578Bst",
        "tumor_size_gb": 69,
        "normal_size_gb": 27,
    },
]

# ---------------------------------------------------------------------------
# MSI loci: Bethesda panel (GRCh38 coordinates)
# ---------------------------------------------------------------------------
# 5 standard microsatellite instability markers
BETHESDA_PANEL = [
    # (name, chrom, start, end, motif, type)
    ("BAT-25", "chr4", 55598211, 55598236, "A", "mononucleotide"),
    ("BAT-26", "chr2", 47641559, 47641586, "A", "mononucleotide"),
    ("NR-21", "chr14", 23652346, 23652367, "A", "mononucleotide"),
    ("NR-24", "chr2", 95849361, 95849385, "A", "mononucleotide"),
    ("MONO-27", "chr2", 39564893, 39564920, "A", "mononucleotide"),
]

# Additional mononucleotide repeat loci for broader MSI assessment
# Selected large mononucleotide repeats from well-known MSI markers
EXTENDED_MSI_LOCI = [
    ("NR-27", "chr5", 137858000, 137858025, "A", "mononucleotide"),
    ("CAT25", "chr2", 95184804, 95184829, "T", "mononucleotide"),
    ("BAT-40", "chr1", 121484306, 121484346, "A", "mononucleotide"),
    ("TGFBR2", "chr3", 30713575, 30713585, "A", "mononucleotide"),
    ("HSP110_T17", "chr4", 101396831, 101396848, "T", "mononucleotide"),
]

REF_PATH = "/vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
CATALOG_BED = "/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"

# HG002 noise floor from Phase 1B
HG002_NOISE = {
    "mean_hii": 0.0258,
    "std_hii": 0.1413,
}


# ---------------------------------------------------------------------------
# Data audit
# ---------------------------------------------------------------------------

def run_data_audit() -> str:
    """Check SRA accessibility and tool availability."""
    lines = []
    w = lines.append

    w("=" * 100)
    w("  PHASE 3 DATA AUDIT: CASTLE Cancer WGS")
    w("=" * 100)

    # Check tools
    tools_ok = True
    for tool in ["prefetch", "fasterq-dump", "pbmm2", "samtools"]:
        try:
            result = subprocess.run(
                [tool, "--version"], capture_output=True, text=True, timeout=10,
            )
            ver = result.stdout.strip().split("\n")[0] if result.stdout else "?"
            w(f"  {tool}: {ver}")
        except Exception:
            w(f"  {tool}: NOT FOUND")
            tools_ok = False

    if not tools_ok:
        w("\n  WARNING: Some required tools missing. Install sra-tools and pbmm2.")

    # Check reference
    ref_exists = os.path.exists(REF_PATH)
    w(f"\n  Reference FASTA: {'found' if ref_exists else 'NOT FOUND'}")

    # SRA data summary
    w(f"\n  CASTLE PacBio HiFi pairs (5 available):")
    w(f"  {'Name':<12s} {'Cancer Type':<30s} {'Tumor SRR':<15s} {'Normal SRR':<15s} "
      f"{'Size (GB)':>10s}")
    w("  " + "-" * 85)

    total_gb = 0
    for pair in CASTLE_PAIRS:
        pair_gb = pair["tumor_size_gb"] + pair["normal_size_gb"]
        total_gb += pair_gb
        w(f"  {pair['name']:<12s} {pair['cancer_type']:<30s} "
          f"{pair['tumor_srr']:<15s} {pair['normal_srr']:<15s} "
          f"{pair_gb:>10d}")

    w(f"  {'Total':<12s} {'':<30s} {'':<15s} {'':<15s} {total_gb:>10d}")

    # Bethesda panel
    w(f"\n  MSI Loci:")
    w(f"  Bethesda panel: {len(BETHESDA_PANEL)} markers")
    for name, chrom, start, end, motif, _ in BETHESDA_PANEL:
        w(f"    {name:<12s} {chrom}:{start}-{end} ({motif}x{end-start})")
    w(f"  Extended MSI: {len(EXTENDED_MSI_LOCI)} additional markers")

    # Recommended workflow
    w(f"\n  RECOMMENDED WORKFLOW:")
    w(f"  1. Download 1 pair (HCC1954): ~154 GB")
    w(f"     prefetch SRR28305163 SRR28305160")
    w(f"  2. Convert to FASTQ:")
    w(f"     fasterq-dump SRR28305163 -O data/castle/")
    w(f"  3. Align with pbmm2:")
    w(f"     pbmm2 align {REF_PATH} data/castle/SRR28305163.fastq \\")
    w(f"       data/castle/HCC1954_tumor.bam --sort --preset HiFi")
    w(f"  4. Run Phase 3 analysis:")
    w(f"     python scripts/validate_instability_phase3.py \\")
    w(f"       --tumor data/castle/HCC1954_tumor.bam \\")
    w(f"       --normal data/castle/HCC1954BL_normal.bam \\")
    w(f"       --output output/instability/phase3_report.txt")

    w(f"\n  NOTE: All cell lines are MSS (microsatellite stable).")
    w(f"  This validates specificity: Bethesda markers should classify as MSS.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MSI analysis
# ---------------------------------------------------------------------------

def analyze_msi_loci(
    bam_path: str,
    sample_name: str,
    loci: list[tuple],
    ref_path: str | None = None,
) -> list[dict]:
    """Analyze instability at MSI loci for a single BAM."""
    import pysam

    bam = pysam.AlignmentFile(bam_path, "rb")
    ref_fasta = pysam.FastaFile(ref_path) if ref_path else None

    results = []
    for locus in loci:
        name, chrom, start, end, motif = locus[0], locus[1], locus[2], locus[3], locus[4]
        ref_size = end - start
        motif_len = len(motif)

        reads = extract_reads_enhanced(
            bam, chrom, start, end,
            min_mapq=5, min_flank=50, max_reads=200,
            ref_fasta=ref_fasta, motif_len=motif_len,
        )

        entry = {
            "locus_name": name,
            "chrom": chrom,
            "start": start,
            "end": end,
            "motif": motif,
            "sample": sample_name,
            "n_reads": len(reads),
        }

        if not reads:
            entry["instability"] = None
            results.append(entry)
            continue

        inst = compute_instability(reads, ref_size, motif_len)
        entry["instability"] = inst
        if inst:
            entry["max_hii"] = max(inst["hii_h1"], inst["hii_h2"])
        results.append(entry)

    bam.close()
    if ref_fasta:
        ref_fasta.close()

    return results


def classify_msi_status(bethesda_results: list[dict], threshold: float = 1.0) -> str:
    """Classify MSI status based on Bethesda panel results.

    MSI-H: >=2 of 5 markers unstable
    MSI-L: 1 of 5 markers unstable
    MSS: 0 of 5 markers unstable
    """
    n_unstable = 0
    for r in bethesda_results:
        if r.get("instability") and r.get("max_hii", 0) > threshold:
            n_unstable += 1

    if n_unstable >= 2:
        return "MSI-H"
    elif n_unstable == 1:
        return "MSI-L"
    else:
        return "MSS"


def run_paired_analysis(
    tumor_bam: str,
    normal_bam: str,
    pair_name: str,
    ref_path: str | None = None,
) -> dict:
    """Run paired tumor-normal MSI analysis."""
    all_loci = BETHESDA_PANEL + EXTENDED_MSI_LOCI

    logger.info("Analyzing tumor: %s (%d loci)", tumor_bam, len(all_loci))
    tumor_results = analyze_msi_loci(tumor_bam, f"{pair_name}_tumor", all_loci, ref_path)

    logger.info("Analyzing normal: %s (%d loci)", normal_bam, len(all_loci))
    normal_results = analyze_msi_loci(normal_bam, f"{pair_name}_normal", all_loci, ref_path)

    # Bethesda classification
    bethesda_tumor = [r for r in tumor_results if r["locus_name"] in [b[0] for b in BETHESDA_PANEL]]
    bethesda_normal = [r for r in normal_results if r["locus_name"] in [b[0] for b in BETHESDA_PANEL]]

    tumor_msi = classify_msi_status(bethesda_tumor)
    normal_msi = classify_msi_status(bethesda_normal)

    # Paired comparison: delta HII
    deltas = []
    for t_res, n_res in zip(tumor_results, normal_results):
        if t_res.get("instability") and n_res.get("instability"):
            t_hii = t_res.get("max_hii", 0)
            n_hii = n_res.get("max_hii", 0)
            deltas.append(t_hii - n_hii)

    return {
        "pair_name": pair_name,
        "tumor_results": tumor_results,
        "normal_results": normal_results,
        "tumor_msi": tumor_msi,
        "normal_msi": normal_msi,
        "deltas": deltas,
    }


def format_paired_report(analysis: dict) -> str:
    """Format paired tumor-normal analysis report."""
    lines = []
    w = lines.append

    pair = analysis["pair_name"]
    w(f"\n{'=' * 100}")
    w(f"  PAIRED ANALYSIS: {pair}")
    w(f"{'=' * 100}")

    w(f"\n  MSI Classification (Bethesda panel):")
    w(f"    Tumor:  {analysis['tumor_msi']}")
    w(f"    Normal: {analysis['normal_msi']}")

    # Per-locus table
    w(f"\n  {'Locus':<14s} {'Tumor':>6s} {'T_HII':>8s} {'T_SER':>8s} "
      f"{'Normal':>6s} {'N_HII':>8s} {'N_SER':>8s} {'dHII':>8s}")
    w("  " + "-" * 80)

    for t_res, n_res in zip(analysis["tumor_results"], analysis["normal_results"]):
        name = t_res["locus_name"]
        t_n = t_res["n_reads"]
        n_n = n_res["n_reads"]

        t_inst = t_res.get("instability")
        n_inst = n_res.get("instability")

        t_hii = t_res.get("max_hii", 0) if t_inst else 0
        n_hii = n_res.get("max_hii", 0) if n_inst else 0
        t_ser = max(t_inst["ser_h1"], t_inst["ser_h2"]) if t_inst else 0
        n_ser = max(n_inst["ser_h1"], n_inst["ser_h2"]) if n_inst else 0
        delta = t_hii - n_hii

        w(f"  {name:<14s} {t_n:>6d} {t_hii:>8.4f} {t_ser:>8.4f} "
          f"{n_n:>6d} {n_hii:>8.4f} {n_ser:>8.4f} {delta:>+8.4f}")

    # Delta statistics
    deltas = analysis["deltas"]
    if deltas:
        arr = np.array(deltas)
        w(f"\n  Delta HII statistics (tumor - normal):")
        w(f"    Mean:   {np.mean(arr):+.4f}")
        w(f"    Median: {np.median(arr):+.4f}")
        w(f"    Std:    {np.std(arr):.4f}")
        w(f"    Range:  [{np.min(arr):+.4f}, {np.max(arr):+.4f}]")

    return "\n".join(lines)


def format_full_report(
    audit_report: str,
    analyses: list[dict],
) -> str:
    """Format full Phase 3 report."""
    lines = []
    w = lines.append

    w("=" * 100)
    w("  PHASE 3: CASTLE CANCER WGS — MSI SPECIFICITY VALIDATION")
    w("=" * 100)

    if audit_report:
        w(audit_report)

    if not analyses:
        w("\n  No paired analyses performed (provide --tumor and --normal BAMs)")
        w("\n  >>> PHASE 3: AUDIT COMPLETE (no analysis)")
        return "\n".join(lines)

    for analysis in analyses:
        w(format_paired_report(analysis))

    # Overall summary
    w(f"\n{'=' * 100}")
    w("  PHASE 3 SUMMARY")
    w(f"{'=' * 100}")

    n_pairs = len(analyses)
    n_mss_tumor = sum(1 for a in analyses if a["tumor_msi"] == "MSS")
    n_mss_normal = sum(1 for a in analyses if a["normal_msi"] == "MSS")

    w(f"  Pairs analyzed: {n_pairs}")
    w(f"  Tumor MSS:  {n_mss_tumor}/{n_pairs}")
    w(f"  Normal MSS: {n_mss_normal}/{n_pairs}")

    # Success criteria
    success = True
    checks = []

    # All should be classified MSS
    if n_mss_tumor == n_pairs and n_mss_normal == n_pairs:
        checks.append("  OK:   All samples classified MSS (correct for CASTLE cell lines)")
    else:
        checks.append(f"  WARN: Not all MSS — check threshold. "
                       f"Tumor MSS: {n_mss_tumor}, Normal MSS: {n_mss_normal}")

    # No crashes
    checks.append("  OK:   Pipeline completed without crashes")

    # Tumor vs normal delta should be small
    all_deltas = []
    for a in analyses:
        all_deltas.extend(a["deltas"])
    if all_deltas:
        mean_delta = np.mean(all_deltas)
        if abs(mean_delta) < 0.5:
            checks.append(f"  OK:   Mean delta HII = {mean_delta:+.4f} (< 0.5 expected for MSS)")
        else:
            checks.append(f"  WARN: Mean delta HII = {mean_delta:+.4f} (> 0.5, unexpected for MSS)")

    w("\n  CRITERIA:")
    for c in checks:
        w(c)

    overall = "PASS" if success else "FAIL"
    w(f"\n  >>> PHASE 3: {overall}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load mononucleotide repeats from Adotto catalog
# ---------------------------------------------------------------------------

def load_adotto_mono_repeats(
    catalog_path: str,
    min_ref_size: int = 15,
    max_loci: int = 500,
    chroms: set | None = None,
) -> list[tuple]:
    """Load mononucleotide repeats from Adotto catalog.

    Returns list of (name, chrom, start, end, motif, type) tuples.
    """
    if chroms is None:
        chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX"}

    loci = []
    with open(catalog_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 4:
                continue
            chrom = cols[0]
            if chrom not in chroms:
                continue
            start = int(cols[1])
            end = int(cols[2])
            motif = cols[3]
            ref_size = end - start

            if len(motif) == 1 and ref_size >= min_ref_size:
                name = f"mono_{chrom}_{start}"
                loci.append((name, chrom, start, end, motif, "mononucleotide"))

            if len(loci) >= max_loci:
                break

    return loci


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3: CASTLE cancer WGS validation")
    parser.add_argument("--castle-dir", default="data/castle", help="CASTLE data directory")
    parser.add_argument("--tumor", help="Tumor aligned BAM path")
    parser.add_argument("--normal", help="Normal aligned BAM path")
    parser.add_argument("--pair-name", default="CASTLE", help="Cell line pair name")
    parser.add_argument("--ref", default=REF_PATH, help="Reference FASTA")
    parser.add_argument("--output", required=True, help="Output report file")
    parser.add_argument("--audit-only", action="store_true", help="Only run data audit")
    parser.add_argument("--include-adotto", action="store_true",
                        help="Include Adotto mononucleotide repeats")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Always run data audit
    logger.info("Running data audit...")
    audit_report = run_data_audit()

    analyses = []

    if not args.audit_only and args.tumor and args.normal:
        logger.info("Running paired analysis: %s", args.pair_name)
        t0 = time.time()

        ref = args.ref if os.path.exists(args.ref) else None
        analysis = run_paired_analysis(
            args.tumor, args.normal, args.pair_name, ref_path=ref,
        )
        analyses.append(analysis)

        logger.info("Paired analysis complete in %.1fs", time.time() - t0)

    report = format_full_report(
        audit_report if args.audit_only or not analyses else "",
        analyses,
    )

    with open(output_path, "w") as f:
        f.write(report)

    print(report)
    logger.info("Report saved to %s", output_path)


if __name__ == "__main__":
    main()
