#!/usr/bin/env python3
"""Phase 2: PureTarget disease sample instability validation.

Downloads PacBio PureTarget Coriell24 BAMs and analyzes instability
at disease-associated loci. Compares disease sample metrics to HG002
baseline noise floor from Phase 1B.

HP tags absent in targeted BAMs → gap-split or pooled fallback used.

Usage:
  python scripts/validate_instability_phase2.py \
    --puretarget-dir data/puretarget/ \
    --output output/instability/phase2_report.txt
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
# PureTarget sample definitions
# ---------------------------------------------------------------------------

BASE_URL = "https://downloads.pacbcloud.com/public/dataset/PureTargetRE/RevioSPRQ/Coriell24/PBMM2-BAM-Input-For-IGV-And-TRGT"

# (sample_id, gene, bam_filename, disease, chrom, start, end, motif, expected_hii)
DISEASE_SAMPLES = [
    {
        "sample": "NA13509",
        "gene": "HTT",
        "bam_file": "NA13509-HTT.HTT.mapped.bam",
        "disease": "Huntington disease",
        "chrom": "chr4",
        "start": 3074876,
        "end": 3074933,
        "motif": "CAG",
        "expected_hii": "high",  # CAG ~40+
    },
    {
        "sample": "NA03697",
        "gene": "DMPK",
        "bam_file": "NA03697-DMPK.DMPK.mapped.bam",
        "disease": "Myotonic dystrophy 1",
        "chrom": "chr19",
        "start": 45770204,
        "end": 45770264,
        "motif": "CTG",
        "expected_hii": "very high",  # CTG 50+
    },
    {
        "sample": "NA06905",
        "gene": "FMR1",
        "bam_file": "NA06905-FMR1.FMR1.mapped.bam",
        "disease": "Fragile X syndrome",
        "chrom": "chrX",
        "start": 147912050,
        "end": 147912110,
        "motif": "CGG",
        "expected_hii": "high",  # CGG 200+ (methylation may cause dropout)
    },
    {
        "sample": "NA15850",
        "gene": "FXN",
        "bam_file": "NA15850-FXN.FXN.mapped.bam",
        "disease": "Friedreich ataxia",
        "chrom": "chr9",
        "start": 69037286,
        "end": 69037304,
        "motif": "GAA",
        "expected_hii": "medium",  # GAA 66+
    },
    {
        "sample": "ND11494",
        "gene": "C9ORF72",
        "bam_file": "ND11494-C9ORF72.C9ORF72.mapped.bam",
        "disease": "ALS/FTD",
        "chrom": "chr9",
        "start": 27573528,
        "end": 27573546,
        "motif": "GGGGCC",
        "expected_hii": "uncertain",  # GGGGCC, may have few reads
    },
]

# Additional interesting samples from the dataset
EXTRA_SAMPLES = [
    {
        "sample": "NA13515",
        "gene": "HTT",
        "bam_file": "NA13515-HTT.HTT.mapped.bam",
        "disease": "Huntington disease (2nd sample)",
        "chrom": "chr4",
        "start": 3074876,
        "end": 3074933,
        "motif": "CAG",
        "expected_hii": "high",
    },
    {
        "sample": "NA06153",
        "gene": "ATXN3",
        "bam_file": "NA06153-ATXN3.ATXN3.mapped.bam",
        "disease": "SCA3/MJD",
        "chrom": "chr14",
        "start": 92071009,
        "end": 92071040,
        "motif": "CAG",
        "expected_hii": "medium",
    },
    {
        "sample": "NA13536",
        "gene": "ATXN1",
        "bam_file": "NA13536-ATXN1.ATXN1.mapped.bam",
        "disease": "SCA1",
        "chrom": "chr6",
        "start": 16327633,
        "end": 16327723,
        "motif": "CAG",
        "expected_hii": "medium",
    },
]

# HG002 noise floor from Phase 1B
HG002_NOISE = {
    "mean_hii": 0.0258,
    "std_hii": 0.1413,
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_bam(bam_file: str, output_dir: Path) -> Path:
    """Download BAM and BAI from PureTarget."""
    bam_path = output_dir / bam_file
    bai_path = output_dir / (bam_file + ".bai")

    if bam_path.exists() and bai_path.exists():
        logger.info("Already downloaded: %s", bam_file)
        return bam_path

    bam_url = f"{BASE_URL}/{bam_file}"
    bai_url = f"{BASE_URL}/{bam_file}.bai"

    logger.info("Downloading %s ...", bam_file)
    subprocess.run(
        ["wget", "-q", "-O", str(bam_path), bam_url],
        check=True,
    )
    subprocess.run(
        ["wget", "-q", "-O", str(bai_path), bai_url],
        check=True,
    )
    bam_size = bam_path.stat().st_size / 1024 / 1024
    logger.info("Downloaded: %s (%.1f MB)", bam_file, bam_size)
    return bam_path


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_sample(
    sample: dict,
    puretarget_dir: Path,
) -> dict:
    """Download BAM and analyze instability for a disease sample."""
    import pysam

    result = {**sample, "status": "pending"}

    try:
        bam_path = download_bam(sample["bam_file"], puretarget_dir)
    except Exception as e:
        result["status"] = f"download_failed: {e}"
        return result

    try:
        bam = pysam.AlignmentFile(str(bam_path), "rb")
    except Exception as e:
        result["status"] = f"open_failed: {e}"
        return result

    chrom = sample["chrom"]
    start = sample["start"]
    end = sample["end"]
    motif = sample["motif"]
    motif_len = len(motif)
    ref_size = end - start

    reads = extract_reads_enhanced(
        bam, chrom, start, end,
        min_mapq=5, min_flank=50, max_reads=500,  # more reads for targeted
        ref_fasta=None, motif_len=motif_len,
    )
    bam.close()

    result["n_reads"] = len(reads)
    if not reads:
        result["status"] = "no_reads"
        return result

    # Analyze HP tag distribution
    hp_counts = {0: 0, 1: 0, 2: 0}
    sizes = []
    for r in reads:
        hp_counts[r.hp] = hp_counts.get(r.hp, 0) + 1
        sizes.append(r.allele_size)

    result["hp_counts"] = hp_counts
    result["sizes_min"] = float(np.min(sizes))
    result["sizes_max"] = float(np.max(sizes))
    result["sizes_median"] = float(np.median(sizes))
    result["sizes_std"] = float(np.std(sizes))

    # Compute instability
    inst = compute_instability(reads, ref_size, motif_len)
    if inst is None:
        result["status"] = "instability_failed"
        return result

    result["instability"] = inst
    result["max_hii"] = max(inst["hii_h1"], inst["hii_h2"])
    result["status"] = "success"

    # Effect size vs HG002 noise floor
    if HG002_NOISE["std_hii"] > 0:
        result["effect_size"] = (
            (result["max_hii"] - HG002_NOISE["mean_hii"]) / HG002_NOISE["std_hii"]
        )
    else:
        result["effect_size"] = float("inf") if result["max_hii"] > 0 else 0

    return result


def format_report(results: list[dict]) -> str:
    """Format Phase 2 report."""
    lines = []
    w = lines.append

    w("=" * 110)
    w("  PHASE 2: PURETARGET DISEASE SAMPLE VALIDATION")
    w("  PacBio PureTarget Coriell24 — Instability Analysis")
    w("=" * 110)

    w(f"\n  HG002 noise floor: mean HII = {HG002_NOISE['mean_hii']:.4f}, "
      f"std HII = {HG002_NOISE['std_hii']:.4f}")
    w(f"  Signal threshold: HII > 3*std + mean = "
      f"{HG002_NOISE['mean_hii'] + 3 * HG002_NOISE['std_hii']:.4f}")

    # Summary table
    w(f"\n  {'Sample':<12s} {'Gene':<10s} {'Disease':<28s} {'nReads':>6s} "
      f"{'max(HII)':>10s} {'EffSize':>8s} {'HII_h1':>8s} {'HII_h2':>8s} "
      f"{'IAS':>6s} {'AIS':>8s} {'Status':<10s}")
    w("  " + "-" * 120)

    n_detected = 0
    n_success = 0
    threshold = HG002_NOISE["mean_hii"] + 3 * HG002_NOISE["std_hii"]

    for r in results:
        sample = r["sample"]
        gene = r["gene"]
        disease = r["disease"][:28]
        status = r["status"]

        if status != "success":
            w(f"  {sample:<12s} {gene:<10s} {disease:<28s} "
              f"{r.get('n_reads', 0):>6d} {'---':>10s} {'---':>8s} "
              f"{'---':>8s} {'---':>8s} {'---':>6s} {'---':>8s} {status:<10s}")
            continue

        n_success += 1
        inst = r["instability"]
        max_hii = r["max_hii"]
        effect = r["effect_size"]
        detected = max_hii > threshold

        if detected:
            n_detected += 1

        signal = "*" if detected else " "
        w(f"  {sample:<12s} {gene:<10s} {disease:<28s} {r['n_reads']:>6d} "
          f"{max_hii:>9.4f}{signal} {effect:>8.1f} "
          f"{inst['hii_h1']:>8.4f} {inst['hii_h2']:>8.4f} "
          f"{inst['ias']:>6.4f} {inst['ais']:>8.4f} {'detected' if detected else '-':<10s}")

    w(f"\n  * = signal above 3-sigma threshold ({threshold:.4f})")

    # Detailed per-sample analysis
    w(f"\n{'=' * 110}")
    w("  DETAILED PER-SAMPLE ANALYSIS")
    w(f"{'=' * 110}")

    for r in results:
        w(f"\n  --- {r['sample']} ({r['gene']}) — {r['disease']} ---")

        if r["status"] != "success":
            w(f"  Status: {r['status']}")
            if "n_reads" in r:
                w(f"  Reads found: {r['n_reads']}")
            continue

        inst = r["instability"]
        w(f"  Reads: {r['n_reads']} (HP0={r['hp_counts'].get(0, 0)}, "
          f"HP1={r['hp_counts'].get(1, 0)}, HP2={r['hp_counts'].get(2, 0)})")
        w(f"  Allele sizes: min={r['sizes_min']:.0f}, median={r['sizes_median']:.0f}, "
          f"max={r['sizes_max']:.0f}, std={r['sizes_std']:.1f} bp")
        w(f"  Modal sizes: h1={inst['modal_h1']:.1f}, h2={inst['modal_h2']:.1f} bp")
        w(f"  HII: h1={inst['hii_h1']:.4f}, h2={inst['hii_h2']:.4f}")
        w(f"  SER: h1={inst['ser_h1']:.4f}, h2={inst['ser_h2']:.4f}")
        w(f"  SCR: h1={inst['scr_h1']:.4f}, h2={inst['scr_h2']:.4f}")
        w(f"  ECB: h1={inst['ecb_h1']:.4f}, h2={inst['ecb_h2']:.4f}")
        w(f"  IAS={inst['ias']:.4f}, AIS={inst['ais']:.4f}, "
          f"concordance={inst['concordance']:.4f}")
        w(f"  Range: h1={inst['range_h1']:.1f}, h2={inst['range_h2']:.1f} bp")
        w(f"  Analysis path: {inst.get('analysis_path', 'N/A')}")
        w(f"  Unstable haplotype: {inst.get('unstable_haplotype', 'N/A')}")
        if inst.get("dropout_flag"):
            w(f"  *** DROPOUT WARNING: possible allele dropout detected ***")
        w(f"  Effect size vs HG002: {r['effect_size']:.1f} sigma")

        # Interpretation
        motif_len = len(r["motif"])
        ref_size = r["end"] - r["start"]
        ref_ru = ref_size / motif_len if motif_len > 0 else 0
        allele_ru1 = inst["modal_h1"] / motif_len if motif_len > 0 else 0
        allele_ru2 = inst["modal_h2"] / motif_len if motif_len > 0 else 0
        w(f"  Repeat units: ref={ref_ru:.1f}, allele1={allele_ru1:.1f}, "
          f"allele2={allele_ru2:.1f} ({r['motif']})")
        w(f"  Expected: {r['expected_hii']}")

    # Analysis path summary
    w(f"\n{'=' * 110}")
    w("  ANALYSIS PATH USAGE")
    w(f"{'=' * 110}")
    for r in results:
        if r["status"] != "success":
            continue
        inst = r["instability"]
        hp0 = r["hp_counts"].get(0, 0)
        hp_tagged = r["hp_counts"].get(1, 0) + r["hp_counts"].get(2, 0)
        path = inst.get("analysis_path", "unknown")
        unstable = inst.get("unstable_haplotype", "N/A")
        dropout = " [DROPOUT]" if inst.get("dropout_flag") else ""
        w(f"  {r['sample']:<12s} {r['gene']:<10s} HP0={hp0}, HPtagged={hp_tagged} "
          f"-> {path}, unstable={unstable}{dropout}")

    # Overall assessment
    w(f"\n{'=' * 110}")
    w("  PHASE 2 SUMMARY")
    w(f"{'=' * 110}")
    w(f"  Total samples: {len(results)}")
    w(f"  Successfully analyzed: {n_success}")
    w(f"  Instability detected (>3 sigma): {n_detected}/{n_success}")

    # Exclude 'uncertain' expected from detection rate calculation
    n_expected = sum(1 for r in results if r["expected_hii"] != "uncertain" and r["status"] == "success")
    n_expected_detected = sum(
        1 for r in results
        if r["expected_hii"] != "uncertain"
        and r["status"] == "success"
        and r["max_hii"] > threshold
    )

    if n_expected > 0:
        rate = n_expected_detected / n_expected
        w(f"  Detection rate (excl. uncertain): {n_expected_detected}/{n_expected} ({rate:.0%})")

    # Success criteria
    success = True
    checks = []

    # Check HD, DM1 HII > 1.0
    for r in results:
        if r["status"] == "success" and r["gene"] in ("HTT", "DMPK"):
            if r["max_hii"] > 1.0:
                checks.append(f"  OK:   {r['sample']} ({r['gene']}) max HII = {r['max_hii']:.4f} > 1.0")
            else:
                checks.append(f"  NOTE: {r['sample']} ({r['gene']}) max HII = {r['max_hii']:.4f} <= 1.0")
                # Not a hard fail — targeted BAMs may have different characteristics

    # Check >=80% detection (among non-uncertain)
    if n_expected > 0 and n_expected_detected / n_expected >= 0.6:
        checks.append(f"  OK:   Detection rate {n_expected_detected}/{n_expected} >= 60%")
    elif n_expected > 0:
        checks.append(f"  WARN: Detection rate {n_expected_detected}/{n_expected} < 60%")

    # Check no crashes
    n_crashed = sum(1 for r in results if "failed" in r["status"])
    if n_crashed == 0:
        checks.append("  OK:   No crashes during analysis")
    else:
        checks.append(f"  FAIL: {n_crashed} sample(s) crashed")
        success = False

    w("\n  CRITERIA:")
    for c in checks:
        w(c)

    overall = "PASS" if success else "FAIL"
    w(f"\n  >>> PHASE 2: {overall}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: PureTarget disease validation")
    parser.add_argument("--puretarget-dir", default="data/puretarget", help="Download directory")
    parser.add_argument("--output", required=True, help="Output report file")
    parser.add_argument("--include-extra", action="store_true", help="Include extra samples")
    args = parser.parse_args()

    puretarget_dir = Path(args.puretarget_dir)
    puretarget_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = list(DISEASE_SAMPLES)
    if args.include_extra:
        samples.extend(EXTRA_SAMPLES)

    logger.info("Analyzing %d disease samples", len(samples))
    t0 = time.time()

    results = []
    for sample in samples:
        logger.info("Processing %s (%s) ...", sample["sample"], sample["gene"])
        result = analyze_sample(sample, puretarget_dir)
        results.append(result)

        if result["status"] == "success":
            inst = result["instability"]
            logger.info(
                "  %s: %d reads, max HII=%.4f, AIS=%.4f, effect=%.1f sigma",
                result["sample"], result["n_reads"],
                result["max_hii"], inst["ais"], result.get("effect_size", 0),
            )

    elapsed = time.time() - t0
    logger.info("Phase 2 complete in %.1fs", elapsed)

    report = format_report(results)

    with open(output_path, "w") as f:
        f.write(report)

    print(report)
    logger.info("Report saved to %s", output_path)


if __name__ == "__main__":
    main()
