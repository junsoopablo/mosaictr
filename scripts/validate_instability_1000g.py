#!/usr/bin/env python3
"""1000G ONT ATXN10 carrier instability analysis pipeline.

Demonstrates cross-platform (ONT) + disease locus per-haplotype instability.

Pipeline steps:
  1. find-carrier   — Guide for identifying ATXN10 carrier from pathSTR
  2. download       — Download ONT CRAM from IGSR (1000 Genomes)
  3. haplotag       — Haplotag with whatshap using 1000G phased VCF
  4. instability    — Run MosaicTR instability on ATXN10 locus
  5. run-all        — Complete pipeline (steps 2-4)

Usage:
  # Step 1: Find carrier sample ID
  python scripts/validate_instability_1000g.py find-carrier

  # Steps 2-4: Complete pipeline (after identifying sample)
  python scripts/validate_instability_1000g.py run-all \
    --sample HG00XXX --outdir output/instability/1000g/

  # Or individual steps:
  python scripts/validate_instability_1000g.py download \
    --sample HG00XXX --outdir output/instability/1000g/
  python scripts/validate_instability_1000g.py haplotag \
    --bam output/instability/1000g/HG00XXX.hg38.cram \
    --sample HG00XXX --outdir output/instability/1000g/
  python scripts/validate_instability_1000g.py instability \
    --bam output/instability/1000g/HG00XXX_atxn10_tagged.bam \
    --output output/instability/1000g/atxn10_report.txt
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATXN10_LOCUS = {
    "chrom": "chr22",
    "start": 45795354,
    "end": 45795424,
    "motif": "ATTCT",
    "gene": "ATXN10",
    "disease": "SCA10",
    "normal_range": (10, 29),
    "pathological": 800,
}

# Reference genome
REF_PATH = (
    "/vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/"
    "GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
)

# 1000G ONT data base URL (IGSR)
IGSR_BASE = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1KG_ONT_VIENNA/hg38"
)

# 1000G phased VCF for chr22
PHASED_VCF_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/working/"
    "20220422_3202_phased_SNV_INDEL_SV/"
    "1kGP_high_coverage_Illumina.chr22.filtered."
    "SNV_INDEL_SV_phased_panel.vcf.gz"
)

# Surrounding TR loci for broader instability context (±500kb of ATXN10)
SURROUNDING_LOCI = [
    # (chrom, start, end, motif) — a few loci near ATXN10 for noise floor
    ("chr22", 45795354, 45795424, "ATTCT"),   # ATXN10 itself
    ("chr22", 46191235, 46191304, "ATTCT"),   # ATXN10-related region
]


# ===========================================================================
# Subcommand: find-carrier
# ===========================================================================

def cmd_find_carrier(args):
    """Guide for finding ATXN10 expansion carrier from pathSTR."""
    print("=" * 80)
    print("  FINDING ATXN10 CARRIER FROM pathSTR / 1000G ONT DATA")
    print("=" * 80)
    print()
    print("  ATXN10 locus details:")
    print(f"    Location: {ATXN10_LOCUS['chrom']}:"
          f"{ATXN10_LOCUS['start']}-{ATXN10_LOCUS['end']}")
    print(f"    Motif: {ATXN10_LOCUS['motif']} (5bp)")
    print(f"    Normal: {ATXN10_LOCUS['normal_range'][0]}-"
          f"{ATXN10_LOCUS['normal_range'][1]} RU")
    print(f"    Pathological: >{ATXN10_LOCUS['pathological']} RU")
    print()
    print("  Option 1: pathSTR web interface")
    print("    1. Visit https://pathstr.org")
    print("    2. Search for ATXN10")
    print("    3. Filter for samples with >800 repeat units")
    print("    4. Note the sample ID (e.g., HG00XXX or NA12XXX)")
    print()
    print("  Option 2: pathSTR GitHub data")
    print("    1. Clone https://github.com/Illumina/pathSTR")
    print("    2. Look in database files for ATXN10 genotypes")
    print("    3. Filter for expansion carriers")
    print()
    print("  Option 3: Literature search")
    print("    - Dolzhenko et al. (2024): TR variation in 1000G ONT")
    print("    - Supplementary tables may list ATXN10 carriers")
    print("    - Also check: Tanudisastro et al. pathSTR paper")
    print()
    print("  Option 4: Direct TRGT/STRchive genotypes on 1000G ONT")
    print("    - IGSR may provide pre-computed TR genotypes")
    print("    - Filter for ATXN10 expansions")
    print()
    print("  Verify ONT data availability:")
    print(f"    URL pattern: {IGSR_BASE}/<SAMPLE_ID>/")
    print("    File: <SAMPLE_ID>.hg38.cram (~40-50GB)")
    print()
    print("  Once carrier identified, run:")
    print("    python scripts/validate_instability_1000g.py run-all \\")
    print("      --sample <SAMPLE_ID> \\")
    print("      --outdir output/instability/1000g/")


# ===========================================================================
# Subcommand: download
# ===========================================================================

def cmd_download(args) -> str:
    """Download ONT CRAM from 1000G IGSR."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample = args.sample
    cram_name = f"{sample}.hg38.cram"
    cram_url = f"{IGSR_BASE}/{sample}/{cram_name}"
    crai_url = f"{cram_url}.crai"

    cram_path = outdir / cram_name
    crai_path = outdir / f"{cram_name}.crai"

    # Download CRAM
    if cram_path.exists():
        logger.info("CRAM already exists: %s", cram_path)
    else:
        logger.info("Downloading CRAM: %s", cram_url)
        logger.info("Expected size: ~40-50GB (may take hours)")
        cmd = ["wget", "-c", "-O", str(cram_path), cram_url]
        logger.info("Command: %s", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        else:
            logger.info("[DRY RUN] Skipping download")

    # Download index
    if crai_path.exists():
        logger.info("CRAM index already exists: %s", crai_path)
    elif not args.dry_run:
        logger.info("Downloading CRAM index: %s", crai_url)
        subprocess.run(
            ["wget", "-c", "-O", str(crai_path), crai_url], check=True
        )
    else:
        logger.info("[DRY RUN] Skipping index download")

    logger.info("CRAM path: %s", cram_path)
    return str(cram_path)


# ===========================================================================
# Subcommand: haplotag
# ===========================================================================

def cmd_haplotag(args) -> str:
    """Haplotag BAM/CRAM with whatshap using 1000G phased VCF."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_bam = args.bam
    sample = args.sample
    ref = args.ref

    # Extract ATXN10 region (±500kb for context and flanking loci)
    region = (f"{ATXN10_LOCUS['chrom']}:"
              f"{ATXN10_LOCUS['start'] - 500000}-"
              f"{ATXN10_LOCUS['end'] + 500000}")
    region_bam = outdir / f"{sample}_atxn10_region.bam"
    tagged_bam = outdir / f"{sample}_atxn10_tagged.bam"

    # Step A: Extract region
    if not region_bam.exists():
        logger.info("Extracting ATXN10 region (%s) from %s", region, input_bam)
        cmd = [
            "samtools", "view", "-b", "-h",
            "-T", ref,
            "-o", str(region_bam),
            input_bam,
            region,
        ]
        logger.info("Command: %s", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            subprocess.run(["samtools", "index", str(region_bam)], check=True)
    else:
        logger.info("Region BAM exists: %s", region_bam)

    # Step B: Download phased VCF if needed
    vcf = args.vcf
    if not vcf:
        vcf_local = outdir / "1kgp_chr22_phased.vcf.gz"
        if not vcf_local.exists():
            logger.info("Downloading 1000G phased VCF for chr22...")
            if not args.dry_run:
                subprocess.run(
                    ["wget", "-c", "-O", str(vcf_local), PHASED_VCF_URL],
                    check=True,
                )
                subprocess.run(
                    ["wget", "-c", "-O", str(vcf_local) + ".tbi",
                     PHASED_VCF_URL + ".tbi"],
                    check=True,
                )
            else:
                logger.info("[DRY RUN] Would download VCF")
        vcf = str(vcf_local)

    # Step C: Haplotag with whatshap
    if not tagged_bam.exists():
        logger.info("Running whatshap haplotag...")
        cmd = [
            "whatshap", "haplotag",
            "--reference", ref,
            "--sample", sample,
            "-o", str(tagged_bam),
            vcf,
            str(region_bam),
        ]
        logger.info("Command: %s", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            subprocess.run(["samtools", "index", str(tagged_bam)], check=True)
    else:
        logger.info("Tagged BAM exists: %s", tagged_bam)

    logger.info("Haplotagged BAM: %s", tagged_bam)
    return str(tagged_bam)


# ===========================================================================
# Subcommand: instability
# ===========================================================================

def cmd_instability(args):
    """Run MosaicTR instability on ATXN10 locus and produce report."""
    from mosaictr.genotype import ReadInfo, extract_reads_enhanced
    from mosaictr.instability import (
        _instability_pooled_fallback,
        compute_instability,
    )
    import pysam

    outdir = Path(args.output).parent
    outdir.mkdir(parents=True, exist_ok=True)

    bam_path = args.bam
    ref_path = args.ref

    logger.info("Analyzing instability at ATXN10 locus")
    logger.info("BAM: %s", bam_path)

    bam = pysam.AlignmentFile(bam_path, "rb")
    ref_fasta = pysam.FastaFile(ref_path) if ref_path else None

    chrom = ATXN10_LOCUS["chrom"]
    start = ATXN10_LOCUS["start"]
    end = ATXN10_LOCUS["end"]
    motif = ATXN10_LOCUS["motif"]
    motif_len = len(motif)
    ref_size = end - start

    reads = extract_reads_enhanced(
        bam, chrom, start, end,
        min_mapq=5, min_flank=50, max_reads=500,
        ref_fasta=ref_fasta, motif_len=motif_len,
    )

    bam.close()
    if ref_fasta:
        ref_fasta.close()

    if not reads:
        logger.error("No reads found at ATXN10 locus")
        return

    logger.info("Extracted %d reads at ATXN10", len(reads))

    # HP-tagged instability
    result = compute_instability(reads, ref_size, motif_len)
    if result is None:
        logger.error("compute_instability returned None")
        return

    # Also compute forced-pooled for comparison
    pooled = _instability_pooled_fallback(reads, ref_size, motif_len)

    # Read statistics
    hp1_reads = [r for r in reads if r.hp == 1]
    hp2_reads = [r for r in reads if r.hp == 2]
    hp0_reads = [r for r in reads if r.hp == 0]
    all_sizes = np.array([r.allele_size for r in reads])

    # Build report
    lines = []
    w = lines.append

    w("=" * 80)
    w("  1000G ONT — ATXN10 CARRIER INSTABILITY ANALYSIS")
    w("=" * 80)

    w(f"\n  LOCUS")
    w(f"  Gene:       {ATXN10_LOCUS['gene']}")
    w(f"  Disease:    {ATXN10_LOCUS['disease']} (Spinocerebellar Ataxia Type 10)")
    w(f"  Position:   {chrom}:{start}-{end}")
    w(f"  Motif:      {motif} ({motif_len}bp)")
    w(f"  Ref size:   {ref_size}bp ({ref_size / motif_len:.0f} RU)")
    w(f"  BAM:        {bam_path}")

    w(f"\n  READ STATISTICS")
    w(f"  Total reads:    {len(reads)}")
    w(f"  HP=1 reads:     {len(hp1_reads)}")
    w(f"  HP=2 reads:     {len(hp2_reads)}")
    w(f"  HP=0 reads:     {len(hp0_reads)}")
    w(f"  Size range:     {np.min(all_sizes):.0f} - {np.max(all_sizes):.0f} bp")
    w(f"  Size median:    {np.median(all_sizes):.0f} bp")

    w(f"\n  PER-HAPLOTYPE INSTABILITY METRICS")
    w(f"  Analysis path:  {result['analysis_path']}")
    w(f"  Modal H1:       {result['modal_h1']:.1f} bp "
      f"({result['modal_h1'] / motif_len:.1f} RU)")
    w(f"  Modal H2:       {result['modal_h2']:.1f} bp "
      f"({result['modal_h2'] / motif_len:.1f} RU)")
    w(f"  HII H1:         {result['hii_h1']:.4f}")
    w(f"  HII H2:         {result['hii_h2']:.4f}")
    w(f"  SER H1:         {result['ser_h1']:.4f}")
    w(f"  SER H2:         {result['ser_h2']:.4f}")
    w(f"  SCR H1:         {result['scr_h1']:.4f}")
    w(f"  SCR H2:         {result['scr_h2']:.4f}")
    w(f"  ECB H1:         {result['ecb_h1']:.4f}")
    w(f"  ECB H2:         {result['ecb_h2']:.4f}")
    w(f"  IAS:            {result['ias']:.4f}")
    w(f"  AIS:            {result['ais']:.4f}")
    w(f"  Concordance:    {result['concordance']:.4f}")
    w(f"  Range H1:       {result['range_h1']:.1f} bp")
    w(f"  Range H2:       {result['range_h2']:.1f} bp")
    w(f"  Unstable hap:   {result['unstable_haplotype']}")
    w(f"  Dropout flag:   {result['dropout_flag']}")

    # Pooled comparison
    w(f"\n  POOLED COMPARISON")
    w(f"  Pooled HII:     {pooled['hii_h1']:.4f}")
    w(f"  HP-tagged max:  {max(result['hii_h1'], result['hii_h2']):.4f}")
    pooled_hii = pooled['hii_h1']
    hp_max_hii = max(result['hii_h1'], result['hii_h2'])
    if pooled_hii > hp_max_hii + 0.01:
        w(f"  Pooled inflated by {pooled_hii - hp_max_hii:.4f} "
          f"due to allele mixing")
    else:
        w(f"  Similar (locus may be homozygous or pooled captured "
          f"expanded allele)")

    # Interpretation
    w(f"\n  INTERPRETATION")
    normal_allele = min(result['modal_h1'], result['modal_h2'])
    expanded_allele = max(result['modal_h1'], result['modal_h2'])
    normal_ru = normal_allele / motif_len
    expanded_ru = expanded_allele / motif_len

    w(f"  Normal allele:    {normal_allele:.0f} bp ({normal_ru:.0f} RU)")
    w(f"  Expanded allele:  {expanded_allele:.0f} bp ({expanded_ru:.0f} RU)")

    if expanded_ru > ATXN10_LOCUS['pathological']:
        w(f"  Status: PATHOLOGICAL (>{ATXN10_LOCUS['pathological']} RU)")
    elif expanded_ru > ATXN10_LOCUS['normal_range'][1]:
        w(f"  Status: INTERMEDIATE / REDUCED PENETRANCE")
    else:
        w(f"  Status: WITHIN NORMAL RANGE")

    unstable_hii = max(result['hii_h1'], result['hii_h2'])
    stable_hii = min(result['hii_h1'], result['hii_h2'])

    if result['ias'] > 0.8:
        w(f"\n  Per-haplotype decomposition demonstrates:")
        w(f"    - Normal allele:   HII = {stable_hii:.4f} (stable)")
        w(f"    - Expanded allele: HII = {unstable_hii:.4f} (unstable)")
        w(f"    - IAS = {result['ias']:.4f} -> strong asymmetric instability")
        w(f"  Only the pathogenic allele shows somatic instability,")
        w(f"  confirming per-haplotype decomposition is essential.")
    elif unstable_hii > 0.45:
        w(f"\n  Elevated instability detected (max HII = {unstable_hii:.4f})")
        w(f"  IAS = {result['ias']:.4f}")
    else:
        w(f"\n  Low instability signal (max HII = {unstable_hii:.4f})")
        w(f"  Possible explanations:")
        w(f"    - Carrier may not show somatic mosaicism at this coverage")
        w(f"    - ONT noise may obscure subtle signals")
        w(f"    - ATXN10 expansions may be somatically stable in blood")

    w(f"\n  ONT-SPECIFIC CONSIDERATIONS")
    w(f"  - ONT basecalling error rate (~5%) is higher than HiFi (<1%)")
    w(f"  - Baseline HII may be elevated compared to HiFi")
    w(f"  - Interpret relative to ONT-specific noise floor, not HiFi 3-sigma")
    w(f"  - Long reads (>10kb) from ONT can span very large expansions")
    w(f"    that HiFi may miss due to shorter read length")

    w(f"\n{'=' * 80}")

    report = "\n".join(lines)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)
    logger.info("Report saved to %s", args.output)

    # Figure: read size distribution by haplotype
    if HAS_MPL:
        fig_dir = outdir / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Size distribution by HP tag
        hp1_sizes = [r.allele_size for r in hp1_reads]
        hp2_sizes = [r.allele_size for r in hp2_reads]
        hp0_sizes = [r.allele_size for r in hp0_reads]

        if hp0_sizes:
            axes[0].hist(hp0_sizes, bins=30, alpha=0.4, label="HP=0",
                         color="gray")
        if hp1_sizes:
            axes[0].hist(hp1_sizes, bins=30, alpha=0.6, label="HP=1",
                         color="#4C72B0")
        if hp2_sizes:
            axes[0].hist(hp2_sizes, bins=30, alpha=0.6, label="HP=2",
                         color="#C44E52")
        axes[0].axvline(result['modal_h1'], color="#4C72B0", ls="--", lw=1)
        axes[0].axvline(result['modal_h2'], color="#C44E52", ls="--", lw=1)
        axes[0].set_xlabel("Allele size (bp)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"ATXN10 Read Size Distribution (n={len(reads)})")
        axes[0].legend()

        # Panel B: Per-haplotype instability bar chart
        metrics = ["HII", "SER", "SCR"]
        h1_vals = [result['hii_h1'], result['ser_h1'], result['scr_h1']]
        h2_vals = [result['hii_h2'], result['ser_h2'], result['scr_h2']]

        x = np.arange(len(metrics))
        width = 0.35
        axes[1].bar(x - width / 2, h1_vals, width, label="H1",
                    color="#4C72B0")
        axes[1].bar(x + width / 2, h2_vals, width, label="H2",
                    color="#C44E52")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metrics)
        axes[1].set_ylabel("Value")
        axes[1].set_title("Per-haplotype Instability Metrics")
        axes[1].legend()

        plt.suptitle(f"ATXN10 Carrier — {args.bam.split('/')[-1]}")
        plt.tight_layout()
        fig_path = fig_dir / "atxn10_instability.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info("Figure saved: %s", fig_path)


# ===========================================================================
# Subcommand: run-all
# ===========================================================================

def cmd_run_all(args):
    """Run complete pipeline: download -> haplotag -> instability."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample = args.sample
    ref = args.ref
    t0 = time.time()

    logger.info("Starting complete pipeline for %s", sample)

    # Step 1: Download
    logger.info("=" * 60)
    logger.info("Step 1/3: Download CRAM")
    logger.info("=" * 60)
    dl_args = argparse.Namespace(
        sample=sample, outdir=str(outdir), dry_run=args.dry_run,
    )
    cram_path = cmd_download(dl_args)

    if args.dry_run:
        logger.info("[DRY RUN] Would continue with haplotag and instability")
        return

    # Step 2: Haplotag
    logger.info("=" * 60)
    logger.info("Step 2/3: Haplotag with whatshap")
    logger.info("=" * 60)
    ht_args = argparse.Namespace(
        bam=cram_path, sample=sample, ref=ref,
        vcf=args.vcf, outdir=str(outdir), dry_run=False,
    )
    tagged_bam = cmd_haplotag(ht_args)

    # Step 3: Instability
    logger.info("=" * 60)
    logger.info("Step 3/3: Run instability analysis")
    logger.info("=" * 60)
    output_report = str(outdir / f"{sample}_atxn10_instability_report.txt")
    inst_args = argparse.Namespace(
        bam=tagged_bam, ref=ref, output=output_report,
    )
    cmd_instability(inst_args)

    elapsed = time.time() - t0
    logger.info("Pipeline complete for %s in %.0fs", sample, elapsed)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="1000G ONT ATXN10 carrier instability analysis pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # find-carrier
    subparsers.add_parser(
        "find-carrier",
        help="Guide for finding ATXN10 carrier from pathSTR",
    )

    # download
    p_dl = subparsers.add_parser(
        "download", help="Download ONT CRAM from 1000G IGSR")
    p_dl.add_argument("--sample", required=True, help="1000G sample ID")
    p_dl.add_argument("--outdir", required=True, help="Output directory")
    p_dl.add_argument("--dry-run", action="store_true",
                      help="Print commands without executing")

    # haplotag
    p_ht = subparsers.add_parser(
        "haplotag", help="Haplotag BAM with whatshap")
    p_ht.add_argument("--bam", required=True, help="Input BAM/CRAM")
    p_ht.add_argument("--sample", required=True, help="Sample ID for VCF")
    p_ht.add_argument("--ref", default=REF_PATH, help="Reference FASTA")
    p_ht.add_argument("--vcf", default=None,
                      help="Phased VCF (auto-downloads 1000G if omitted)")
    p_ht.add_argument("--outdir", required=True, help="Output directory")
    p_ht.add_argument("--dry-run", action="store_true")

    # instability
    p_inst = subparsers.add_parser(
        "instability", help="Run instability on ATXN10 locus")
    p_inst.add_argument("--bam", required=True, help="HP-tagged BAM")
    p_inst.add_argument("--ref", default=REF_PATH, help="Reference FASTA")
    p_inst.add_argument("--output", required=True, help="Output report file")

    # run-all
    p_all = subparsers.add_parser(
        "run-all", help="Complete pipeline (download + haplotag + instability)")
    p_all.add_argument("--sample", required=True, help="1000G sample ID")
    p_all.add_argument("--outdir", required=True, help="Output directory")
    p_all.add_argument("--ref", default=REF_PATH, help="Reference FASTA")
    p_all.add_argument("--vcf", default=None, help="Phased VCF (optional)")
    p_all.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "find-carrier":
        cmd_find_carrier(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "haplotag":
        cmd_haplotag(args)
    elif args.command == "instability":
        cmd_instability(args)
    elif args.command == "run-all":
        cmd_run_all(args)


if __name__ == "__main__":
    main()
