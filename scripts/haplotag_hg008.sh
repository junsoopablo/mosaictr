#!/bin/bash
# Haplotag all 4 HG008 BAMs using the normal's phased VCF.
# Usage: bash scripts/haplotag_hg008.sh [sample]
# If no sample given, runs all 4 sequentially.

set -euo pipefail

WHATSHAP="/blaze/junsoopablo/conda/envs/claudecode/bin/whatshap"
SAMTOOLS="/blaze/junsoopablo/conda/envs/bioinfo3/bin/samtools"
REF="/vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
VCF="/vault/external-datasets/2026/HG008_phased_VCF/HG008-N-P.GRCh38.deepvariant.phased.stdchr.vcf.gz"
OUTDIR="/vault/external-datasets/2026/HG008_haplotagged"

mkdir -p "$OUTDIR"

declare -A BAMS
BAMS[normal]="/vault/external-datasets/2026/HG008_PacBio-HiFi-Revio_GRCh38-GIABv3_TumorNormal/GRCh38_GIABv3.HG008-N-P.Revio.hifi_reads.bam"
BAMS[p21]="/vault/external-datasets/2026/HG008-T_PacBio-HiFi-Revio_p21-p41_GRCh38-GIABv3_PassageDrift/HG008-T_NIST-bulk-20240508p21_PacBio-HiFi-Revio_20241011_43x_GRCh38-GIABv3.bam"
BAMS[p23]="/vault/external-datasets/2026/HG008_PacBio-HiFi-Revio_GRCh38-GIABv3_TumorNormal/GRCh38_GIABv3.HG008-T.Revio.hifi_reads.bam"
BAMS[p41]="/vault/external-datasets/2026/HG008-T_PacBio-HiFi-Revio_p21-p41_GRCh38-GIABv3_PassageDrift/HG008-T_NIST-bulk-20240508p41_PacBio-HiFi-Revio_20241011_43x_GRCh38-GIABv3.bam"

haplotag_sample() {
    local sample=$1
    local bam=${BAMS[$sample]}
    local out="$OUTDIR/HG008_${sample}_haplotagged.bam"

    if [ -f "$out" ] && [ -f "${out}.bai" ]; then
        echo "[$(date)] SKIP $sample (already exists: $out)"
        return
    fi

    echo "[$(date)] START haplotagging: $sample"
    echo "  BAM: $bam"
    echo "  OUT: $out"

    $WHATSHAP haplotag \
        --reference "$REF" \
        --output "$out" \
        --ignore-read-groups \
        --skip-missing-contigs \
        "$VCF" "$bam" \
        2>&1 | tail -5

    echo "[$(date)] Indexing: $out"
    $SAMTOOLS index "$out"

    echo "[$(date)] DONE $sample"

    # Check HP tag rate
    local total=$($SAMTOOLS view -c "$out" | head -1)
    local tagged=$($SAMTOOLS view "$out" | head -100000 | grep -c "HP:i:" || true)
    echo "  HP tag rate (first 100K reads): $tagged / 100000"
}

if [ $# -gt 0 ]; then
    haplotag_sample "$1"
else
    for sample in normal p21 p23 p41; do
        haplotag_sample "$sample"
    done
fi
