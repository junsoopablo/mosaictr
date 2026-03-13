#!/bin/bash
# Run genome-wide MosaicTR genotyping for HG002/HG003/HG004 trio.
#
# Run HG002 first (can start immediately), then HG003/HG004 after downloads complete.
# Each takes ~5-7 hours with nprocs=1.
#
# Usage:
#   bash scripts/run_genome_wide_trio.sh [hg002|hg003|hg004|all]

set -euo pipefail

PROJ="/qbio/junsoopablo/02_Projects/10_internship/deeptr"
OUTDIR="$PROJ/output/genome_wide"
CATALOG="/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"

HG002_BAM="/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio-48x_BAM_GRCh38/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.bam"
HG003_BAM="/vault/external-datasets/2026/HG003_PacBio-HiFi-Revio_BAM_GRCh38/GRCh38.m84039_241002_000337_s3.hifi_reads.bc2020.bam"
HG004_BAM="/vault/external-datasets/2026/HG004_PacBio-HiFi-Revio_BAM_GRCh38/GRCh38.m84039_241002_020632_s4.hifi_reads.bc2021.bam"

mkdir -p "$OUTDIR"
cd "$PROJ"

run_sample() {
    local sample=$1
    local bam=$2
    local output=$3

    echo "============================================"
    echo "  Running MosaicTR genome-wide: $sample"
    echo "  BAM: $bam"
    echo "  Output: $output"
    echo "============================================"

    if [ ! -f "$bam" ]; then
        echo "ERROR: BAM not found: $bam"
        return 1
    fi

    if [ -f "$output" ]; then
        n=$(wc -l < "$output")
        echo "Output already exists ($n lines). Skipping."
        return 0
    fi

    time python -c "
from mosaictr.genotype import genotype
genotype(
    bam_path='$bam',
    loci_bed_path='$CATALOG',
    output_path='$output',
    nprocs=1,
    chunk_size=500,
)
"
    echo "Done: $output"
    wc -l "$output"
}

case "${1:-all}" in
    hg002)
        run_sample "HG002" "$HG002_BAM" "$OUTDIR/v4_hg002_genome_wide.bed"
        ;;
    hg003)
        run_sample "HG003" "$HG003_BAM" "$OUTDIR/v4_hg003_genome_wide.bed"
        ;;
    hg004)
        run_sample "HG004" "$HG004_BAM" "$OUTDIR/v4_hg004_genome_wide.bed"
        ;;
    all)
        # Run all three sequentially (or use & for parallel if enough RAM)
        run_sample "HG002" "$HG002_BAM" "$OUTDIR/v4_hg002_genome_wide.bed"
        run_sample "HG003" "$HG003_BAM" "$OUTDIR/v4_hg003_genome_wide.bed"
        run_sample "HG004" "$HG004_BAM" "$OUTDIR/v4_hg004_genome_wide.bed"
        ;;
    *)
        echo "Usage: $0 [hg002|hg003|hg004|all]"
        exit 1
        ;;
esac

echo ""
echo "All requested genotyping complete."
ls -lh "$OUTDIR"/*.bed 2>/dev/null
