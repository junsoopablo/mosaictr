#!/bin/bash
# Run all analyses for the HaploTR Application Note.
# Prerequisites: genome-wide HaploTR results for HG002, HG003, HG004.
#
# Usage:
#   bash scripts/run_all_analyses.sh

set -euo pipefail

PROJ="/qbio/junsoopablo/02_Projects/10_internship/deeptr"
OUTDIR="$PROJ/output/genome_wide"
FIGDIR="$OUTDIR/figures"

HG002_BED="$OUTDIR/v4_hg002_genome_wide.bed"
HG003_BED="$OUTDIR/v4_hg003_genome_wide.bed"
HG004_BED="$OUTDIR/v4_hg004_genome_wide.bed"

LONGTR_VCF="/qbio/junsoopablo/02_Projects/10_internship/ensembletr-lr/results/HG002.longtr.vcf.gz"
TRGT_VCF="/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio_TRGT-VCF_GRCh38/HG002.GRCh38.trgt.sorted.phased.vcf.gz"

cd "$PROJ"

echo "============================================"
echo "  HaploTR Application Note — Full Analysis"
echo "============================================"

# 1. GIAB Benchmark
echo ""
echo "[1/6] Running GIAB Tier1 benchmark..."
python scripts/benchmark_genome_wide.py \
    --haplotr "$HG002_BED" \
    --longtr-vcf "$LONGTR_VCF" \
    --trgt-vcf "$TRGT_VCF" \
    --output "$OUTDIR/benchmark_genome_wide_report.txt"

# 2. Mendelian inheritance
echo ""
echo "[2/6] Running Mendelian inheritance analysis..."
python scripts/analyze_mendelian.py \
    --child "$HG002_BED" \
    --father "$HG003_BED" \
    --mother "$HG004_BED" \
    --output "$OUTDIR/mendelian_report.txt"

# 3. Disease loci
echo ""
echo "[3/6] Running disease loci analysis..."
python scripts/analyze_disease_loci.py \
    --haplotr "$HG002_BED" \
    --longtr-vcf "$LONGTR_VCF" \
    --trgt-vcf "$TRGT_VCF" \
    --output "$OUTDIR/disease_loci_report.txt"

# 4. Allele delta analysis
echo ""
echo "[4/6] Running allele delta analysis..."
python scripts/analyze_allele_delta.py \
    --haplotr "$HG002_BED" \
    --longtr-vcf "$LONGTR_VCF" \
    --trgt-vcf "$TRGT_VCF" \
    --output "$OUTDIR/allele_delta_report.txt"

# 5. Confidence calibration
echo ""
echo "[5/6] Running confidence calibration analysis..."
python scripts/analyze_confidence.py \
    --haplotr "$HG002_BED" \
    --output "$OUTDIR/confidence_report.txt" \
    --output-tsv "$OUTDIR/confidence_curve.tsv"

# 6. Figures
echo ""
echo "[6/6] Generating publication figures..."
mkdir -p "$FIGDIR"
python scripts/generate_figures.py \
    --haplotr "$HG002_BED" \
    --longtr-vcf "$LONGTR_VCF" \
    --trgt-vcf "$TRGT_VCF" \
    --output-dir "$FIGDIR"

echo ""
echo "============================================"
echo "  All analyses complete!"
echo "  Reports: $OUTDIR/"
echo "  Figures: $FIGDIR/"
echo "============================================"
echo ""
echo "Output files:"
ls -lh "$OUTDIR"/*.txt "$OUTDIR"/*.tsv "$FIGDIR"/*.pdf 2>/dev/null
