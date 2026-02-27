#!/bin/bash
# DeepTR training pipeline via SLURM
# Step 1: Feature extraction (CPU, ~1 hour for 1.6M loci)
# Step 2: Model training (GPU, ~30 min)
# Usage: bash scripts/run_training.sh
set -euo pipefail

CONDA_ENV="/blaze/junsoopablo/conda/envs/ensembletr_lr"
PYTHON="${CONDA_ENV}/bin/python"
DEEPTR="${CONDA_ENV}/bin/deeptr"
PROJECT="/qbio/junsoopablo/02_Projects/10_internship/deeptr"
SLURMDIR="${PROJECT}/slurm_jobs"
OUTDIR="${PROJECT}/output/train_v1"

BAM="/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio-48x_BAM_GRCh38/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.bam"
REF="/vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
TIER1_BED="/vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz"
CATALOG="/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"

mkdir -p "${SLURMDIR}" "${OUTDIR}"

echo "=== DeepTR Training Pipeline ==="
echo "  BAM: ${BAM}"
echo "  Truth: ${TIER1_BED}"
echo "  Catalog: ${CATALOG}"
echo "  Output: ${OUTDIR}"

# Single job: feature extraction + training
# Request GPU for the training phase; feature extraction uses CPUs
JOBID=$(sbatch --parsable \
    --job-name="deeptr_train" \
    --output="${SLURMDIR}/deeptr_train_%j.out" \
    --error="${SLURMDIR}/deeptr_train_%j.err" \
    --time=04:00:00 \
    --mem=64G \
    --cpus-per-task=8 \
    --gpus=l40s:1 \
    --wrap="
        export PATH=${CONDA_ENV}/bin:\${PATH}
        export PYTHONPATH=${PROJECT}:\${PYTHONPATH:-}
        cd ${PROJECT}

        echo '=== Starting DeepTR training pipeline ==='
        echo \"Date: \$(date)\"
        echo \"Host: \$(hostname)\"
        echo \"GPUs: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')\"

        ${DEEPTR} train \
            --bam ${BAM} \
            --ref ${REF} \
            --truth-bed ${TIER1_BED} \
            --catalog ${CATALOG} \
            --output-dir ${OUTDIR} \
            --device cuda:0 \
            --threads 8 \
            --epochs 50 \
            --batch-size 512 \
            --lr 0.001 \
            --d-model 64 \
            --n-heads 4 \
            --n-layers 2 \
            --patience 5 \
            --catalog-tolerance 10

        echo '=== Training complete ==='
        echo \"Date: \$(date)\"
        ls -lh ${OUTDIR}/
    ")

echo "Job submitted: ${JOBID}"
echo "Monitor: squeue -u junsoopablo"
echo "Logs: ${SLURMDIR}/deeptr_train_${JOBID}.out"
echo ""
echo "Expected timeline:"
echo "  Feature extraction: ~60-90 min (1.6M loci, 8 procs)"
echo "  Model training:     ~30 min (50 epochs, early stopping)"
echo "  Total:              ~2 hours"
