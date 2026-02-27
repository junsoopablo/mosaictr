#!/bin/bash
# DeepTR v2 training — uses pre-computed features, only retrains model
set -euo pipefail

CONDA_ENV="/blaze/junsoopablo/conda/envs/ensembletr_lr"
PYTHON="${CONDA_ENV}/bin/python"
DEEPTR="${CONDA_ENV}/bin/deeptr"
PROJECT="/qbio/junsoopablo/02_Projects/10_internship/deeptr"
SLURMDIR="${PROJECT}/slurm_jobs"
OUTDIR="${PROJECT}/output/train_v2"
H5="${PROJECT}/output/train_v1/deeptr_features.h5"

mkdir -p "${SLURMDIR}" "${OUTDIR}"

JOBID=$(sbatch --parsable \
    --job-name="deeptr_v2" \
    --output="${SLURMDIR}/deeptr_v2_%j.out" \
    --error="${SLURMDIR}/deeptr_v2_%j.err" \
    --time=02:00:00 \
    --mem=64G \
    --cpus-per-task=8 \
    --gpus=l40s:1 \
    --wrap="
        export PATH=${CONDA_ENV}/bin:\${PATH}
        export PYTHONPATH=${PROJECT}:\${PYTHONPATH:-}
        cd ${PROJECT}

        echo '=== Starting DeepTR v2 training ==='
        echo \"Date: \$(date)\"
        echo \"Host: \$(hostname)\"

        ${DEEPTR} train \
            --bam /vault/external-datasets/2026/HG002_PacBio-HiFi-Revio-48x_BAM_GRCh38/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.bam \
            --ref /vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta \
            --truth-bed /vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz \
            --catalog /vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed \
            --output-dir ${OUTDIR} \
            --device cuda:0 \
            --threads 8 \
            --epochs 60 \
            --batch-size 512 \
            --lr 0.001 \
            --d-model 64 \
            --n-heads 4 \
            --n-layers 2 \
            --patience 7 \
            --catalog-tolerance 10 \
            --skip-features \
            --features-h5 ${H5}

        echo '=== Training v2 complete ==='
        echo \"Date: \$(date)\"
        ls -lh ${OUTDIR}/
    ")

echo "Job submitted: ${JOBID}"
echo "Monitor: squeue -u junsoopablo"
