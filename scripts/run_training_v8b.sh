#!/bin/bash
# DeepTR v8B — Gradient-Stop TN
#
# Hypothesis: TN allele gradient pollutes backbone representations.
#   Detach combined repr for TN samples before allele_head, so TN
#   allele gradient only updates allele_head (learns to predict 0)
#   but does NOT flow back through backbone/transformer.
#
# All v8A bug fixes included + --tn-allele-detach flag.
#
# Success criterion: TP exact >= 35%, TN exact >= 85%
set -euo pipefail

CONDA_ENV="/blaze/junsoopablo/conda/envs/ensembletr_lr"
PYTHON="${CONDA_ENV}/bin/python"
DEEPTR="${CONDA_ENV}/bin/deeptr"
PROJECT="/qbio/junsoopablo/02_Projects/10_internship/deeptr"
SLURMDIR="${PROJECT}/slurm_jobs"
OUTDIR="${PROJECT}/output/train_v8b"
H5="${PROJECT}/output/train_v1/deeptr_features.h5"

BAM="/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio-48x_BAM_GRCh38/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.bam"
REF="/vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
TRUTH="/vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz"
CATALOG="/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"

mkdir -p "${SLURMDIR}" "${OUTDIR}"

TRAIN_JOB=$(sbatch --parsable \
    --job-name="deeptr_v8b" \
    --output="${SLURMDIR}/deeptr_v8b_%j.out" \
    --error="${SLURMDIR}/deeptr_v8b_%j.err" \
    --time=02:00:00 \
    --mem=64G \
    --cpus-per-task=8 \
    --gpus=l40s:1 \
    --wrap="
        export PATH=${CONDA_ENV}/bin:\${PATH}
        export PYTHONPATH=${PROJECT}:\${PYTHONPATH:-}
        cd ${PROJECT}

        echo '=== DeepTR v8B: Gradient-Stop TN ==='
        echo \"Date: \$(date)\"
        echo \"Host: \$(hostname)\"
        echo 'scheduler=reduce_plateau, tn-allele-detach'
        echo 'tp-loss-weight=5, tp-sample-weight=5, focal-gamma=1.0'

        ${DEEPTR} train \
            --bam ${BAM} \
            --ref ${REF} \
            --truth-bed ${TRUTH} \
            --catalog ${CATALOG} \
            --output-dir ${OUTDIR} \
            --device cuda:0 \
            --threads 8 \
            --epochs 60 \
            --batch-size 512 \
            --lr 0.001 \
            --d-model 64 \
            --n-heads 4 \
            --n-layers 2 \
            --patience 10 \
            --catalog-tolerance 10 \
            --skip-features \
            --features-h5 ${H5} \
            --focal-gamma 1.0 \
            --tp-loss-weight 5.0 \
            --tp-sample-weight 5.0 \
            --scheduler reduce_plateau \
            --tn-allele-detach

        echo '=== Training complete ==='
        echo \"Date: \$(date)\"
        ls -lh ${OUTDIR}/

        echo '=== Eval: threshold=0.0 ==='
        ${PYTHON} ${PROJECT}/scripts/evaluate_test.py \
            --model ${OUTDIR}/deeptr_best.pt \
            --normalizer ${OUTDIR}/normalizer.npz \
            --h5 ${H5} \
            --output-dir ${OUTDIR}/eval_t0

        echo '=== Eval: threshold=0.3 ==='
        ${PYTHON} ${PROJECT}/scripts/evaluate_test.py \
            --model ${OUTDIR}/deeptr_best.pt \
            --normalizer ${OUTDIR}/normalizer.npz \
            --h5 ${H5} \
            --output-dir ${OUTDIR}/eval_t03 \
            --variant-threshold 0.3

        echo '=== Eval: threshold=0.5 ==='
        ${PYTHON} ${PROJECT}/scripts/evaluate_test.py \
            --model ${OUTDIR}/deeptr_best.pt \
            --normalizer ${OUTDIR}/normalizer.npz \
            --h5 ${H5} \
            --output-dir ${OUTDIR}/eval_t05 \
            --variant-threshold 0.5
    ")

echo "v8B gradient-stop TN job submitted: ${TRAIN_JOB}"
echo ""
echo "Monitor:"
echo "  squeue -u junsoopablo"
echo "  tail -f ${SLURMDIR}/deeptr_v8b_${TRAIN_JOB}.out"
