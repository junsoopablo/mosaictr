#!/bin/bash
# DeepTR v8C — Two-Stage Training
#
# Hypothesis: Classification and regression tasks interfere when trained
#   jointly. Separating them maximizes TP allele prediction accuracy.
#
# Stage 1 (~20 epochs): Train backbone + variant_head on full data.
#   Only variant loss, StratifiedBatchSampler(tp_fraction=0.3).
# Stage 2 (~40 epochs): Freeze backbone, train allele_head on TP-only data.
#   FocalSmoothL1Loss(gamma=1.0), lr=3e-4.
#
# Inference: variant_prob > threshold → Stage 2 allele prediction, else 0.
#
# Success criterion: TP exact >= 45%, TN exact >= 93%
set -euo pipefail

CONDA_ENV="/blaze/junsoopablo/conda/envs/ensembletr_lr"
PYTHON="${CONDA_ENV}/bin/python"
DEEPTR="${CONDA_ENV}/bin/deeptr"
PROJECT="/qbio/junsoopablo/02_Projects/10_internship/deeptr"
SLURMDIR="${PROJECT}/slurm_jobs"
OUTDIR="${PROJECT}/output/train_v8c"
H5="${PROJECT}/output/train_v1/deeptr_features.h5"

BAM="/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio-48x_BAM_GRCh38/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.bam"
REF="/vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
TRUTH="/vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz"
CATALOG="/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"

mkdir -p "${SLURMDIR}" "${OUTDIR}"

TRAIN_JOB=$(sbatch --parsable \
    --job-name="deeptr_v8c" \
    --output="${SLURMDIR}/deeptr_v8c_%j.out" \
    --error="${SLURMDIR}/deeptr_v8c_%j.err" \
    --time=02:30:00 \
    --mem=64G \
    --cpus-per-task=8 \
    --gpus=l40s:1 \
    --wrap="
        export PATH=${CONDA_ENV}/bin:\${PATH}
        export PYTHONPATH=${PROJECT}:\${PYTHONPATH:-}
        cd ${PROJECT}

        echo '=== DeepTR v8C: Two-Stage Training ==='
        echo \"Date: \$(date)\"
        echo \"Host: \$(hostname)\"
        echo 'Stage 1: variant classifier (20 epochs)'
        echo 'Stage 2: allele regressor on TP-only (40 epochs, backbone frozen)'

        ${DEEPTR} train-two-stage \
            --bam ${BAM} \
            --ref ${REF} \
            --truth-bed ${TRUTH} \
            --catalog ${CATALOG} \
            --output-dir ${OUTDIR} \
            --device cuda:0 \
            --threads 8 \
            --batch-size 512 \
            --d-model 64 \
            --n-heads 4 \
            --n-layers 2 \
            --catalog-tolerance 10 \
            --skip-features \
            --features-h5 ${H5} \
            --stage1-lr 0.001 \
            --stage1-epochs 20 \
            --stage1-patience 7 \
            --stage2-lr 0.0003 \
            --stage2-epochs 40 \
            --stage2-patience 10 \
            --focal-gamma 1.0 \
            --tp-batch-fraction 0.3

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

echo "v8C two-stage job submitted: ${TRAIN_JOB}"
echo ""
echo "Monitor:"
echo "  squeue -u junsoopablo"
echo "  tail -f ${SLURMDIR}/deeptr_v8c_${TRAIN_JOB}.out"
