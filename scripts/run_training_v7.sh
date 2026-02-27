#!/bin/bash
# DeepTR v7 — Step B: TP-Only Allele Loss + Balanced Variant BCE
#
# Hypothesis: TN gradient pollution is the root cause. Removing TN from
#   allele loss lets the head focus on TP magnitude.
#
# Changes vs Step A:
#   --tp-only-allele-loss: allele loss computed only on TP samples
#   --variant-pos-weight 15.0: BCEWithLogitsLoss with pos_weight for variant head
#   --tp-batch-fraction 0.3: StratifiedBatchSampler (30% TP per batch)
#
# Success criterion: TP exact match >= 55%
# Uses existing HDF5 from train_v1
set -euo pipefail

CONDA_ENV="/blaze/junsoopablo/conda/envs/ensembletr_lr"
PYTHON="${CONDA_ENV}/bin/python"
DEEPTR="${CONDA_ENV}/bin/deeptr"
PROJECT="/qbio/junsoopablo/02_Projects/10_internship/deeptr"
SLURMDIR="${PROJECT}/slurm_jobs"
OUTDIR="${PROJECT}/output/train_v7"
H5="${PROJECT}/output/train_v1/deeptr_features.h5"

BAM="/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio-48x_BAM_GRCh38/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.bam"
REF="/vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
TRUTH="/vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz"
CATALOG="/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"

mkdir -p "${SLURMDIR}" "${OUTDIR}"

# ---------------------------------------------------------------------------
# Training — TP-only allele loss + balanced variant BCE + stratified sampler
# ---------------------------------------------------------------------------
TRAIN_JOB=$(sbatch --parsable \
    --job-name="deeptr_v7_stepB" \
    --output="${SLURMDIR}/deeptr_v7_stepB_%j.out" \
    --error="${SLURMDIR}/deeptr_v7_stepB_%j.err" \
    --time=02:00:00 \
    --mem=64G \
    --cpus-per-task=8 \
    --gpus=l40s:1 \
    --wrap="
        export PATH=${CONDA_ENV}/bin:\${PATH}
        export PYTHONPATH=${PROJECT}:\${PYTHONPATH:-}
        cd ${PROJECT}

        echo '=== DeepTR v7: Step B — TP-Only Allele Loss ==='
        echo \"Date: \$(date)\"
        echo \"Host: \$(hostname)\"
        echo 'tp-only-allele-loss, variant-pos-weight=15.0, tp-batch-fraction=0.3'

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
            --patience 7 \
            --catalog-tolerance 10 \
            --skip-features \
            --features-h5 ${H5} \
            --focal-gamma 2.0 \
            --tp-only-allele-loss \
            --variant-pos-weight 15.0 \
            --tp-batch-fraction 0.3

        echo '=== Step B training complete ==='
        echo \"Date: \$(date)\"
        ls -lh ${OUTDIR}/

        # Evaluate on test set (with soft-gating threshold=0.3)
        echo '=== Evaluating on test set (threshold=0.3) ==='
        ${PYTHON} ${PROJECT}/scripts/evaluate_test.py \
            --model ${OUTDIR}/deeptr_best.pt \
            --normalizer ${OUTDIR}/normalizer.npz \
            --h5 ${H5} \
            --output-dir ${OUTDIR}/eval_test \
            --variant-threshold 0.3

        # Also evaluate without soft-gating for comparison
        echo '=== Evaluating on test set (no threshold) ==='
        ${PYTHON} ${PROJECT}/scripts/evaluate_test.py \
            --model ${OUTDIR}/deeptr_best.pt \
            --normalizer ${OUTDIR}/normalizer.npz \
            --h5 ${H5} \
            --output-dir ${OUTDIR}/eval_test_nothresh
    ")

echo "Step B training job submitted: ${TRAIN_JOB}"
echo ""
echo "Monitor:"
echo "  squeue -u junsoopablo"
echo "  tail -f ${SLURMDIR}/deeptr_v7_stepB_${TRAIN_JOB}.out"
