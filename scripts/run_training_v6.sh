#!/bin/bash
# DeepTR v6 — Step A: Aggressive Reweighting
#
# Hypothesis: current effective TP weight (~10x) is too weak.
#   tp_loss_weight: 2.0 → 10.0
#   tp_sample_weight: 5.0 → 15.0
#   Effective TP batch fraction ~50%, effective loss weight ~83%
#
# Success criterion: TP exact match >= 45% (from baseline 20.2%)
# Uses existing HDF5 from train_v1 (no re-extraction needed)
set -euo pipefail

CONDA_ENV="/blaze/junsoopablo/conda/envs/ensembletr_lr"
PYTHON="${CONDA_ENV}/bin/python"
DEEPTR="${CONDA_ENV}/bin/deeptr"
PROJECT="/qbio/junsoopablo/02_Projects/10_internship/deeptr"
SLURMDIR="${PROJECT}/slurm_jobs"
OUTDIR="${PROJECT}/output/train_v6"
H5="${PROJECT}/output/train_v1/deeptr_features.h5"

BAM="/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio-48x_BAM_GRCh38/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.bam"
REF="/vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
TRUTH="/vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz"
CATALOG="/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"

mkdir -p "${SLURMDIR}" "${OUTDIR}"

# ---------------------------------------------------------------------------
# Training — aggressive reweighting only (config change, no code change)
# ---------------------------------------------------------------------------
TRAIN_JOB=$(sbatch --parsable \
    --job-name="deeptr_v6_stepA" \
    --output="${SLURMDIR}/deeptr_v6_stepA_%j.out" \
    --error="${SLURMDIR}/deeptr_v6_stepA_%j.err" \
    --time=02:00:00 \
    --mem=64G \
    --cpus-per-task=8 \
    --gpus=l40s:1 \
    --wrap="
        export PATH=${CONDA_ENV}/bin:\${PATH}
        export PYTHONPATH=${PROJECT}:\${PYTHONPATH:-}
        cd ${PROJECT}

        echo '=== DeepTR v6: Step A — Aggressive Reweighting ==='
        echo \"Date: \$(date)\"
        echo \"Host: \$(hostname)\"
        echo 'tp_loss_weight=10.0, tp_sample_weight=15.0'

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
            --tp-loss-weight 10.0 \
            --tp-sample-weight 15.0

        echo '=== Step A training complete ==='
        echo \"Date: \$(date)\"
        ls -lh ${OUTDIR}/

        # Evaluate on test set
        echo '=== Evaluating on test set ==='
        ${PYTHON} ${PROJECT}/scripts/evaluate_test.py \
            --model ${OUTDIR}/deeptr_best.pt \
            --normalizer ${OUTDIR}/normalizer.npz \
            --h5 ${H5} \
            --output-dir ${OUTDIR}/eval_test
    ")

echo "Step A training job submitted: ${TRAIN_JOB}"
echo ""
echo "Monitor:"
echo "  squeue -u junsoopablo"
echo "  tail -f ${SLURMDIR}/deeptr_v6_stepA_${TRAIN_JOB}.out"
