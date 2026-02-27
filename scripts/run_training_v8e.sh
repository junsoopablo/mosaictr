#!/bin/bash
# DeepTR v8E — Larger Model on TP-Only
#
# Hypothesis: TP-only training benefits from a larger model because
#   the ~93K TP loci have much more signal variety than the mostly-zero
#   TN loci. d_model=128, n_layers=3 provides more capacity to learn
#   complex allele size patterns.
#
# Key settings:
#   --tp-only: Filter to TP loci only, plain shuffle DataLoader
#   d_model=128, n_layers=3, n_heads=8 (head_dim=16)
#   lr=5e-4, batch=256, dropout=0.15, weight_decay=5e-4
#   focal-gamma=2.0 (stronger focus on hard examples)
#
# Evaluation: Two-model inference (v8C classifier + v8E regressor)
# Success criterion: TP exact >= 50%
set -euo pipefail

CONDA_ENV="/blaze/junsoopablo/conda/envs/ensembletr_lr"
PYTHON="${CONDA_ENV}/bin/python"
DEEPTR="${CONDA_ENV}/bin/deeptr"
PROJECT="/qbio/junsoopablo/02_Projects/10_internship/deeptr"
SLURMDIR="${PROJECT}/slurm_jobs"
OUTDIR="${PROJECT}/output/train_v8e"
H5="${PROJECT}/output/train_v1/deeptr_features.h5"

BAM="/vault/external-datasets/2026/HG002_PacBio-HiFi-Revio-48x_BAM_GRCh38/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.bam"
REF="/vault/external-datasets/2026/GRCh38_no-alt-analysis_reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
TRUTH="/vault/external-datasets/2026/GIAB_TR-benchmark-v1.0.1_HG002_GRCh38/HG002_GRCh38_TandemRepeats_v1.0.1_Tier1.bed.gz"
CATALOG="/vault/external-datasets/2026/adotto-TR-catalog-v1.2/adotto_v1.2_longtr_v1.2_format.bed"

# v8C output for classifier
V8C_DIR="${PROJECT}/output/train_v8c"

mkdir -p "${SLURMDIR}" "${OUTDIR}"

TRAIN_JOB=$(sbatch --parsable \
    --job-name="deeptr_v8e" \
    --output="${SLURMDIR}/deeptr_v8e_%j.out" \
    --error="${SLURMDIR}/deeptr_v8e_%j.err" \
    --time=03:00:00 \
    --mem=64G \
    --cpus-per-task=8 \
    --gpus=l40s:1 \
    --wrap="
        export PATH=${CONDA_ENV}/bin:\${PATH}
        export PYTHONPATH=${PROJECT}:\${PYTHONPATH:-}
        cd ${PROJECT}

        echo '=== DeepTR v8E: Larger Model on TP-Only ==='
        echo \"Date: \$(date)\"
        echo \"Host: \$(hostname)\"
        echo 'd_model=128, n_layers=3, n_heads=8, lr=5e-4, dropout=0.15'
        echo 'weight_decay=5e-4, focal-gamma=2.0, batch=256'

        ${DEEPTR} train \
            --bam ${BAM} \
            --ref ${REF} \
            --truth-bed ${TRUTH} \
            --catalog ${CATALOG} \
            --output-dir ${OUTDIR} \
            --device cuda:0 \
            --threads 8 \
            --epochs 60 \
            --batch-size 256 \
            --lr 0.0005 \
            --d-model 128 \
            --n-heads 8 \
            --n-layers 3 \
            --patience 10 \
            --catalog-tolerance 10 \
            --skip-features \
            --features-h5 ${H5} \
            --focal-gamma 2.0 \
            --tp-loss-weight 1.0 \
            --tp-sample-weight 1.0 \
            --variant-pos-weight 0 \
            --scheduler reduce_plateau \
            --tp-only \
            --weight-decay 0.0005

        echo '=== Training complete ==='
        echo \"Date: \$(date)\"
        ls -lh ${OUTDIR}/

        # Standalone eval (TP-only model, threshold=0)
        echo '=== Eval: standalone, threshold=0.0 ==='
        ${PYTHON} ${PROJECT}/scripts/evaluate_test.py \
            --model ${OUTDIR}/deeptr_best.pt \
            --normalizer ${OUTDIR}/normalizer.npz \
            --h5 ${H5} \
            --output-dir ${OUTDIR}/eval_standalone

        # Two-model eval (v8C classifier + v8E regressor)
        if [ -f '${V8C_DIR}/deeptr_stage1.pt' ] && [ -f '${V8C_DIR}/normalizer.npz' ]; then
            echo '=== Eval: two-model (v8C cls + v8E reg), threshold=0.5 ==='
            ${PYTHON} ${PROJECT}/scripts/evaluate_test_twomodel.py \
                --classifier-model ${V8C_DIR}/deeptr_stage1.pt \
                --regressor-model ${OUTDIR}/deeptr_best.pt \
                --classifier-normalizer ${V8C_DIR}/normalizer.npz \
                --regressor-normalizer ${OUTDIR}/normalizer.npz \
                --h5 ${H5} \
                --output-dir ${OUTDIR}/eval_twomodel_t05 \
                --variant-threshold 0.5

            echo '=== Eval: two-model (v8C cls + v8E reg), threshold=0.3 ==='
            ${PYTHON} ${PROJECT}/scripts/evaluate_test_twomodel.py \
                --classifier-model ${V8C_DIR}/deeptr_stage1.pt \
                --regressor-model ${OUTDIR}/deeptr_best.pt \
                --classifier-normalizer ${V8C_DIR}/normalizer.npz \
                --regressor-normalizer ${OUTDIR}/normalizer.npz \
                --h5 ${H5} \
                --output-dir ${OUTDIR}/eval_twomodel_t03 \
                --variant-threshold 0.3
        else
            echo 'WARNING: v8C classifier not found, skipping two-model eval'
        fi
    ")

echo "v8E larger model TP-only job submitted: ${TRAIN_JOB}"
echo ""
echo "Monitor:"
echo "  squeue -u junsoopablo"
echo "  tail -f ${SLURMDIR}/deeptr_v8e_${TRAIN_JOB}.err"
