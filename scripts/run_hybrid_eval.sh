#!/bin/bash
# Hybrid evaluation: DL classifier + classical genotyper
#
# Runs 4 evaluations:
# 1. Oracle gating (GT labels) + GMM → upper bound with perfect classification
# 2. Oracle gating (GT labels) + split-median → oracle ceiling
# 3. v8C classifier (t=0.5) + GMM → hybrid approach
# 4. v8C classifier (t=0.3) + GMM → hybrid approach (lower threshold)
set -euo pipefail

CONDA_ENV="/blaze/junsoopablo/conda/envs/ensembletr_lr"
PYTHON="${CONDA_ENV}/bin/python"
PROJECT="/qbio/junsoopablo/02_Projects/10_internship/deeptr"
SLURMDIR="${PROJECT}/slurm_jobs"
H5="${PROJECT}/output/train_v1/deeptr_features.h5"

V8C_DIR="${PROJECT}/output/train_v8c"
OUTDIR="${PROJECT}/output/hybrid_eval"

mkdir -p "${SLURMDIR}" "${OUTDIR}"

EVAL_JOB=$(sbatch --parsable \
    --job-name="hybrid_eval" \
    --output="${SLURMDIR}/hybrid_eval_%j.out" \
    --error="${SLURMDIR}/hybrid_eval_%j.err" \
    --time=01:00:00 \
    --mem=64G \
    --cpus-per-task=8 \
    --gpus=l40s:1 \
    --wrap="
        export PATH=${CONDA_ENV}/bin:\${PATH}
        export PYTHONPATH=${PROJECT}:/qbio/junsoopablo/02_Projects/10_internship/ensembletr-lr:\${PYTHONPATH:-}
        cd ${PROJECT}

        echo '=== 1. Oracle gating + both methods ==='
        echo \"Date: \$(date)\"
        ${PYTHON} ${PROJECT}/scripts/evaluate_hybrid.py \
            --h5 ${H5} \
            --output-dir ${OUTDIR}/oracle \
            --method both

        echo '=== 2. v8C classifier (t=0.5) + GMM ==='
        echo \"Date: \$(date)\"
        ${PYTHON} ${PROJECT}/scripts/evaluate_hybrid.py \
            --classifier-model ${V8C_DIR}/deeptr_stage1.pt \
            --classifier-normalizer ${V8C_DIR}/normalizer.npz \
            --h5 ${H5} \
            --output-dir ${OUTDIR}/v8c_t05 \
            --variant-threshold 0.5 \
            --method gmm

        echo '=== 3. v8C classifier (t=0.3) + GMM ==='
        echo \"Date: \$(date)\"
        ${PYTHON} ${PROJECT}/scripts/evaluate_hybrid.py \
            --classifier-model ${V8C_DIR}/deeptr_stage1.pt \
            --classifier-normalizer ${V8C_DIR}/normalizer.npz \
            --h5 ${H5} \
            --output-dir ${OUTDIR}/v8c_t03 \
            --variant-threshold 0.3 \
            --method gmm

        echo '=== 4. v8C classifier (t=0.5) + split_median ==='
        echo \"Date: \$(date)\"
        ${PYTHON} ${PROJECT}/scripts/evaluate_hybrid.py \
            --classifier-model ${V8C_DIR}/deeptr_stage1.pt \
            --classifier-normalizer ${V8C_DIR}/normalizer.npz \
            --h5 ${H5} \
            --output-dir ${OUTDIR}/v8c_t05_splitmed \
            --variant-threshold 0.5 \
            --method split_median

        echo '=== All evaluations complete ==='
        echo \"Date: \$(date)\"
    ")

echo "Hybrid evaluation job submitted: ${EVAL_JOB}"
echo ""
echo "Monitor:"
echo "  tail -f ${SLURMDIR}/hybrid_eval_${EVAL_JOB}.err"
