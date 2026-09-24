#!/bin/bash
#SBATCH --job-name=ZINC_flow_smoke
#SBATCH --output=logs/zinc_flow_smoke_output.log
#SBATCH --error=logs/zinc_flow_smoke_error.log
#SBATCH --mail-type=NONE
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

set -euo pipefail

PYTHON="${ZINC_GNF_PYTHON:-${HOME}/miniconda3/envs/zinc_gnf/bin/python}"
WORKDIR="${SLURM_SUBMIT_DIR:-${HOME}/zinc-conditional-gnf}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python executable not found: ${PYTHON}" >&2
    echo "Set ZINC_GNF_PYTHON to the zinc_gnf environment Python." >&2
    exit 1
fi

if [[ ! -f "${WORKDIR}/pyproject.toml" ]]; then
    echo "ERROR: Repository not found at ${WORKDIR}" >&2
    echo "Submit this job from the repository root." >&2
    exit 1
fi

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

export WANDB_MODE=offline

echo "=============================================="
echo "ZINC conditional GNF: smoke training"
echo "Start time: $(date)"
echo "=============================================="

srun "${PYTHON}" -u scripts/train_flow.py \
    --config configs/zinc250k.yaml \
    --embeddings outputs/embeddings/zinc250k \
    --output-dir outputs/checkpoints/conditional_flow_smoke \
    --device cuda \
    --epochs 2 \
    --batch-size 32 \
    --max-train-batches 100 \
    --max-validation-batches 20 \
    --learning-rate 0.0001 \
    --gradient-clip 5.0

echo "=============================================="
echo "ZINC conditional GNF smoke training complete"
echo "End time: $(date)"
echo "=============================================="
