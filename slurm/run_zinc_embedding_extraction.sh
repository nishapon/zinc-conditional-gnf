#!/bin/bash
#SBATCH --job-name=ZINC_embeddings
#SBATCH --output=logs/zinc_embeddings_output.log
#SBATCH --error=logs/zinc_embeddings_error.log
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
echo "ZINC property-aware embedding extraction"
echo "Start time: $(date)"
echo "=============================================="

srun "${PYTHON}" -u scripts/extract_embeddings.py \
    --config configs/zinc250k.yaml \
    --splits data/processed/zinc250k/zinc_splits.pkl \
    --checkpoint outputs/checkpoints/property_aware_autoencoder/best.pt \
    --output-dir outputs/embeddings/zinc250k \
    --device cuda \
    --batch-size 32 \
    --shard-size 2048 \
    --noise-seed 42 \
    --overwrite

echo "=============================================="
echo "ZINC embedding extraction complete"
echo "End time: $(date)"
echo "=============================================="
