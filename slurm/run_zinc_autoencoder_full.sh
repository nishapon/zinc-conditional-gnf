#!/bin/bash
#SBATCH --job-name=ZINC_molecular_ae
#SBATCH --output=logs/zinc_autoencoder_output.log
#SBATCH --error=logs/zinc_autoencoder_error.log
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
echo "ZINC molecular autoencoder: full dataset"
echo "Start time: $(date)"
echo "=============================================="

srun "${PYTHON}" -u scripts/train_autoencoder.py \
    --config configs/zinc250k.yaml \
    --splits data/processed/zinc250k/zinc_splits.pkl \
    --output-dir outputs/checkpoints/autoencoder \
    --device cuda \
    --epochs 30 \
    --gradient-clip 5.0

echo "=============================================="
echo "ZINC molecular autoencoder training complete"
echo "End time: $(date)"
echo "=============================================="
