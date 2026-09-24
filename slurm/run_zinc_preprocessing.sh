#!/bin/bash
#SBATCH --job-name=ZINC_preprocess
#SBATCH --output=logs/zinc_preprocess_output.log
#SBATCH --error=logs/zinc_preprocess_error.log
#SBATCH --mail-type=NONE
#SBATCH --partition=STUD

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
echo "ZINC250K preprocessing"
echo "Start time: $(date)"
echo "=============================================="

srun "${PYTHON}" -u scripts/preprocess_zinc.py \
    --csv data/raw/zinc250k.csv \
    --output-dir data/processed/zinc250k \
    --target-molecules 0 \
    --candidate-count 0 \
    --seed 42 \
    --max-nodes 38 \
    --train-fraction 0.80 \
    --validation-fraction 0.10

echo "=============================================="
echo "ZINC250K preprocessing complete"
echo "End time: $(date)"
echo "=============================================="
