#!/bin/bash
#SBATCH --job-name=ZINC_condition_check
#SBATCH --output=logs/zinc_condition_check_output.log
#SBATCH --error=logs/zinc_condition_check_error.log
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

echo "=============================================="
echo "ZINC QED and node-count distribution analysis"
echo "Start time: $(date)"
echo "=============================================="

srun "${PYTHON}" -u scripts/analyze_condition_distribution.py \
    --config configs/zinc250k.yaml \
    --splits data/processed/zinc250k/zinc_splits.pkl \
    --output-dir outputs/metrics/condition_distribution \
    --qed-bins 20

echo "=============================================="
echo "Condition-distribution analysis complete"
echo "Results: outputs/metrics/condition_distribution"
echo "End time: $(date)"
echo "=============================================="
