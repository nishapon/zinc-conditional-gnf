#!/bin/bash
#SBATCH --job-name=ZINC_condition_check
#SBATCH --output=logs/zinc_condition_check_output.log
#SBATCH --error=logs/zinc_condition_check_error.log
#SBATCH --mail-type=NONE
#SBATCH --partition=STUD

set -euo pipefail

PYTHON=/home/${USER}/miniconda3/envs/zinc_gnf/bin/python
WORKDIR=/home/${USER}/zinc-conditional-gnf

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
