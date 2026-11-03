#!/bin/bash
#SBATCH --job-name=gnf_gaussian_audit
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit with sbatch; do not run on login node.' >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:-$HOME/zinc-conditional-gnf}"
PYTHON="${ZINC_GNF_PYTHON:-$HOME/miniconda3/envs/zinc_gnf/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing environment interpreter: $PYTHON" >&2; exit 2; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/logs outputs/metrics/conditional_gaussian_audit
srun "$PYTHON" -u scripts/28_conditional_gaussian_audit.py \
  --flow-checkpoint "${FLOW_CHECKPOINT:-outputs/checkpoints/warm_start_stageA/last.pt}" \
  --data-dir "${DATA_DIR:-data/processed/zinc250k_charge_aware_moflow_fixed}" \
  --output-dir "${OUT:-outputs/metrics/conditional_gaussian_audit}" \
  --global-molecules "${GLOBAL_MOLECULES:-256}" \
  --per-cell-molecules "${CELL_MOLECULES:-32}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --device cpu
