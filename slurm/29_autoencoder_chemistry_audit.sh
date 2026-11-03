#!/bin/bash
#SBATCH --job-name=zinc_ae_chem_audit
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Read-only Stage 17 autoencoder audit: no training or GNF generation.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run using sbatch, never directly on the login node" >&2
  exit 64
fi
cd "${SLURM_SUBMIT_DIR:-$HOME/zinc-conditional-gnf}"
PYTHON="${ZINC_GNF_PYTHON:-$HOME/miniconda3/envs/zinc_gnf/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing environment Python: $PYTHON" >&2; exit 65; }
FLOW="${FLOW:-outputs/checkpoints/warm_start_stageA/last.pt}"
OUT="${OUT:-outputs/metrics/ae_chemistry_audit}"
COUNT="${COUNT:-800}"
mkdir -p outputs/logs "$OUT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ARGS=(--flow-checkpoint "$FLOW" --output-dir "$OUT" --max-molecules "$COUNT" --batch-size 32 --device cpu --save-per-molecule)
if [[ "${SKIP_DEGREE:-0}" == 1 ]]; then ARGS+=(--skip-degree); fi
if [[ -n "${AE_CHECKPOINT:-}" ]]; then ARGS+=(--ae-checkpoint "$AE_CHECKPOINT"); fi
srun "$PYTHON" -u scripts/29_autoencoder_chemistry_audit.py "${ARGS[@]}"
echo "DONE: $OUT/ae_chemistry_audit.txt"
