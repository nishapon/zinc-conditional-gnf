#!/bin/bash
#SBATCH --job-name=zinc_ae_analysis
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00

# ---------------------------------------------------------------------------
# Read-only failure analysis of an existing autoencoder checkpoint. CPU only.
# Default target: the GPU smoke-test checkpoint (job 324485). It loads the
# checkpoint and the VALIDATION split, writes a report, and changes nothing.
#
#     sbatch slurm/03_analyze_reconstruction.sh
#     sbatch --export=ALL,CHECKPOINT=outputs/ablation/ae_baseline/seed_1/best.pt,\
#            OUT=outputs/analysis/ae_baseline_seed1 slurm/03_analyze_reconstruction.sh
#
# Never run this file with `bash` directly on the master node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "JOB FAILED rc=${rc} line=${1:-?} command: ${BASH_COMMAND}" >&2
    echo "analysis rc: 2=file missing, 3=checkpoint/vocabulary mismatch, 4=bad args, 1=see traceback" >&2
}
trap 'on_error $LINENO' ERR

WORKDIR="${SLURM_SUBMIT_DIR:-$HOME/zinc-conditional-gnf}"
cd "${WORKDIR}"

PYTHON="${ZINC_GNF_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    for candidate in "$HOME/miniconda3/envs/zinc_gnf/bin/python" \
                     "$HOME/anaconda3/envs/zinc_gnf/bin/python" \
                     "$HOME/miniforge3/envs/zinc_gnf/bin/python"; do
        if [[ -x "${candidate}" ]]; then PYTHON="${candidate}"; break; fi
    done
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "ERROR: could not find the python of the Conda env 'zinc_gnf'." >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/ae_smoke_gpu/best.pt}"
OUT="${OUT:-outputs/analysis/ae_smoke_gpu}"
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Reconstruction failure analysis (read-only)"
echo "Start time:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:         ${SLURM_JOB_ID}   Node: $(hostname)   Partition: ${SLURM_JOB_PARTITION:-?}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Output:      ${OUT}"
echo "Git commit:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "=============================================="

[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: ${CHECKPOINT} not found" >&2; exit 66; }
[[ -f "${DATA_DIR}/splits.pkl" ]] || { echo "ERROR: ${DATA_DIR}/splits.pkl not found" >&2; exit 66; }

srun "${PYTHON}" -u scripts/03_analyze_reconstruction.py \
    --checkpoint "${CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --split val \
    --max-molecules 3000 \
    --rescue-sample 300 \
    --device cpu \
    --output-dir "${OUT}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo "Summary: ${OUT}/reconstruction_analysis.txt   Report: ${OUT}/reconstruction_analysis.json"
echo "=============================================="
