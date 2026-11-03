#!/bin/bash
#SBATCH --job-name=zinc_ncond_smoke
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# ---------------------------------------------------------------------------
# STAGE 12 SMOKE TEST: N-conditioned flow, short GPU run, then the conditioning
# verification and a small N-conditioned sample. Nothing here is a result.
#
# What it answers, in order:
#   1. does the conditional model train at all (finite loss, sane log|det J|)?
#   2. does the trainer actually feed each molecule its own N? The validation line
#      prints cond_gap and cond_win: the same validation molecules scored against
#      PERMUTED conditions. A gap of 0.0000 with cond_win 0.000 means the condition
#      is not entering the density - a wiring bug, visible in minutes.
#   3. can the sampler turn a REQUESTED N into molecules of that size, and does the
#      whole generation/evaluation path run end to end?
#
# A smoke run of ~600 steps cannot have LEARNED conditioning, so the generation step
# runs with --allow-weak-conditioning: a failed verification is reported, not fatal.
# The full run (slurm/12_flow_ncond_gpu.sh) is the one that must pass it.
#
# Writes only under outputs/checkpoints/flow_ncond_smoke and
# outputs/samples/flow_ncond_smoke. The frozen unconditional results
# (outputs/checkpoints/flow_uncond*, outputs/samples/flow_uncond*) are never touched.
#
#   STEPS=600          training steps
#   COUNT=200          samples per requested N
#   TARGET_N=...       comma-separated requested atom counts (default: 15,23,30)
#
#     sbatch slurm/12_flow_ncond_smoke_gpu.sh
#
# Never run this file with `bash` directly on the login node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; never train on the login node." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "rc: 2=data/checkpoint missing, 3=autoencoder mismatch, 4=bad args/config," >&2
    echo "    5=conditioning verification failed, 130=interrupted, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT:-outputs/checkpoints/flow_ncond_smoke}/train.log" >&2
    echo "==============================================" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/12_flow_ncond_smoke_gpu.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline
export MPLBACKEND=Agg

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
AE_CHECKPOINT="${ZINC_GNF_AE:-outputs/ablation/ae_degree/seed_3/last.pt}"
OUT="outputs/checkpoints/flow_ncond_smoke"
SAMPLES="outputs/samples/flow_ncond_smoke"
STEPS="${STEPS:-600}"
COUNT="${COUNT:-200}"
TARGET_N="${TARGET_N:-15,23,30}"
mkdir -p outputs/logs "${OUT}" "${SAMPLES}"

echo "=============================================="
echo "Stage 12 smoke test: N-conditioned flow"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Config:           configs/zinc250k_flow_ncond.yaml (condition_features: [n], Arm B)"
echo "Autoencoder:      ${AE_CHECKPOINT} (read-only, frozen)"
echo "Steps:            ${STEPS}   samples per N: ${COUNT}   requested N: ${TARGET_N}"
echo "Checkpoints:      ${OUT}"
echo "Samples:          ${SAMPLES}"
echo "=============================================="
"${PYTHON}" -V
nvidia-smi || { echo "ERROR: nvidia-smi failed: no GPU visible to this job" >&2; exit 67; }

if [[ ! -f "${DATA_DIR}/splits.pkl" ]]; then
    echo "ERROR: ${DATA_DIR}/splits.pkl not found." >&2
    exit 66
fi
if [[ ! -f "${DATA_DIR}/scaler.json" ]]; then
    echo "ERROR: ${DATA_DIR}/scaler.json not found; conditional sampling needs the" >&2
    echo "train-only condition scaler the preprocessing writes." >&2
    exit 66
fi
if [[ ! -f "${AE_CHECKPOINT}" ]]; then
    echo "ERROR: frozen autoencoder ${AE_CHECKPOINT} not found." >&2
    exit 66
fi

"${PYTHON}" - <<'PYCHECK'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False; the Conda env needs the GPU build.")
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
import rdkit, numpy, scipy, yaml
from scipy.optimize import milp
print("dependency check ok; rdkit", rdkit.__version__, "| scipy", scipy.__version__)
PYCHECK

echo
echo "---- 1/2 short conditional training (${STEPS} steps) ----"
srun "${PYTHON}" -u scripts/04_train_flow.py \
    --config configs/zinc250k_flow_ncond.yaml \
    --data-dir "${DATA_DIR}" \
    --ae-checkpoint "${AE_CHECKPOINT}" \
    --output-dir "${OUT}" \
    --device cuda \
    --seed 1 \
    --max-steps "${STEPS}" \
    --eval-every 200 \
    --save-every 200 \
    --log-every 50 \
    --val-molecules 2000 \
    --statistics-molecules 20000 \
    --resume auto

echo
echo "---- 2/2 conditioning verification + N-conditioned sample ----"
srun "${PYTHON}" -u scripts/12_generate_ncond.py \
    --flow-checkpoint "${OUT}/last.pt" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${SAMPLES}" \
    --target-n "${TARGET_N}" \
    --count "${COUNT}" \
    --temperature 0.2 \
    --batch-size 128 \
    --check-molecules 2000 \
    --device cuda \
    --allow-weak-conditioning

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "READ THESE THREE THINGS:"
echo "  1. grep VAL ${OUT}/train.log        -> cond_gap and cond_win per evaluation."
echo "     cond_gap > 0 and cond_win > 0.5 mean the condition is entering the density."
echo "     cond_gap exactly 0.0000 is a WIRING BUG, not a training-length problem."
echo "  2. ${SAMPLES}/conditioning_check.txt -> the same check in full, plus mu(N)/sigma(N)."
echo "  3. ${SAMPLES}/ncond_report.txt       -> per requested N: delivered N, raw validity,"
echo "     degree validity, fallback, QED against real molecules of the SAME size."
echo
echo "A short run is NOT expected to pass the verification thresholds; it is expected to"
echo "show a non-zero gap and a working end-to-end path. If both look right, start the"
echo "full run:  sbatch slurm/12_flow_ncond_gpu.sh"
echo "=============================================="
