#!/bin/bash
#SBATCH --job-name=zinc_nqed_smoke
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
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# STAGE 13 SMOKE TEST: joint (N, QED) conditioning. Short GPU run, then the three
# permutation controls, the DECODE CEILING and a small (N, QED) grid at two
# temperatures. Nothing here is a result.
#
# What it answers:
#   1. does the joint model train (finite loss, cond_gap non-zero in the VAL lines)?
#   2. is QED used for anything N does not already explain? The qed_only control
#      permutes QED alone and keeps every molecule's true N. A gap of 0 there means
#      QED is not reaching the density - a wiring problem, visible in minutes, and
#      not something more steps would fix.
#   3. IS THE RAW-VALIDITY / AROMATICITY PROBLEM A DECODING DEFECT OR A MODELLING ONE?
#      The decode ceiling runs the same raw / greedy / degree decoders on REAL
#      validation embeddings from the frozen autoencoder. If the raw argmax decodes
#      real embeddings to valid, aromatic molecules, the bond prediction is fine and
#      the sampled embeddings are what is wrong (modelling, needs training/temperature).
#      If it does not, the raw argmax itself is the ceiling and raw validity can never
#      be high, however long the flow trains. This single row decides which.
#   4. does the whole (N, QED) -> molecules -> QED-achievement path run end to end?
#
# A ~600-step model cannot have learned QED control, so the evaluation runs with
# --allow-weak-conditioning: a failed control is reported, not fatal. The full run
# (slurm/13_flow_nqed_gpu.sh) is the one that must pass it.
#
# Writes only under outputs/checkpoints/flow_nqed_smoke and outputs/samples/flow_nqed_smoke.
# The frozen unconditional results and the N-only stage are never touched.
#
#   STEPS=600                     training steps
#   COUNT=150                     samples per (T, N, QED) cell
#   TARGET_N=15,23                requested atom counts
#   TARGET_QED=0.4,0.8            requested QED values (far apart on purpose)
#   TEMPERATURES=0.2,0.85         the trade-off the frozen result never measured jointly
#
#     sbatch slurm/13_flow_nqed_smoke_gpu.sh
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
    echo "    5=control failed, 130=interrupted, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT:-outputs/checkpoints/flow_nqed_smoke}/train.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/13_flow_nqed_smoke_gpu.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline
export MPLBACKEND=Agg

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
AE_CHECKPOINT="${ZINC_GNF_AE:-outputs/ablation/ae_degree/seed_3/last.pt}"
OUT="outputs/checkpoints/flow_nqed_smoke"
SAMPLES="outputs/samples/flow_nqed_smoke"
STEPS="${STEPS:-600}"
COUNT="${COUNT:-150}"
TARGET_N="${TARGET_N:-15,23}"
TARGET_QED="${TARGET_QED:-0.4,0.8}"
TEMPERATURES="${TEMPERATURES:-0.2,0.85}"
mkdir -p outputs/logs "${OUT}" "${SAMPLES}"

echo "=============================================="
echo "Stage 13 smoke test: joint (N, QED)-conditioned flow"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Config:           configs/zinc250k_flow_nqed.yaml (condition_features: [n, qed], Arm B)"
echo "Autoencoder:      ${AE_CHECKPOINT} (read-only, frozen)"
echo "Steps:            ${STEPS}"
echo "Grid:             N=${TARGET_N}  QED=${TARGET_QED}  T=${TEMPERATURES}  x ${COUNT} samples"
echo "Checkpoints:      ${OUT}"
echo "Samples:          ${SAMPLES}"
echo "=============================================="
"${PYTHON}" -V
nvidia-smi || { echo "ERROR: nvidia-smi failed: no GPU visible to this job" >&2; exit 67; }

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${AE_CHECKPOINT}"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: ${required} not found." >&2
        exit 66
    fi
done

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
echo "---- 1/2 short joint training (${STEPS} steps) ----"
srun "${PYTHON}" -u scripts/04_train_flow.py \
    --config configs/zinc250k_flow_nqed.yaml \
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
echo "---- 2/2 controls + decode ceiling + (N, QED) grid ----"
srun "${PYTHON}" -u scripts/13_generate_nqed.py \
    --flow-checkpoint "${OUT}/last.pt" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${SAMPLES}" \
    --target-n "${TARGET_N}" \
    --target-qed "${TARGET_QED}" \
    --temperature "${TEMPERATURES}" \
    --count "${COUNT}" \
    --batch-size 128 \
    --check-molecules 2000 \
    --ceiling-molecules 1000 \
    --device cuda \
    --allow-weak-conditioning

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "READ THESE FOUR THINGS:"
echo "  1. grep VAL ${OUT}/train.log"
echo "     cond_gap non-zero = the joint conditioning reaches the density."
echo "  2. ${SAMPLES}/conditioning_check.json -> controls.qed_only.gap_nats_per_node"
echo "     This is the only number that shows QED doing something N cannot. A gap of 0"
echo "     after 600 steps is weak evidence; a gap of EXACTLY 0 is a wiring bug."
echo "  3. ${SAMPLES}/conditioning_check.json -> decode_ceiling.decoders.raw"
echo "     THE ANSWER TO THE RAW-VALIDITY QUESTION. High validity / aromaticity there"
echo "     means the decoder is fine and the flow's embeddings are the problem; low"
echo "     validity means raw argmax is the ceiling and raw validity is not a model"
echo "     metric at all. Either way, stop treating 0% raw validity as unexplained."
echo "  4. ${SAMPLES}/nqed_report.txt -> per-cell table incl. the T=0.2 vs T=0.85"
echo "     trade-off: higher temperature usually buys aromaticity and costs validity."
echo
echo "If 1-4 look right, start the full run: sbatch slurm/13_flow_nqed_gpu.sh"
echo "=============================================="
