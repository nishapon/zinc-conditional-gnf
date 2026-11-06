#!/bin/bash
#SBATCH --job-name=zinc_flow_qedcontrast
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
#SBATCH --time=12:00:00

# ---------------------------------------------------------------------------
# FULL RUN: 128D joint (N, QED) flow + N-matched QED-CONTRASTIVE objective, seed 3.
#
# The ONE controlled change against configs/zinc250k_flow_nqed_ae128.yaml: the
# forward-only contrastive term. Same FROZEN 128D autoencoder, same split, 12 blocks,
# same optimiser/schedule/noise/whitening, same 50,000-step budget, same seed 3.
#
# Replaces the sampling-based objective that diverged (jobs 359287/359294/360343). The new
# term backpropagates through the FORWARD map only, so it never calls torch.linalg.inv and
# never enters the sequential node-autoregressive inverse - the two operations present
# ONLY in flow.inverse and therefore never constrained by the NLL.
#
# RESUMABLE: checkpoint every 1,000 steps; resubmitting continues from last.pt at the same
# step, batch position and RNG state. A walltime kill is not a lost run.
#
# SAFETY: writes ONLY to outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_<n>. The
# 128D baseline and every failed-experiment directory are untouched. If the auxiliary term
# is non-finite in more than 2% of applications the run STOPS with a clear error rather
# than training with the signal silently absent (the previous attempt's real failure).
#
#   SEEDS="3"          flow seeds (seed 3 is the baseline comparison)
#   MAX_STEPS=50000
#   PAUSE_AT=          optional clean stop (stays unfinished; resubmit to continue)
#
#     sbatch slurm/13c_flow_nqed_qedcontrast_gpu.sh
#
# Run slurm/13c_qedcontrast_activation_gpu.sh FIRST and confirm its pass criteria.
# Never run this file with `bash` on the login node.
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
    echo "flow rc: 2=data/checkpoint missing, 3=resume or autoencoder mismatch," >&2
    echo "         4=bad args/config, 5=non-finite loss OR auxiliary skip-rate exceeded," >&2
    echo "         130=interrupted (resubmit), 1=traceback" >&2
    echo "rc=5 with 'QED-contrastive objective was non-finite' means the objective is" >&2
    echo "unstable: investigate, do NOT just lower the weight." >&2
    echo "inspect: tail -60 ${ROOT:-outputs/checkpoints/flow_nqed_ae128_qedcontrast}/seed_*/train.log" >&2
    echo "A walltime kill is NOT a lost run: resubmit the same command to resume." >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/13c_flow_nqed_qedcontrast_gpu.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline

CONFIG="${CONFIG:-configs/zinc250k_flow_nqed_ae128_qedcontrast.yaml}"
DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
AE_CHECKPOINT="${AE:-outputs/ablation/ae_degree_128d/seed_3/last.pt}"
ROOT="${ROOT:-outputs/checkpoints/flow_nqed_ae128_qedcontrast}"
SEEDS="${SEEDS:-3}"
MAX_STEPS="${MAX_STEPS:-50000}"
PAUSE_AT="${PAUSE_AT:-}"
PAUSE_ARGS=()
if [[ -n "${PAUSE_AT}" ]]; then PAUSE_ARGS=(--pause-at "${PAUSE_AT}"); fi
mkdir -p outputs/logs "${ROOT}"

echo "=============================================="
echo "128D (N, QED) flow + N-matched QED-contrastive objective (forward-only)"
echo "Start:            $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID}   Node: $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Config:           ${CONFIG}"
echo "Autoencoder:      ${AE_CHECKPOINT} (FROZEN, read-only, never retrained)"
echo "Seeds:            ${SEEDS}    steps per seed: ${MAX_STEPS}"
echo "Output root:      ${ROOT}  (128D baseline + failed experiments NOT touched)"
echo "=============================================="
"${PYTHON}" -V
nvidia-smi || { echo "ERROR: nvidia-smi failed: no GPU visible to this job" >&2; exit 67; }

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${AE_CHECKPOINT}" "${CONFIG}"; do
    [[ -f "${required}" ]] || { echo "ERROR: ${required} not found." >&2; exit 66; }
done
"${PYTHON}" -c "import hashlib,sys;print('autoencoder sha256:',hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "${AE_CHECKPOINT}"

"${PYTHON}" - <<'PYCHECK'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False; the Conda env needs the GPU build.")
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
import rdkit, numpy, scipy, yaml
print("dependency check ok; rdkit", rdkit.__version__, "| scipy", scipy.__version__)
PYCHECK

for SEED in ${SEEDS}; do
    OUT="${ROOT}/seed_${SEED}"
    if [[ -f "${OUT}/summary.json" ]]; then
        STATE=$("${PYTHON}" -c "import json,sys;d=json.load(open(sys.argv[1]));print('done' if d.get('completed') or d.get('early_stopped') else ('paused' if d.get('paused') else 'unfinished'))" "${OUT}/summary.json")
        if [[ "${STATE}" == "done" ]]; then
            echo "seed ${SEED}: already finished (${OUT}/summary.json), skipping"
            continue
        fi
        echo "seed ${SEED}: ${STATE} run found, resuming from ${OUT}/last.pt"
    fi
    echo
    echo "---- 128D (N,QED)+contrastive flow, seed ${SEED} ($(date -u +%H:%M:%SZ)) ----"
    srun "${PYTHON}" -u scripts/04_train_flow.py \
        --config "${CONFIG}" \
        --data-dir "${DATA_DIR}" \
        --ae-checkpoint "${AE_CHECKPOINT}" \
        --output-dir "${OUT}" \
        --device cuda \
        --seed "${SEED}" \
        --max-steps "${MAX_STEPS}" \
        "${PAUSE_ARGS[@]+${PAUSE_ARGS[@]}}" \
        --resume auto
done

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for SEED in ${SEEDS}; do
    echo "---- seed ${SEED} ----"
    grep "VAL step" "${ROOT}/seed_${SEED}/train.log" 2>/dev/null | tail -3 || echo "  (no train.log)"
    grep "QED-contrastive" "${ROOT}/seed_${SEED}/train.log" 2>/dev/null | tail -3 || true
done
echo
echo "If summary.json says completed=false, resubmit this same command to continue."
echo
echo "EVALUATE (unchanged protocol: N=15,23,30 x QED=0.4,0.6,0.8 x T=0.2 x 100/cell):"
echo "  sbatch --export=ALL,FLOW_CHECKPOINT=${ROOT}/seed_3/best.pt,OUT=outputs/samples/flow_nqed_ae128_qedcontrast/seed_3 slurm/13_generate_nqed.sh"
echo "No QED claim is valid until that job reports RDKit-measured QED on generated molecules."
echo "=============================================="
