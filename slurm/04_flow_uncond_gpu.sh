#!/bin/bash
#SBATCH --job-name=zinc_flow_uncond
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
#SBATCH --time=06:00:00

# ---------------------------------------------------------------------------
# UNCONDITIONAL node-feature graph normalizing flow, full training run.
#
# Trains on the whole TRAIN split (202,111 molecules) of the MoFlow-fixed
# dataset, through the FROZEN learned-degree autoencoder
# (outputs/ablation/ae_degree/seed_3/last.pt, selected on validation only).
# The validation split is used for early stopping; the test split is never
# loaded. The autoencoder file is opened read-only and its SHA-256 is written
# into every flow checkpoint.
#
# RESUMABLE BY DESIGN. The runtime estimate (about 1-2 h for 50,000 steps on
# an RTX 2070 SUPER) is an estimate, not a promise: a checkpoint is written
# every 1,000 steps, and resubmitting this exact command continues from
# last.pt at the same step, batch position and RNG state. If the walltime
# below is not enough, just submit again (and again) until summary.json says
# completed; a seed whose summary.json already says completed or early-stopped
# is skipped.
#
#   SEEDS="1"           which flow seeds to train, in order (e.g. SEEDS="1 2 3 4 5")
#   MAX_STEPS=50000     training steps per seed
#   ZINC_GNF_AE=...     a different frozen autoencoder (not for the main results)
#
#     sbatch slurm/04_flow_uncond_gpu.sh
#     sbatch --export=ALL,SEEDS="1 2 3 4 5" slurm/04_flow_uncond_gpu.sh
#
# Never run this file with `bash` directly on the master node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; never run" >&2
    echo "training on the master node." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "flow rc: 2=data/checkpoint missing, 3=resume or autoencoder mismatch," >&2
    echo "         4=bad args/config, 5=non-finite loss, 130=interrupted (resubmit), 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList" >&2
    echo "         tail -50 ${ROOT:-outputs/checkpoints/flow_uncond}/seed_*/train.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/04_flow_uncond_gpu.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
AE_CHECKPOINT="${ZINC_GNF_AE:-outputs/ablation/ae_degree/seed_3/last.pt}"
ROOT="outputs/checkpoints/flow_uncond"
SEEDS="${SEEDS:-1}"
MAX_STEPS="${MAX_STEPS:-50000}"
mkdir -p outputs/logs "${ROOT}"

echo "=============================================="
echo "Unconditional flow training"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Walltime limit:   ${SBATCH_TIMELIMIT:-see --time in this file} (resubmit to resume)"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Autoencoder:      ${AE_CHECKPOINT} (read-only, frozen)"
echo "Seeds:            ${SEEDS}    steps per seed: ${MAX_STEPS}"
echo "Output root:      ${ROOT}"
echo "=============================================="
"${PYTHON}" -V
nvidia-smi || { echo "ERROR: nvidia-smi failed: no GPU visible to this job" >&2; exit 67; }

if [[ ! -f "${DATA_DIR}/splits.pkl" ]]; then
    echo "ERROR: ${DATA_DIR}/splits.pkl not found (sbatch slurm/01d_derive_moflow_split.sh first)." >&2
    exit 66
fi
if [[ ! -f "${AE_CHECKPOINT}" ]]; then
    echo "ERROR: frozen autoencoder ${AE_CHECKPOINT} not found." >&2
    exit 66
fi
"${PYTHON}" -c "import hashlib,sys;print('autoencoder sha256:',hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "${AE_CHECKPOINT}"

"${PYTHON}" - <<'PYCHECK'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False; the Conda env needs the GPU build of PyTorch.")
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
import rdkit, numpy, scipy, yaml
print("dependency check ok; rdkit", rdkit.__version__, "| scipy", scipy.__version__)
PYCHECK

for SEED in ${SEEDS}; do
    OUT="${ROOT}/seed_${SEED}"
    if [[ -f "${OUT}/summary.json" ]]; then
        STATE=$("${PYTHON}" -c "import json,sys;d=json.load(open(sys.argv[1]));print('done' if d.get('completed') or d.get('early_stopped') else 'unfinished')" "${OUT}/summary.json")
        if [[ "${STATE}" == "done" ]]; then
            echo "seed ${SEED}: already finished (${OUT}/summary.json), skipping"
            continue
        fi
        echo "seed ${SEED}: unfinished run found, resuming from ${OUT}/last.pt"
    fi
    echo
    echo "---- unconditional flow, seed ${SEED} ($(date -u +%H:%M:%SZ)) ----"
    srun "${PYTHON}" -u scripts/04_train_flow.py \
        --config configs/zinc250k_flow_uncond.yaml \
        --data-dir "${DATA_DIR}" \
        --ae-checkpoint "${AE_CHECKPOINT}" \
        --output-dir "${OUT}" \
        --device cuda \
        --seed "${SEED}" \
        --max-steps "${MAX_STEPS}" \
        --resume auto
done

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for SEED in ${SEEDS}; do
    echo "seed ${SEED}:"
    tail -3 "${ROOT}/seed_${SEED}/train.log" 2>/dev/null || echo "  (no train.log)"
done
echo
echo "Per-seed: ${ROOT}/seed_<n>/{train.log,metrics.jsonl,summary.json,best.pt,last.pt}"
echo "If summary.json says completed=false, resubmit this same command to continue."
echo "When a seed has finished: sbatch slurm/05_generate_uncond.sh"
echo "=============================================="
