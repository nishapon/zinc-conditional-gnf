#!/bin/bash
#SBATCH --job-name=zinc_flow_nqed
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
# STAGE 13 FULL RUN: joint (N, QED)-conditioned flow (Arm B architecture), resumable.
#
# Same frozen autoencoder, dataset split, smoothing sigma, whitening, optimiser and
# schedule as the frozen unconditional Arm B run; the ONLY differences are
# condition_features = ["n", "qed"] and the output directory. Nothing here can touch the
# unconditional or the N-only outputs.
#
# QED is a per-molecule label the preprocessing computed and standardised with the
# TRAIN-ONLY scaler; this is plain conditional maximum likelihood, not optimisation
# against a QED oracle.
#
# RESUMABLE: a checkpoint every 1,000 steps; resubmitting this exact command continues
# from last.pt at the same step, batch position and RNG state. A walltime kill is not a
# lost run. A seed whose summary.json says completed or early-stopped is skipped.
#
# Every evaluation prints cond_gap / cond_win (validation molecules scored against
# PERMUTED conditions - here BOTH features at once). That is the running evidence that
# the conditioning is alive, but it cannot separate QED from N, because the two are
# correlated in ZINC. The QED-only control lives in the evaluation job
# (slurm/13_generate_nqed.sh), which refuses to print a QED table if it fails.
#
#   SEEDS="1"           which flow seeds to train, in order
#   MAX_STEPS=50000     steps per seed (matches the frozen unconditional budget)
#   PAUSE_AT=           optional clean stop at that step (stays unfinished; resubmit)
#   ZINC_GNF_AE=...     a different frozen autoencoder (not for the main results)
#
#     sbatch slurm/13_flow_nqed_gpu.sh
#     sbatch --export=ALL,PAUSE_AT=10000 slurm/13_flow_nqed_gpu.sh
#     sbatch --export=ALL,SEEDS="1 2 3" slurm/13_flow_nqed_gpu.sh
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
    echo "flow rc: 2=data/checkpoint missing, 3=resume or autoencoder mismatch," >&2
    echo "         4=bad args/config, 5=non-finite loss, 130=interrupted (resubmit), 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList" >&2
    echo "         tail -50 ${ROOT:-outputs/checkpoints/flow_nqed}/seed_*/train.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/13_flow_nqed_gpu.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
AE_CHECKPOINT="${ZINC_GNF_AE:-outputs/ablation/ae_degree/seed_3/last.pt}"
ROOT="outputs/checkpoints/flow_nqed"
SEEDS="${SEEDS:-1}"
MAX_STEPS="${MAX_STEPS:-50000}"
PAUSE_AT="${PAUSE_AT:-}"
PAUSE_ARGS=()
if [[ -n "${PAUSE_AT}" ]]; then PAUSE_ARGS=(--pause-at "${PAUSE_AT}"); fi
mkdir -p outputs/logs "${ROOT}"

echo "=============================================="
echo "Stage 13: joint (N, QED)-conditioned flow training (Arm B architecture)"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Config:           configs/zinc250k_flow_nqed.yaml (condition_features: [n, qed])"
echo "Autoencoder:      ${AE_CHECKPOINT} (read-only, frozen)"
echo "Seeds:            ${SEEDS}    steps per seed: ${MAX_STEPS}"
echo "Pause at step:    ${PAUSE_AT:-none (train straight through)}"
echo "Output root:      ${ROOT}  (unconditional and N-only runs are NOT touched)"
echo "=============================================="
"${PYTHON}" -V
nvidia-smi || { echo "ERROR: nvidia-smi failed: no GPU visible to this job" >&2; exit 67; }

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${AE_CHECKPOINT}"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: ${required} not found." >&2
        exit 66
    fi
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
    echo "---- N-conditioned flow, seed ${SEED} ($(date -u +%H:%M:%SZ)) ----"
    srun "${PYTHON}" -u scripts/04_train_flow.py \
        --config configs/zinc250k_flow_nqed.yaml \
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
    echo "seed ${SEED}:"
    grep VAL "${ROOT}/seed_${SEED}/train.log" 2>/dev/null | tail -3 || echo "  (no train.log)"
done
echo
echo "Per-seed: ${ROOT}/seed_<n>/{train.log,metrics.jsonl,summary.json,best.pt,last.pt}"
echo "If summary.json says completed=false, resubmit this same command to continue."
echo
echo "cond_gap in the VAL lines shows the conditioning is alive (0.0000 = ignored);"
echo "the QED-only control that separates QED from N runs in the evaluation job."
echo "When a seed has finished, evaluate it:"
echo "  sbatch slurm/13_generate_nqed.sh"
echo "=============================================="
