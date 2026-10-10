#!/bin/bash
#SBATCH --job-name=zinc_ae_smoke
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00

# ---------------------------------------------------------------------------
# GPU SMOKE TEST of the charge-aware graph autoencoder. Short on purpose.
#
# Purpose: verify on a real GPU that the autoencoder trains, that validation
# model selection / checkpointing / resume work, and that it can reconstruct
# charge-aware ZINC graphs. It is NOT a research run: 3,000 steps on 50,000
# training molecules, validation split only (the test split is never loaded).
#
# GPU partition per the ISML cluster tutorial: --partition=STUD --gres=gpu:1.
# (Nothing here needs more than one GPU.) The job FAILS FAST if torch cannot
# see a GPU, instead of silently training on CPU.
#
# Prerequisite (CPU jobs, see slurm/README_ae.md):
#   data/processed/zinc250k_charge_aware_moflow_fixed/splits.pkl
#
#     sbatch slurm/02_ae_smoke_gpu.sh
#
# Resubmitting the same command resumes from outputs/checkpoints/ae_smoke_gpu/last.pt.
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
    echo "train.py rc: 2=data missing, 3=resume mismatch, 4=bad args/config/data format," >&2
    echo "             130=interrupted (checkpoint saved; resubmit to resume), 1=see traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList" >&2
    echo "         tail -50 ${OUT:-outputs/checkpoints/ae_smoke_gpu}/train.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/02_ae_smoke_gpu.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
OUT="outputs/checkpoints/ae_smoke_gpu"
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Autoencoder GPU smoke test"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'"
echo "Workdir:          ${WORKDIR}"
echo "Python:           ${PYTHON}"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Data:             ${DATA_DIR}"
echo "Output:           ${OUT}"
echo "=============================================="
"${PYTHON}" -V
nvidia-smi || { echo "ERROR: nvidia-smi failed: no GPU visible to this job" >&2; exit 67; }

if [[ ! -f "${DATA_DIR}/splits.pkl" ]]; then
    echo "ERROR: ${DATA_DIR}/splits.pkl not found." >&2
    echo "Run the CPU derivation job first: sbatch slurm/01d_derive_moflow_split.sh" >&2
    exit 66
fi
ls -la "${DATA_DIR}"

# GPU preflight: the GPU build of PyTorch must see the device.
"${PYTHON}" - <<'PYCHECK'
import sys
import torch
print("torch", torch.__version__, "| cuda build:", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False. The Conda env needs the GPU build of PyTorch "
             "(pip3 install torch --index-url https://download.pytorch.org/whl/cu121, on the login node).")
print("GPU:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0),
      "| memory GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
import rdkit, numpy, scipy, yaml
print("dependency check ok; rdkit", rdkit.__version__)
PYCHECK

echo
echo "---- training (smoke) ----"
srun "${PYTHON}" -u scripts/02_train_autoencoder.py \
    --config configs/zinc250k_charge_aware_ae.yaml \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --device cuda \
    --seed 42 \
    --batch-size 32 \
    --max-steps 3000 \
    --max-train-molecules 50000 \
    --eval-every 500 \
    --save-every 500 \
    --log-every 100 \
    --val-loss-molecules 2000 \
    --val-recon-molecules 500 \
    --resume auto

echo
echo "---- reconstruction evaluation of the selected checkpoint (validation split) ----"
srun "${PYTHON}" -u scripts/02b_evaluate_autoencoder.py \
    --checkpoint "${OUT}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --split val \
    --max-molecules 2000 \
    --device cuda \
    --output "${OUT}/val_reconstruction.json"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo "Training log:   ${OUT}/train.log"
echo "Metrics:        ${OUT}/metrics.jsonl"
echo "Summary:        ${OUT}/summary.json"
echo "Reconstruction: ${OUT}/val_reconstruction.json"
echo "Checkpoints:    ${OUT}/best.pt  ${OUT}/last.pt"
echo "=============================================="
