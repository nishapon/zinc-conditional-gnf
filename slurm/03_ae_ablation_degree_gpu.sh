#!/bin/bash
#SBATCH --job-name=zinc_ae_degree
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
#SBATCH --time=01:30:00

# ---------------------------------------------------------------------------
# SAME-BUDGET autoencoder ablation, variant: degree   (equivariant per-atom degree head + degree-constrained assembly; deterministic)
#
# IMPROVED ARM (primary candidate). Same encoder and embeddings as the baseline, plus a small degree head;
# the degree-constrained decoder is reported next to the greedy one for the SAME model.
#
# Every arm of the ablation (outputs/ablation/ae_<variant>/seed_<n>/) uses the SAME
# data, subset size, steps, optimiser, seeds and validation molecules, and
# reports the LAST-step model (no checkpoint-selection advantage). Seeds 1,2,3
# are run one after another; a seed whose result.json exists is skipped, so
# resubmitting after an interruption continues instead of overwriting.
#
#   GPU : 1 (STUD partition)      CPUs: 4      RAM: 16 GB
#   Time: about 5-8 min per seed on an RTX 2070 SUPER; 1h30 limit covers all 3 seeds
#   Data: data/processed/zinc250k_charge_aware_moflow_fixed (validation only; test never loaded)
#
#     sbatch slurm/03_ae_ablation_degree_gpu.sh
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
    echo "experiment rc: 2=data missing, 4=bad args or result already exists, 1=see traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList" >&2
    echo "         tail -50 ${ROOT:-outputs/ablation/ae_degree}/seed_*/train.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/03_ae_ablation_degree_gpu.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
ROOT="outputs/ablation/ae_degree"
mkdir -p outputs/logs "${ROOT}"

echo "=============================================="
echo "Autoencoder ablation: degree"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'"
echo "Workdir:          ${WORKDIR}"
echo "Python:           ${PYTHON}"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Output root:      ${ROOT}"
echo "=============================================="
"${PYTHON}" -V
nvidia-smi || { echo "ERROR: nvidia-smi failed: no GPU visible to this job" >&2; exit 67; }

if [[ ! -f "${DATA_DIR}/splits.pkl" ]]; then
    echo "ERROR: ${DATA_DIR}/splits.pkl not found (run slurm/01d_derive_moflow_split.sh first)." >&2
    exit 66
fi

"${PYTHON}" - <<'PYCHECK'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False; the Conda env needs the GPU build of PyTorch.")
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
import rdkit, numpy, scipy, yaml
print("dependency check ok; rdkit", rdkit.__version__, "| scipy", scipy.__version__)
PYCHECK

for SEED in 1 2 3; do
    OUT="${ROOT}/seed_${SEED}"
    if [[ -f "${OUT}/result.json" ]]; then
        echo "seed ${SEED}: ${OUT}/result.json exists, skipping (never overwritten)"
        continue
    fi
    if [[ -d "${OUT}" ]]; then
        echo "seed ${SEED}: removing an INCOMPLETE earlier attempt in ${OUT} (no result.json yet)"
        rm -rf "${OUT}"
    fi
    echo
    echo "---- variant degree, seed ${SEED} ($(date -u +%H:%M:%SZ)) ----"
    srun "${PYTHON}" -u scripts/03b_reconstruction_experiment.py \
        --variant degree \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUT}" \
        --seed "${SEED}" \
        --subset all \
        --train-molecules 100000 \
        --val-molecules 2000 \
        --steps 6000 \
        --batch-size 32 \
        --eval-every 2000 \
        --device cuda
done

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${ROOT}"/seed_*/result.json
echo "Per-seed results: ${ROOT}/seed_<n>/result.json  (train.log, last.pt, best.pt beside it)"
echo "When BOTH variants have finished: sbatch slurm/03c_summarize_ablation.sh"
echo "=============================================="
