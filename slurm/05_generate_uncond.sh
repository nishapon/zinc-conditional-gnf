#!/bin/bash
#SBATCH --job-name=zinc_flow_generate
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

# ---------------------------------------------------------------------------
# Sample and score molecules from a trained unconditional flow. CPU only: the
# flow inverse is tiny, and the cost is the degree-constrained integer program
# (about 10 ms per molecule) plus RDKit, both CPU-bound. This keeps the GPU
# free for training.
#
# For every requested temperature it writes validity (raw / greedy-constrained
# / degree-constrained, reported separately), uniqueness, novelty against the
# TRAINING set, N.U.V., internal diversity, scaffold counts and the decoder
# fallback rate.
#
#   SEEDS="1"                     flow seeds to sample from
#   COUNT=10000                   molecules per temperature
#   TEMPERATURES="0.7,0.85,1.0"   comma-separated
#
#     sbatch slurm/05_generate_uncond.sh
#     sbatch --export=ALL,COUNT=2000,TEMPERATURES="0.6,0.7,0.85,1.0" slurm/05_generate_uncond.sh
#
# Choose ONE temperature from this sweep and report every metric at it; never
# quote validity from one temperature and uniqueness from another. The test
# split is never loaded: novelty is measured against training molecules.
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
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "generate rc: 2=missing checkpoint/data, 3=autoencoder mismatch, 4=bad args, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
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
[[ -n "${PYTHON}" && -x "${PYTHON}" ]] || { echo "ERROR: zinc_gnf python not found" >&2; exit 65; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
AE_CHECKPOINT="${ZINC_GNF_AE:-outputs/ablation/ae_degree/seed_3/last.pt}"
ROOT="outputs/checkpoints/flow_uncond"
SAMPLES="outputs/samples/flow_uncond"
SEEDS="${SEEDS:-1}"
COUNT="${COUNT:-10000}"
TEMPERATURES="${TEMPERATURES:-0.7,0.85,1.0}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pt}"
mkdir -p outputs/logs "${SAMPLES}"

echo "=============================================="
echo "Generation from the unconditional flow"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU on purpose)"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Seeds:            ${SEEDS}"
echo "Molecules:        ${COUNT} per temperature; temperatures ${TEMPERATURES}"
echo "Checkpoint:       ${CHECKPOINT_NAME}"
echo "=============================================="
"${PYTHON}" -V

# Dependency preflight. scipy's MILP solver is what makes the degree-constrained decoder
# exact; without it every molecule would quietly fall back to greedy decoding and the
# headline number would describe a different decoder.
"${PYTHON}" - <<'PYCHECK'
import sys
import torch, rdkit, numpy, scipy
print("torch", torch.__version__, "| rdkit", rdkit.__version__, "| scipy", scipy.__version__)
try:
    from scipy.optimize import milp
except ImportError as error:
    sys.exit(f"ERROR: scipy.optimize.milp is unavailable ({error}); the degree-constrained "
             "decoder would fall back to greedy for every molecule. Install scipy >= 1.9.")
print("dependency check ok")
PYCHECK

for SEED in ${SEEDS}; do
    FLOW="${ROOT}/seed_${SEED}/${CHECKPOINT_NAME}"
    if [[ ! -f "${FLOW}" ]]; then
        echo "ERROR: ${FLOW} not found (train it first: sbatch slurm/04_flow_uncond_gpu.sh)" >&2
        exit 66
    fi
    OUT="${SAMPLES}/seed_${SEED}"
    echo
    echo "---- sampling from seed ${SEED} ($(date -u +%H:%M:%SZ)) ----"
    srun "${PYTHON}" -u scripts/05_generate.py \
        --flow-checkpoint "${FLOW}" \
        --ae-checkpoint "${AE_CHECKPOINT}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUT}" \
        --count "${COUNT}" \
        --temperature "${TEMPERATURES}" \
        --batch-size 128 \
        --seed 0 \
        --headline degree \
        --device cpu
done

echo
echo "---- aggregating across seeds ----"
srun "${PYTHON}" -u scripts/06_summarize_generation.py --root "${SAMPLES}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Per seed/temperature: ${SAMPLES}/seed_<n>/T<temp>/{metrics.json,metrics.txt,valid_smiles.txt,samples.jsonl}"
echo "Across seeds:         ${SAMPLES}/generation_summary.{json,txt}"
echo
echo "Read the three validity numbers separately: 'raw' is the decoder's argmax"
echo "with no chemistry, 'constrained' is greedy valence-aware assembly, 'degree'"
echo "is the degree-constrained program. Reconstruction quality (93.4% exact) is"
echo "NOT a prediction of generation validity."
echo "=============================================="
