#!/bin/bash
#SBATCH --job-name=zinc_gen_ncond
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# ---------------------------------------------------------------------------
# STAGE 12 EVALUATION: conditioning verification + N-conditioned generation.
# CPU only, read-only on every checkpoint; no training.
#
# First it VERIFIES the conditioning on validation molecules (same molecules and noise,
# their own conditions versus conditions permuted across molecules) and inspects
# mu(N)/sigma(N). If the model cannot beat its own permuted-condition control, the job
# exits 5 and reports no generation numbers - because generated atom counts would match
# the request even if N were ignored (N sets how many node vectors are drawn and the
# decoder emits one atom per vector), so size agreement is never the evidence.
#
# Then it samples COUNT molecules at each requested N and evaluates each size on its own:
# delivered N before and after decoding, raw / greedy / degree-constrained validity kept
# strictly apart, degree fallback, QED against real training molecules of the SAME size,
# uniqueness, novelty (training split only) and internal diversity.
#
#   FLOW_CHECKPOINT=...   default: outputs/checkpoints/flow_ncond/seed_1/best.pt
#   OUT=...               default: outputs/samples/flow_ncond/seed_1
#   TARGET_N=...          comma-separated requested counts; empty = the training
#                         distribution's 5/25/50/75/95th percentiles
#   COUNT=1000            samples per requested N
#   TEMPERATURE=0.2       the frozen unconditional result used 0.2
#   ALLOW_WEAK=1          report a failed verification instead of exiting 5
#
#     sbatch slurm/12_generate_ncond.sh
#     sbatch --export=ALL,TARGET_N=15,23,30,COUNT=2000 slurm/12_generate_ncond.sh
#
# Never run this file with `bash` directly on the login node.
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
    echo "rc: 2=missing checkpoint/data, 3=autoencoder mismatch, 4=bad args," >&2
    echo "    5=conditioning verification failed (the model is not using N), 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT:-outputs/samples/flow_ncond/seed_1}/generate_ncond.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/12_generate_ncond.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-outputs/checkpoints/flow_ncond/seed_1/best.pt}"
OUT="${OUT:-outputs/samples/flow_ncond/seed_1}"
COUNT="${COUNT:-1000}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TARGET_N="${TARGET_N:-}"
EXTRA=()
if [[ -n "${TARGET_N}" ]]; then EXTRA+=(--target-n "${TARGET_N}"); fi
if [[ "${ALLOW_WEAK:-0}" == "1" ]]; then EXTRA+=(--allow-weak-conditioning); fi
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Stage 12 evaluation: N-conditioned generation"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU on purpose)"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Flow:             ${FLOW_CHECKPOINT} (read-only)"
echo "Requested N:      ${TARGET_N:-training-distribution percentiles}"
echo "Samples per N:    ${COUNT}   temperature ${TEMPERATURE}"
echo "Output:           ${OUT}"
echo "=============================================="
"${PYTHON}" -V

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: ${required} not found." >&2
        exit 66
    fi
done
if [[ ! -f "${FLOW_CHECKPOINT}" ]]; then
    echo "ERROR: ${FLOW_CHECKPOINT} not found. Train it first:" >&2
    echo "  sbatch slurm/12_flow_ncond_gpu.sh" >&2
    exit 66
fi

"${PYTHON}" - <<'PYCHECK'
import sys
import torch, rdkit, numpy, scipy
try:
    from scipy.optimize import milp
except ImportError as error:
    sys.exit(f"ERROR: scipy.optimize.milp unavailable ({error}); install scipy >= 1.9.")
print("torch", torch.__version__, "| rdkit", rdkit.__version__, "| scipy", scipy.__version__)
PYCHECK

echo
echo "---- verification + generation ----"
srun "${PYTHON}" -u scripts/12_generate_ncond.py \
    --flow-checkpoint "${FLOW_CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --count "${COUNT}" \
    --temperature "${TEMPERATURE}" \
    --batch-size 128 \
    --check-molecules 2000 \
    --device cpu \
    "${EXTRA[@]+${EXTRA[@]}}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo
echo "Verdict:   ${OUT}/conditioning_check.txt   (section 1 is the evidence)"
echo "Per-N:     ${OUT}/ncond_report.txt"
echo "Data:      ${OUT}/summary.json, ${OUT}/N<k>/{metrics.json,samples.jsonl}"
echo
echo "Read raw validity and degree validity as DIFFERENT quantities, and read the"
echo "delivered-N column as a decoder property: it counts an invalid sample as a miss,"
echo "so it tracks validity rather than size control."
echo "=============================================="
