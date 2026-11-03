#!/bin/bash
#SBATCH --job-name=zinc_gen_nqed
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

# ---------------------------------------------------------------------------
# STAGE 13 EVALUATION: joint (N, QED) generation, target achievement and controls.
# CPU only, read-only on every checkpoint, no training.
#
# Order of business:
#   A  three permutation controls on validation molecules - both features, QED alone
#      (N kept true), N alone (QED kept true). The QED-only control is decisive: N and
#      QED are correlated in ZINC, so a model that learned only N still looks conditional
#      when both are permuted. If it fails, the job exits 5 and prints NO QED table.
#   B  decode ceiling: the same decoders on REAL validation embeddings. This is what
#      makes a 0% raw validity interpretable - if real embeddings decode to valid
#      aromatic molecules under the raw argmax, the bond prediction works and the
#      sampled embeddings are the problem; if they do not, the raw argmax is the ceiling.
#   C  generation over the requested (N, QED) grid, optionally at several temperatures,
#      with raw / greedy / degree validity SEPARATE, fallback, aromaticity and QED.
#   D  QED target achievement: MAE, bias, RMSE, hit rate within +-TOLERANCE, correlation
#      and response slope, grouped by N and requested QED, with the WITHIN-N correlation
#      as the headline that the N-QED correlation cannot explain.
#
#   FLOW_CHECKPOINT=...   default outputs/checkpoints/flow_nqed/seed_1/best.pt
#   OUT=...               default outputs/samples/flow_nqed/seed_1
#   TARGET_N=15,23,30     requested atom counts
#   TARGET_QED=0.4,0.6,0.8 requested QED values
#   TEMPERATURES=0.2      comma-separated; e.g. 0.2,0.5,0.85 measures the
#                         validity-versus-aromaticity trade-off in one job
#   COUNT=500             samples per (T, N, QED) cell
#   TOLERANCE=0.1         QED hit-rate tolerance
#   ALLOW_WEAK=1          report a failed control instead of exiting 5
#
#     sbatch slurm/13_generate_nqed.sh
#     sbatch --export=ALL,TEMPERATURES=0.2,0.5,0.85,COUNT=1000 slurm/13_generate_nqed.sh
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
    echo "    5=a conditioning control failed (QED not used beyond N), 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT:-outputs/samples/flow_nqed/seed_1}/generate_nqed.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/13_generate_nqed.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-outputs/checkpoints/flow_nqed/seed_1/best.pt}"
OUT="${OUT:-outputs/samples/flow_nqed/seed_1}"
TARGET_N="${TARGET_N:-15,23,30}"
TARGET_QED="${TARGET_QED:-0.4,0.6,0.8}"
TEMPERATURES="${TEMPERATURES:-0.2}"
COUNT="${COUNT:-500}"
TOLERANCE="${TOLERANCE:-0.1}"
EXTRA=()
if [[ "${ALLOW_WEAK:-0}" == "1" ]]; then EXTRA+=(--allow-weak-conditioning); fi
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Stage 13 evaluation: joint (N, QED) generation"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU on purpose)"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Flow:             ${FLOW_CHECKPOINT} (read-only)"
echo "Grid:             N=${TARGET_N}  QED=${TARGET_QED}  T=${TEMPERATURES}"
echo "Samples per cell: ${COUNT}    QED tolerance +-${TOLERANCE}"
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
    echo "  sbatch slurm/13_flow_nqed_gpu.sh" >&2
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
echo "---- controls, decode ceiling, generation, QED achievement ----"
srun "${PYTHON}" -u scripts/13_generate_nqed.py \
    --flow-checkpoint "${FLOW_CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --target-n "${TARGET_N}" \
    --target-qed "${TARGET_QED}" \
    --temperature "${TEMPERATURES}" \
    --count "${COUNT}" \
    --qed-tolerance "${TOLERANCE}" \
    --batch-size 128 \
    --check-molecules 2000 \
    --ceiling-molecules 1000 \
    --device cpu \
    "${EXTRA[@]+${EXTRA[@]}}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo
echo "QED control + ceiling: ${OUT}/conditioning_check.json"
echo "Report:               ${OUT}/nqed_report.txt"
echo "Data:                 ${OUT}/summary.json, ${OUT}/T<t>_N<n>_QED<q>/metrics.json"
echo
echo "Read, in this order: (1) the qed_only control - without it an apparent QED effect"
echo "is N acting through the N-QED correlation; (2) the decode ceiling row - raw validity"
echo "must be compared with the raw ceiling on REAL embeddings, not with 100%; (3) the"
echo "per-cell table, where raw and degree validity are never merged."
echo "=============================================="
