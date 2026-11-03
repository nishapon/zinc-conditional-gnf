#!/bin/bash
#SBATCH --job-name=zinc_qed_support
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00

# ---------------------------------------------------------------------------
# STAGE 14 (Part A): is weak QED control the DATA, the TARGETS, the LABELS, or the
# MODEL? Read-only, CPU only, no training, no sampling. Nothing is modified: the 50k
# checkpoints are opened read-only (and only for the stability check).
#
# Produces, under ${OUT}:
#   qed_distribution.png/.pdf   overall QED frequency (train vs validation) and the
#                               per-N QED frequency at each requested N
#   n_qed_heatmap.png/.pdf      counts and WITHIN-N density on the (N, QED) grid, with
#                               the conditional 5/50/95th percentiles and the requested
#                               targets marked
#   generated_overlay.png/.pdf  the 50k run's generated QED distribution at each target
#                               over the real distribution at the SAME N
#   qed_support.txt/.json       per-N support table, (N, QED) feasibility verdicts,
#                               the RDKit label check and the stability check
#
# The four explanations are tested separately:
#   data imbalance      per-N counts and high-QED mass (>=0.7, >=0.8)
#   infeasible targets  each requested (N, QED) classified against the empirical
#                       conditional distribution at that N: well_represented,
#                       thin_tail or outside_support
#   label / scaler bug  QED recomputed by RDKit from the STORED GRAPH (atom types +
#                       bond matrix, i.e. what the model trains on - not the SMILES)
#                       and the standardised condition re-derived from scaler.json
#   the model           generated vs real QED as DISTRIBUTIONS (quantiles, histogram
#                       overlap, Wasserstein), not as means
#
# It also checks the numerical forward/inverse stability of the 50k checkpoint, because
# log|det J|/node around 180 only matters if the inverse actually degrades - and the
# whitening term sum(log std) is a constant of the data, not a blow-up, so it is
# subtracted and both numbers are printed.
#
#   OUT=outputs/figures/qed_support
#   TARGET_N=15,23,30        requested atom counts to analyse
#   TARGET_QED=0.4,0.6,0.8   requested QED values
#   GENERATED=...            the 50k joint run's sample root (its T*_N*_QED* cells are
#                            found automatically); empty = skip the overlay
#   FLOW_CHECKPOINT=...      checkpoint for the stability check; empty = skip
#
#     sbatch slurm/14_qed_support_analysis.sh
#
# Never run this file with `bash` directly on the login node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; this reads the whole" >&2
    echo "training split and must not run on the login node." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "rc: 2=missing data, 4=bad args, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT:-outputs/figures/qed_support}/qed_support.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/14_qed_support_analysis.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
OUT="${OUT:-outputs/figures/qed_support}"
TARGET_N="${TARGET_N:-15,23,30}"
TARGET_QED="${TARGET_QED:-0.4,0.6,0.8}"
GENERATED="${GENERATED:-outputs/samples/flow_nqed/seed_1}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-outputs/checkpoints/flow_nqed/seed_1/best.pt}"
EXTRA=()
if [[ -n "${GENERATED}" && -d "${GENERATED}" ]]; then
    EXTRA+=(--generated-root "${GENERATED}")
else
    echo "note: ${GENERATED} not found; running without the generated overlay"
fi
if [[ -n "${FLOW_CHECKPOINT}" && -f "${FLOW_CHECKPOINT}" ]]; then
    EXTRA+=(--flow-checkpoint "${FLOW_CHECKPOINT}")
else
    echo "note: ${FLOW_CHECKPOINT} not found; skipping the stability check"
fi
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Stage 14 Part A: QED support analysis"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU on purpose)"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Data:             ${DATA_DIR} (read-only; test split never loaded)"
echo "Requested:        N=${TARGET_N}  QED=${TARGET_QED}"
echo "Generated root:   ${GENERATED:-<none>}"
echo "Stability check:  ${FLOW_CHECKPOINT:-<none>} (read-only)"
echo "Output:           ${OUT}"
echo "=============================================="
"${PYTHON}" -V

if [[ ! -f "${DATA_DIR}/splits.pkl" ]]; then
    echo "ERROR: ${DATA_DIR}/splits.pkl not found." >&2
    exit 66
fi

"${PYTHON}" - <<'PYCHECK'
import rdkit, numpy, scipy, matplotlib
print("rdkit", rdkit.__version__, "| numpy", numpy.__version__,
      "| scipy", scipy.__version__, "| matplotlib", matplotlib.__version__)
PYCHECK

echo
echo "---- analysis ----"
srun "${PYTHON}" -u scripts/14_qed_support_analysis.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --target-n "${TARGET_N}" \
    --target-qed "${TARGET_QED}" \
    --label-sample 2000 \
    --stability-molecules 512 \
    --device cpu \
    "${EXTRA[@]+${EXTRA[@]}}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo
echo "READ IN THIS ORDER:"
echo "  1. ${OUT}/qed_support.txt -> the per-N support table and the FEASIBILITY column."
echo "     A request marked thin_tail or outside_support cannot be blamed on the model:"
echo "     the data has (almost) no molecules there, so its MAE is a property of the ask."
echo "  2. ${OUT}/n_qed_heatmap.png -> where the requested stars sit against the"
echo "     conditional 5/50/95th percentiles of QED given N."
echo "  3. ${OUT}/generated_overlay.png -> whether the generated distribution is shifted,"
echo "     narrowed, or simply elsewhere. A matching mean with a 3x narrow distribution is"
echo "     a different failure from a mean that misses the target."
echo "  4. the label line: if max |RDKit - stored| is ~1e-16 the labels and scaler are fine"
echo "     and that explanation is closed."
echo "  5. the stability line: if both round trips are ~1e-6 the log-determinant warning is"
echo "     cosmetic, and inversion is NOT the cause of poor samples."
echo "=============================================="
