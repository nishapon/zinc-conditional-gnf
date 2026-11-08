#!/bin/bash
#SBATCH --job-name=zinc_qed_support_rerank
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# ---------------------------------------------------------------------------
# QED SUPPORT, DECODER AGREEMENT AND BEST-OF-k RERANK. READ-ONLY. NO MODEL.
#
# Reads the ALREADY SAVED sample grids. It loads no checkpoint, imports no torch,
# generates nothing and trains nothing. The only writes are
# ${OUT}/qed_support_rerank.json and ${OUT}/qed_support_rerank.txt.
#
# Best-of-k reranking is INFERENCE-TIME POSTPROCESSING, not intrinsic conditional
# generation. It is a finite-candidate performance reference at a stated budget, NOT an
# upper bound on what a retrained conditional model could reach.
#
# Inputs (both must already exist; this script never regenerates them):
#   EARLY=outputs/samples/qedcontrast_seed3_early_paired     step ~4,000
#   FINAL=outputs/samples/qedcontrast_seed3_full_paired      step ~50,000
# Each is expected to hold the N={15,23,30} x QED={0.4,0.6,0.8} grid as
# T0.2_N<n>_QED<q>/samples.jsonl with 100 candidates per condition.
#
#     sbatch slurm/33_qed_support_and_rerank.sh
#
# This job is small -- file reading plus RDKit over a few thousand molecules, minutes at
# most. Because it is strictly read-only and loads no model, running the python directly
# on a login node would not violate the no-training rule, but SLURM is used here for
# consistency with every other wrapper in the repository.
#
# TEMP must match the temperature the grids were generated at, because it is part of the
# directory name. If the cells are not found, check TEMP before anything else.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch." >&2
    exit 64
fi

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
[[ -n "${PYTHON}" && -x "${PYTHON}" ]] || {
    echo "ERROR: could not find the zinc_gnf env python; resubmit with ZINC_GNF_PYTHON=..." >&2
    exit 65; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
EARLY="${EARLY:-outputs/samples/qedcontrast_seed3_early_paired}"
FINAL="${FINAL:-outputs/samples/qedcontrast_seed3_full_paired}"
OUT="${OUT:-outputs/metrics/qed_support_rerank}"
TEMP="${TEMP:-0.2}"
TARGET_N="${TARGET_N:-15,23,30}"
QED_VALUES="${QED_VALUES:-0.4,0.6,0.8}"
K_VALUES="${K_VALUES:-1,2,5,10,20}"
PARTITIONS="${PARTITIONS:-5}"
SEED="${SEED:-20261108}"

EXTRA_ARGS=""
if [[ "${SKIP_NOVELTY:-0}" == "1" ]]; then
    echo "NOTE: SKIP_NOVELTY=1 -- novelty will not be computed and the dataset is not read"
    EXTRA_ARGS="${EXTRA_ARGS} --skip-novelty"
fi
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "QED support / decoder agreement / best-of-k rerank (READ-ONLY, no model)"
echo "Start:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:      ${SLURM_JOB_ID}  Node: $(hostname)  Partition: ${SLURM_JOB_PARTITION:-?}"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Early:    ${EARLY}"
echo "Final:    ${FINAL}"
echo "Grid:     T=${TEMP}  N=${TARGET_N}  QED=${QED_VALUES}"
echo "Rerank:   k=${K_VALUES}  partitions=${PARTITIONS}  seed=${SEED}"
echo "Output:   ${OUT}  (only qed_support_rerank.json / .txt are written)"
echo "=============================================="

for root in "${EARLY}" "${FINAL}"; do
    [[ -d "${root}" ]] || { echo "ERROR: ${root} not found." >&2; exit 66; }
done
if [[ "${SKIP_NOVELTY:-0}" != "1" ]]; then
    [[ -f "${DATA_DIR}/splits.pkl" ]] || {
        echo "ERROR: ${DATA_DIR}/splits.pkl not found; rerun with SKIP_NOVELTY=1 to drop novelty." >&2
        exit 66; }
fi

srun "${PYTHON}" -u scripts/33_qed_support_and_rerank.py \
    --run "early=${EARLY}" \
    --run "final=${FINAL}" \
    --output-dir "${OUT}" \
    --data-dir "${DATA_DIR}" \
    --target-n "${TARGET_N}" \
    --qed-values "${QED_VALUES}" \
    --temperature "${TEMP}" \
    --decoders raw,constrained,degree \
    --headline degree \
    --k-values "${K_VALUES}" \
    --partitions "${PARTITIONS}" \
    --seed "${SEED}" \
    ${EXTRA_ARGS}

echo
echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Report:   ${OUT}/qed_support_rerank.txt"
echo "Data:     ${OUT}/qed_support_rerank.json"
echo
cat "${OUT}/qed_support_rerank.txt" 2>/dev/null || true
echo "=============================================="
echo "HOW TO READ IT"
echo "  1. QED SUPPORT decides whether relabeling has anything to learn from. Look at"
echo "     the within-N IQR and the mass at QED>=0.75, per decoder."
echo "  2. CROSS-DECODER AGREEMENT decides whether the relabeling LABEL is even well"
echo "     defined. A large per-embedding |dQED| is not fixable by retraining."
echo "  3. RERANK k=1 IS the native model. The QED gain at larger k is selection on the"
echo "     reported quantity and is expected; read the selection cost and the metrics"
echo "     rerank does not optimise (validity, aromaticity, uniqueness, novelty,"
echo "     diversity) and the accepted/ceiling column."
echo "  4. Rerank is POSTPROCESSING and a finite-candidate reference. It constrains but"
echo "     does not refute a retrained conditional model."
echo "=============================================="
