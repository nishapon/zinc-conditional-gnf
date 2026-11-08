#!/bin/bash
#SBATCH --job-name=zinc_relabel_pool
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# ---------------------------------------------------------------------------
# STEP 1 OF 3 -- COLLECT THE RELABELING POOL. NO TRAINING, NO GRADIENT.
#
# Generates from the FROZEN final checkpoint, decodes with the frozen AE, measures actual
# RDKit QED, and saves the ORIGINAL detached embedding tensors. SMILES are never
# re-encoded: that would change the embeddings and would be a different experiment.
#
# TENSORS ARE LOCAL ONLY. They go to outputs/relabel_pool/, which .gitignore excludes in
# full (directory, not just *.pt), so neither the tensors nor the manifest can be
# committed by accident. The collector itself refuses any --pool-dir that is inside the
# repository but outside outputs/.
#
# Default budget: 3 N x 3 requests x 400 samples = 3,600 generated, of which only the
# ones that decode scorably under the degree assembler carry a label. The report prints
# retained_fraction per cell, which IS the validity-selection bias.
#
# Cost warning: the flow inverse is sequential over nodes, so this is the expensive step.
# If it approaches walltime, resubmit with SAMPLES=200.
#
#     sbatch slurm/34_collect_relabel_pool.sh
#
# Never run this on the login node.
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
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
FLOW="${FLOW:-outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/best.pt}"
AE="${AE:-outputs/ablation/ae_degree_128d/seed_3/last.pt}"
POOL="${POOL:-outputs/relabel_pool/final_step50000}"
TARGET_N="${TARGET_N:-15,23,30}"
QED_REQUESTS="${QED_REQUESTS:-0.4,0.6,0.8}"
SAMPLES="${SAMPLES:-400}"
BATCH="${BATCH:-50}"
TEMP="${TEMP:-0.2}"
HOLDOUT="${HOLDOUT:-0.15}"

EXTRA_ARGS=""
if [[ "${ALLOW_AE_MISMATCH:-0}" == "1" ]]; then
    echo "WARNING: ALLOW_AE_MISMATCH=1 -- provenance mismatch tolerated and recorded"
    EXTRA_ARGS="${EXTRA_ARGS} --allow-ae-mismatch"
fi
mkdir -p outputs/logs "${POOL}"

echo "=============================================="
echo "Relabeling pool collection (no training, no gradient)"
echo "Start:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:      ${SLURM_JOB_ID}  Node: $(hostname)  Partition: ${SLURM_JOB_PARTITION:-?}"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Flow:     ${FLOW}   (frozen, read-only)"
echo "AE:       ${AE}     (frozen, provenance CHECKED)"
echo "Grid:     N=${TARGET_N}  requests=${QED_REQUESTS}  ${SAMPLES}/cell  T=${TEMP}"
echo "Pool:     ${POOL}   (LOCAL ONLY -- git-ignored in full)"
echo "=============================================="

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${FLOW}" "${AE}"; do
    [[ -f "${required}" ]] || { echo "ERROR: ${required} not found." >&2; exit 66; }
done

srun "${PYTHON}" -u scripts/34_collect_relabel_pool.py \
    --flow-checkpoint "${FLOW}" \
    --ae-checkpoint "${AE}" \
    --data-dir "${DATA_DIR}" \
    --pool-dir "${POOL}" \
    --target-n "${TARGET_N}" \
    --qed-requests "${QED_REQUESTS}" \
    --samples-per-cell "${SAMPLES}" \
    --batch-size "${BATCH}" \
    --temperature "${TEMP}" \
    --label-decoder degree \
    --holdout-fraction "${HOLDOUT}" \
    ${EXTRA_ARGS} \
    --device cpu

echo
echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat "${POOL}/pool_summary.txt" 2>/dev/null || true
echo "=============================================="
echo "READ THE QED SUPPORT BLOCK FIRST. It is the CEILING for the fine-tune: relabeling"
echo "cannot address QED values the checkpoint does not already produce. If the mass at"
echo "QED>=0.75 is negligible, the 0.8 request has almost nothing to learn from and the"
echo "fine-tune should not be expected to deliver it."
echo "=============================================="
