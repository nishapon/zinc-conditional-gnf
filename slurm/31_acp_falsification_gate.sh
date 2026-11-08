#!/bin/bash
#SBATCH --job-name=zinc_acp_gate
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# ACP FALSIFICATION GATE -- READ-ONLY, BOUNDED, NO TRAINING.
#
# Decides whether a uniform latent translation at a FIXED reference condition
# (QED=0.6) changes decoded RDKit QED. That is the only thing a conditional prior mean
# can do, so a negative result falsifies the anchored-conditional-prior proposal and we
# drop it immediately rather than spending days on it.
#
# Writes ONLY to $OUT (default outputs/metrics/acp_gate), a new directory. No checkpoint,
# dataset or existing result is modified. The test split is never loaded.
#
# Defaults:
#   FLOW=outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/best.pt   (the final 128D)
#   AE=outputs/ablation/ae_degree_128d/seed_3/last.pt
#   OUT=outputs/metrics/acp_gate
#   REAL=32  GEN=32  TRAIN_BAND=400
#
# Autoencoder provenance is CHECKED. Job 366922 confirmed the hashes match
# (93a89bba...), so the gate should run WITHOUT an override. If it reports a mismatch,
# stop and investigate rather than setting ALLOW_AE_MISMATCH=1.
#
#     sbatch slurm/31_acp_falsification_gate.sh
#
# Cost control: the inverse pass is sequential over nodes (ARM B), so cost scales with
# directions x deltas x N x arms x samples. Defaults are ~3 x 7 x 3 x 2 x 32. If the job
# approaches walltime, resubmit with GEN=16 REAL=16 or DELTAS="-1.0,0.0,1.0".
#
# Never run this on the login node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; do not run on the login node." >&2
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
OUT="${OUT:-outputs/metrics/acp_gate}"
REAL="${REAL:-32}"
GEN="${GEN:-32}"
TRAIN_BAND="${TRAIN_BAND:-400}"
DELTAS="${DELTAS:--1.5,-1.0,-0.5,0.0,0.5,1.0,1.5}"
QED_REF="${QED_REF:-0.6}"

EXTRA_ARGS=""
if [[ "${ALLOW_AE_MISMATCH:-0}" == "1" ]]; then
    echo "WARNING: ALLOW_AE_MISMATCH=1 -- provenance mismatch will be tolerated and recorded"
    EXTRA_ARGS="${EXTRA_ARGS} --allow-ae-mismatch"
fi
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "ACP falsification gate (READ-ONLY, no training)"
echo "Start:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:      ${SLURM_JOB_ID}  Node: $(hostname)  Partition: ${SLURM_JOB_PARTITION:-?}"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Flow:     ${FLOW}"
echo "AE:       ${AE}"
echo "Ref cond: QED=${QED_REF} (fixed for forward AND inverse)"
echo "Deltas:   ${DELTAS}   real=${REAL} generated=${GEN} train/band=${TRAIN_BAND}"
echo "Output:   ${OUT}  (new directory; nothing else is touched)"
echo "=============================================="

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${FLOW}" "${AE}"; do
    [[ -f "${required}" ]] || { echo "ERROR: ${required} not found." >&2; exit 66; }
done

srun "${PYTHON}" -u scripts/31_acp_falsification_gate.py \
    --flow-checkpoint "${FLOW}" \
    --ae-checkpoint "${AE}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --target-n 15,23,30 \
    --qed-reference "${QED_REF}" \
    --qed-low 0.4 \
    --qed-high 0.8 \
    --shift-deltas "${DELTAS}" \
    --real-molecules "${REAL}" \
    --generated-samples "${GEN}" \
    --train-per-band "${TRAIN_BAND}" \
    --temperature 0.2 \
    --decoders degree,constrained \
    --directions real_qed,prior_delta,random \
    ${EXTRA_ARGS} \
    --device cpu

echo
echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Report:   ${OUT}/acp_gate.txt"
echo "Data:     ${OUT}/acp_gate.json"
echo
cat "${OUT}/acp_gate.txt" 2>/dev/null | tail -60 || true
echo "=============================================="
echo "READ THE VERDICT BLOCK. If it is NO-GO, reject ACP and move on -- the gate is"
echo "designed to reject cheaply, and a negative result is a real finding about where"
echo "the conditioning bottleneck is NOT."
echo "=============================================="
