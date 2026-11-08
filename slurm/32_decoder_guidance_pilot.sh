#!/bin/bash
#SBATCH --job-name=zinc_guidance_pilot
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# STAGE 1 -- DECODER-AWARE GUIDANCE PILOT. READ-ONLY, BOUNDED, NO TRAINING.
#
# This measures INFERENCE-TIME GUIDANCE on a FROZEN flow and a FROZEN autoencoder. It
# does NOT demonstrate improved intrinsic conditional generation, and the report says so
# in its own header. Stage 2 (distilling the guidance into the flow) is NOT implemented.
#
# Nothing is trained. No checkpoint is written. No existing result directory is touched.
# The only writes are ${OUT}/guidance_pilot.json and ${OUT}/guidance_pilot.txt. The test
# split is never loaded.
#
# WHAT IT DOES
#   PHASE A  fits a ridge readout on TRAIN real embeddings against the dataset's RDKit
#            QED labels, then validates it on (a) held-out REAL validation embeddings,
#            per N, and (b) unmodified FLOW-GENERATED embeddings against their ACTUAL
#            decoded RDKit QED. Both feature sets (`moments`, `structural`) are measured,
#            plus an aromatic-ring readout. This is a HARD GATE: if no feature set meets
#            the precommitted thresholds on GENERATED embeddings, the QED arms are
#            SKIPPED and only the manifold arm (no learned critic) runs.
#   PHASE B  five arms from IDENTICAL initial embeddings and noise:
#              native / qed_only / manifold_only / combined / random (matched norm)
#            reporting actual decoded RDKit QED, QED MAE vs the request, RAW and
#            CORRECTED validity separately, aromatic rings, uniqueness, internal
#            diversity, surrogate-predicted QED beside the actual value (exploitation
#            detector), displacement, trust-region saturation, non-finite step count and
#            MEASURED per-arm runtime.
#
# THIS IS THE BOUNDED PILOT, NOT THE FULL EVALUATION.
#   SAMPLES=8 by default: 3 N x 3 QED x 5 arms x 8 samples. RUNTIME IS NOT ASSUMED --
#   the report prints the measured total and the implied multiplier for 100 samples/cell.
#   DO NOT scale up until the pilot shows actual RDKit QED moving toward the target with
#   raw validity not below native. To scale later: SAMPLES=100 sbatch ...
#
# Autoencoder provenance is CHECKED, not waived. Job 366922 confirmed the recorded and
# actual hashes match (93a89bba...), so this should run with no override. A mismatch
# means stop and investigate -- do not set ALLOW_AE_MISMATCH=1 to get past it.
#
#     sbatch slurm/32_decoder_guidance_pilot.sh
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
OUT="${OUT:-outputs/metrics/decoder_guidance_pilot}"
TARGET_N="${TARGET_N:-15,23,30}"
QED_VALUES="${QED_VALUES:-0.4,0.6,0.8}"
SAMPLES="${SAMPLES:-8}"
STEPS="${STEPS:-15}"
STEP_SIZE="${STEP_SIZE:-0.15}"
TRUST_SCALE="${TRUST_SCALE:-1.0}"
LAMBDA_QED="${LAMBDA_QED:-1.0}"
LAMBDA_MANIFOLD="${LAMBDA_MANIFOLD:-1.0}"
SUR_TRAIN="${SUR_TRAIN:-3000}"
SUR_VAL="${SUR_VAL:-800}"
SUR_GEN="${SUR_GEN:-12}"
REAL_REF="${REAL_REF:-256}"

EXTRA_ARGS=""
if [[ "${ALLOW_AE_MISMATCH:-0}" == "1" ]]; then
    echo "WARNING: ALLOW_AE_MISMATCH=1 -- provenance mismatch will be tolerated and recorded"
    EXTRA_ARGS="${EXTRA_ARGS} --allow-ae-mismatch"
fi
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Decoder-aware guidance pilot, Stage 1 (READ-ONLY, no training)"
echo "INFERENCE-TIME GUIDANCE -- not intrinsic conditional generation."
echo "Start:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:      ${SLURM_JOB_ID}  Node: $(hostname)  Partition: ${SLURM_JOB_PARTITION:-?}"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Flow:     ${FLOW}"
echo "AE:       ${AE}   (provenance CHECKED)"
echo "Grid:     N=${TARGET_N}  QED=${QED_VALUES}  samples/cell=${SAMPLES}"
echo "Guidance: steps=${STEPS} step=${STEP_SIZE} trust_scale=${TRUST_SCALE}"
echo "          lambda_qed=${LAMBDA_QED} lambda_manifold=${LAMBDA_MANIFOLD}"
echo "Surrogate:train=${SUR_TRAIN} val=${SUR_VAL} generated=${SUR_GEN}"
echo "Output:   ${OUT}  (only guidance_pilot.json / .txt are written)"
echo "=============================================="

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${FLOW}" "${AE}"; do
    [[ -f "${required}" ]] || { echo "ERROR: ${required} not found." >&2; exit 66; }
done

srun "${PYTHON}" -u scripts/32_decoder_guidance_pilot.py \
    --flow-checkpoint "${FLOW}" \
    --ae-checkpoint "${AE}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --target-n "${TARGET_N}" \
    --qed-values "${QED_VALUES}" \
    --samples-per-cell "${SAMPLES}" \
    --guidance-steps "${STEPS}" \
    --step-size "${STEP_SIZE}" \
    --trust-radius-scale "${TRUST_SCALE}" \
    --lambda-qed "${LAMBDA_QED}" \
    --lambda-manifold "${LAMBDA_MANIFOLD}" \
    --surrogate-train-molecules "${SUR_TRAIN}" \
    --surrogate-val-molecules "${SUR_VAL}" \
    --surrogate-generated-samples "${SUR_GEN}" \
    --real-reference-molecules "${REAL_REF}" \
    --batch-size 16 \
    --temperature 0.2 \
    --decoders degree,constrained \
    --eps-seed 20261114 \
    ${EXTRA_ARGS} \
    --device cpu

echo
echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Report:   ${OUT}/guidance_pilot.txt"
echo "Data:     ${OUT}/guidance_pilot.json"
echo
tail -80 "${OUT}/guidance_pilot.txt" 2>/dev/null || true
echo "=============================================="
echo "HOW TO READ IT"
echo "  1. PHASE A first. If 'QED arms enabled: False', the surrogate did not transfer"
echo "     to generated embeddings and the QED numbers were deliberately NOT produced."
echo "  2. Compare ACTUAL decoded RDKit QED, not surrogate QED. A large"
echo "     'surrogate-actual' gap is surrogate exploitation, i.e. a negative result."
echo "  3. The 'random' arm is the control. If it moves QED as much as the guided arms,"
echo "     the effect is displacement, not conditioning."
echo "  4. Raw validity and corrected validity are separate. QED bought by destroying"
echo "     raw validity or uniqueness is not a gain."
echo "  5. Check the MEASURED runtime line before scaling to SAMPLES=100."
echo "=============================================="
