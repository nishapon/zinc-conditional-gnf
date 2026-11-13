#!/bin/bash
#SBATCH --job-name=zinc_condition_probe
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# P0 DIAGNOSTIC: WHERE DOES THE CONDITIONAL SIGNAL DIE?  NO TRAINING.
#
# Two stages, both read-only with respect to models, checkpoints and existing results.
# Output goes to NEW directories; nothing existing is touched.
#
#   STAGE A  scripts/41_metric_reproducibility_audit.py
#            NO MODEL AT ALL. Recomputes from saved samples:
#              * the constant-predictor floor under the evaluation's OWN sample-level
#                aggregation. This matters: validity collapses with N, so the cells
#                contribute wildly unequal numbers of scored samples, the pooled target
#                distribution is not uniform, and the minimiser is the WEIGHTED median
#                of the per-sample targets. The earlier 0.1333 assumed uniform weights
#                and is provisional until this runs.
#              * the nine per-condition aromatic-ring distributions, plus per-N, plus
#                the real-ZINC reference at matched N.
#
#   STAGE B  scripts/40_condition_sensitivity_probe.py
#            Draws bounded controlled samples -- unavoidable, because measuring a
#            response needs the prior noise held fixed while only the condition moves.
#            Three arms per N, all sharing one noise draw:
#              QED ARM        requested QED varied, mask and condition N fixed
#              N-CONDITION    the CONDITION's N moved AWAY from the mask's N. The
#              ARM            redundancy test, and it needs no training: the decoder
#                             takes its node count from the MASK, so if moving only the
#                             condition's N leaves the output unchanged, that channel
#                             carries nothing the mask does not already supply.
#              NOISE ARM      condition fixed, second independent draw -- the reference
#                             scale everything is divided by.
#            Measured at three stages: latent embeddings, RAW decode, DEGREE-corrected
#            decode. That localises the loss to the flow, the decoder or the repair.
#
# N and QED are compared PER UNIT of standardised condition change, because "N 15->30"
# and "QED 0.4->0.8" are different units and the raw comparison is meaningless.
#
#     sbatch slurm/40_condition_probe_and_audit.sh
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
SAMPLES_ROOT="${SAMPLES_ROOT:-outputs/samples/qedcontrast_seed3_full_paired}"
AUDIT_OUT="${AUDIT_OUT:-outputs/metrics/metric_audit}"
PROBE_OUT="${PROBE_OUT:-outputs/metrics/condition_sensitivity}"
TARGET_N="${TARGET_N:-15,23,30}"
QED_VALUES="${QED_VALUES:-0.4,0.6,0.8}"
TEMP="${TEMP:-0.2}"
SAMPLES="${SAMPLES:-32}"
DECODER="${DECODER:-degree}"

EXTRA=""
if [[ "${ALLOW_AE_MISMATCH:-0}" == "1" ]]; then
    echo "WARNING: ALLOW_AE_MISMATCH=1 -- mismatch tolerated and recorded"
    EXTRA="${EXTRA} --allow-ae-mismatch"
fi
AUDIT_EXTRA=""
if [[ "${SKIP_REAL_REFERENCE:-0}" == "1" ]]; then
    AUDIT_EXTRA="${AUDIT_EXTRA} --skip-real-reference"
fi
mkdir -p outputs/logs "${AUDIT_OUT}" "${PROBE_OUT}"

echo "=============================================="
echo "P0 conditioning diagnostic (READ-ONLY, no training)"
echo "Start:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:      ${SLURM_JOB_ID}  Node: $(hostname)"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Flow:     ${FLOW}   (frozen)"
echo "Samples:  ${SAMPLES_ROOT}   (stage A reads these, never regenerates)"
echo "Outputs:  ${AUDIT_OUT} and ${PROBE_OUT}"
echo "=============================================="

echo
echo ">>> STAGE A: metric reproducibility audit (no model loaded)"
[[ -d "${SAMPLES_ROOT}" ]] || { echo "ERROR: ${SAMPLES_ROOT} not found." >&2; exit 66; }
srun "${PYTHON}" -u scripts/41_metric_reproducibility_audit.py \
    --samples-root "${SAMPLES_ROOT}" \
    --output-dir "${AUDIT_OUT}" \
    --data-dir "${DATA_DIR}" \
    --target-n "${TARGET_N}" \
    --qed-values "${QED_VALUES}" \
    --temperature "${TEMP}" \
    --decoder "${DECODER}" \
    ${AUDIT_EXTRA}

echo
echo ">>> STAGE B: condition sensitivity probe (bounded controlled generation)"
for required in "${DATA_DIR}/scaler.json" "${FLOW}" "${AE}"; do
    [[ -f "${required}" ]] || { echo "ERROR: ${required} not found." >&2; exit 66; }
done
srun "${PYTHON}" -u scripts/40_condition_sensitivity_probe.py \
    --flow-checkpoint "${FLOW}" \
    --ae-checkpoint "${AE}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${PROBE_OUT}" \
    --target-n "${TARGET_N}" \
    --qed-values "${QED_VALUES}" \
    --samples "${SAMPLES}" \
    --temperature "${TEMP}" \
    ${EXTRA} \
    --device cpu

echo
echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Reports:  ${AUDIT_OUT}/metric_audit.txt"
echo "          ${PROBE_OUT}/condition_sensitivity.txt"
echo "=============================================="
echo "READ IN THIS ORDER"
echo "  1. The sample-weighted constant floor. If it differs from the uniform 0.1333,"
echo "     the earlier 'worse than a constant' claim must be restated with this number."
echo "  2. The ANSWERS block in the probe report. It answers the five questions"
echo "     mechanically, and it is written so it CAN refute the N-dominance hypothesis."
echo "  3. latent SNR vs raw dQED vs degree dQED. A large latent move with an unchanged"
echo "     molecule indicts the decoder; a changed raw molecule with an unchanged"
echo "     corrected one indicts the repair."
echo "  4. The N-condition arm. Only if moving the condition's N alone leaves the"
echo "     output essentially unchanged is the N-conditioning ablation justified."
echo "=============================================="
