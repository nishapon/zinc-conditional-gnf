#!/bin/bash
#SBATCH --job-name=zinc_relabel_finetune
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# STEP 2 OF 3 -- SHORT FORWARD-ONLY FINE-TUNE. THE ONLY TRAINING JOB.
#
#   L = (1 - w) * NLL(real, c_real) + w * NLL(generated, c_actual_decoded)
#
# Forward-only: the gradient path is flow.forward + prior.log_prob. flow.inverse is never
# called here, so none of the four sites that made the v1 run non-finite is reachable.
#
# NOTHING EXISTING IS OVERWRITTEN. The parent checkpoint is opened read-only. The output
# goes to a NEW directory and the script REFUSES to start if that directory already holds
# last.pt or best.pt, unless ALLOW_EXISTING=1 is set deliberately.
#
# THE BUDGET IS FIXED AND IS NOT SEARCHED:
#   400 steps, lr 1e-5 (one hundredth of the original 1e-3), 50 warm-up, clip 5.0,
#   relabel weight 0.5, batches 32 real + 32 relabeled.
# best.pt is selected on REAL validation NLL only -- never on the QED response, which is
# the quantity under test. last.pt is always kept.
#
#     sbatch slurm/35_finetune_relabel.sh
#
# Requires outputs/relabel_pool/... from step 1. Never run training on the login node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Training must never run on the login node." >&2
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
OUT="${OUT:-outputs/checkpoints/flow_nqed_relabel_finetune/seed_3}"
STEPS="${STEPS:-400}"
LR="${LR:-1e-5}"
WARMUP="${WARMUP:-50}"
WEIGHT="${WEIGHT:-0.5}"
REAL_BATCH="${REAL_BATCH:-32}"
POOL_BATCH="${POOL_BATCH:-32}"
EVAL_EVERY="${EVAL_EVERY:-100}"
VAL_MOLECULES="${VAL_MOLECULES:-2000}"
SEED="${SEED:-20261113}"

EXTRA_ARGS=""
if [[ "${ALLOW_AE_MISMATCH:-0}" == "1" ]]; then
    echo "WARNING: ALLOW_AE_MISMATCH=1 -- provenance mismatch tolerated and recorded"
    EXTRA_ARGS="${EXTRA_ARGS} --allow-ae-mismatch"
fi
if [[ "${ALLOW_EXISTING:-0}" == "1" ]]; then
    echo "WARNING: ALLOW_EXISTING=1 -- an existing fine-tune in ${OUT} WILL be overwritten"
    EXTRA_ARGS="${EXTRA_ARGS} --allow-existing"
fi
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Relabeling fine-tune (forward-only conditional NLL + real anchor)"
echo "Start:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:      ${SLURM_JOB_ID}  Node: $(hostname)  Partition: ${SLURM_JOB_PARTITION:-?}"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "Parent:   ${FLOW}   (READ-ONLY, never written)"
echo "Pool:     ${POOL}"
echo "Output:   ${OUT}    (new directory; refuses to overwrite)"
echo "Budget:   ${STEPS} steps  lr=${LR}  warmup=${WARMUP}  weight=${WEIGHT}"
echo "          batches ${REAL_BATCH} real + ${POOL_BATCH} relabeled"
echo "=============================================="

for required in "${DATA_DIR}/splits.pkl" "${FLOW}" "${AE}" \
                "${POOL}/pool.pt" "${POOL}/pool_manifest.json"; do
    [[ -f "${required}" ]] || { echo "ERROR: ${required} not found." >&2; exit 66; }
done

srun "${PYTHON}" -u scripts/35_finetune_relabel.py \
    --flow-checkpoint "${FLOW}" \
    --ae-checkpoint "${AE}" \
    --pool-dir "${POOL}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --max-steps "${STEPS}" \
    --learning-rate "${LR}" \
    --warmup-steps "${WARMUP}" \
    --relabel-weight "${WEIGHT}" \
    --real-batch-size "${REAL_BATCH}" \
    --pool-batch-size "${POOL_BATCH}" \
    --eval-every "${EVAL_EVERY}" \
    --val-molecules "${VAL_MOLECULES}" \
    --seed "${SEED}" \
    ${EXTRA_ARGS} \
    --device auto

echo
echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat "${OUT}/finetune_report.txt" 2>/dev/null || true
echo "=============================================="
echo "This report says NOTHING about whether QED control improved. Check that the real"
echo "validation NLL did not deteriorate badly, then run step 3:"
echo "    sbatch slurm/36_evaluate_relabel.sh"
echo "=============================================="
