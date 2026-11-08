#!/bin/bash
#SBATCH --job-name=zinc_relabel_eval
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# ---------------------------------------------------------------------------
# STEP 3 OF 3 -- THREE-WAY EVALUATION UNDER COMMON NOISE. READ-ONLY, NO TRAINING.
#
# Compares the fine-tuned model against the ORIGINAL final checkpoint and the EARLY
# contrastive checkpoint. One prior-noise draw per N is shared across every model and
# every requested QED, which gives
#   * a per-molecule PAIRED QED response at fixed N (jointly valid pairs only), and
#   * cross-model comparability that is not sampling noise between runs.
#
# LEAKAGE GUARD: the evaluation noise seed must differ from the pool's collection seed.
# The script exits 3 if they match, because evaluating on the fine-tune's own training
# draws would manufacture a response out of memorisation.
#
# THE LABEL-DECODER QUESTION: the fine-tune's labels are QED under the degree-constrained
# assembler, so the response is measured under raw, constrained AND degree. A gain that
# appears only under degree is a correction-dependent distribution, not intrinsic
# conditioning, and the report names it as such.
#
# SUCCESS CRITERION (fixed in code, not chosen after the fact): improved per-molecule
# paired response at fixed N with a bootstrap CI excluding zero, AND raw validity down by
# no more than 3 points, AND aromaticity/uniqueness/diversity retention within margin.
# A lower MAE alone does NOT count and is excluded from the pass condition.
#
#     sbatch slurm/36_evaluate_relabel.sh
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
AE="${AE:-outputs/ablation/ae_degree_128d/seed_3/last.pt}"
FINAL="${FINAL:-outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/best.pt}"
EARLY="${EARLY:-outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/step_4000.pt}"
TUNED="${TUNED:-outputs/checkpoints/flow_nqed_relabel_finetune/seed_3/last.pt}"
POOL="${POOL:-outputs/relabel_pool/final_step50000}"
OUT="${OUT:-outputs/metrics/relabel_evaluation}"
TARGET_N="${TARGET_N:-15,23,30}"
QED_VALUES="${QED_VALUES:-0.4,0.6,0.8}"
SAMPLES="${SAMPLES:-100}"
BATCH="${BATCH:-50}"
TEMP="${TEMP:-0.2}"
VAL_MOLECULES="${VAL_MOLECULES:-2000}"

EXTRA_ARGS=""
if [[ "${ALLOW_AE_MISMATCH:-0}" == "1" ]]; then
    echo "WARNING: ALLOW_AE_MISMATCH=1 -- provenance mismatch tolerated and recorded"
    EXTRA_ARGS="${EXTRA_ARGS} --allow-ae-mismatch"
fi
if [[ "${SKIP_REAL_NLL:-0}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-real-nll"
fi
MODEL_ARGS=(--model "finetuned=${TUNED}" --model "final=${FINAL}")
if [[ -f "${EARLY}" ]]; then
    MODEL_ARGS+=(--model "early=${EARLY}")
else
    echo "NOTE: ${EARLY} not found; the early checkpoint will be omitted."
    echo "      Set EARLY=<path> to include it. Check the run directory for step_*.pt."
fi
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Relabeling evaluation (READ-ONLY, common noise, no training)"
echo "Start:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:      ${SLURM_JOB_ID}  Node: $(hostname)  Partition: ${SLURM_JOB_PARTITION:-?}"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Tuned:    ${TUNED}"
echo "Final:    ${FINAL}"
echo "Early:    ${EARLY}"
echo "Grid:     N=${TARGET_N}  QED=${QED_VALUES}  ${SAMPLES}/cell  T=${TEMP}"
echo "Output:   ${OUT}"
echo "=============================================="

for required in "${DATA_DIR}/splits.pkl" "${AE}" "${TUNED}" "${FINAL}"; do
    [[ -f "${required}" ]] || { echo "ERROR: ${required} not found." >&2; exit 66; }
done

srun "${PYTHON}" -u scripts/36_evaluate_relabel.py \
    "${MODEL_ARGS[@]}" \
    --reference final \
    --candidate finetuned \
    --ae-checkpoint "${AE}" \
    --pool-dir "${POOL}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --target-n "${TARGET_N}" \
    --qed-values "${QED_VALUES}" \
    --samples-per-cell "${SAMPLES}" \
    --batch-size "${BATCH}" \
    --temperature "${TEMP}" \
    --headline degree \
    --val-molecules "${VAL_MOLECULES}" \
    ${EXTRA_ARGS} \
    --device cpu

echo
echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat "${OUT}/relabel_evaluation.txt" 2>/dev/null || true
echo "=============================================="
echo "HOW TO READ THE VERDICT"
echo "  1. PAIRED response at fixed N, candidate vs reference, with its CI. This is the"
echo "     criterion. A lower MAE on its own is NOT success."
echo "  2. label-decoder specificity. If the response improved under degree but not"
echo "     under raw and constrained, the model learned a correction-dependent"
echo "     distribution and this is a NEGATIVE result."
echo "  3. raw validity drop and quality retention. QED bought by destroying chemistry"
echo "     does not count."
echo "  4. real validation NLL vs the original. A large deterioration means the anchor"
echo "     failed and the fine-tune drifted off the real manifold."
echo
echo "If the verdict is NO-GO, we stop experimenting and consolidate the results."
echo "=============================================="
