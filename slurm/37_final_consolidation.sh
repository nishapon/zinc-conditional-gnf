#!/bin/bash
#SBATCH --job-name=zinc_final_consolidation
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

# ---------------------------------------------------------------------------
# FINAL VERIFICATION AND CONSOLIDATION. READ-ONLY. NO MODEL IS RUN.
#
# Reads the JSON artefacts earlier stages wrote and produces the definitive comparison
# table, the CORRECTED model-to-model statistics, and a checkpoint identity audit.
# Nothing is trained, sampled or decoded; the only model access is torch.load on the
# fine-tune checkpoints to compare their weights.
#
# WHY IT EXISTS. scripts/36 tested `candidate_CI_excludes_zero`, which asks whether the
# fine-tuned model's own response differs from ZERO -- not whether it differs from the
# ORIGINAL model. Two responses can both exclude zero and be indistinguishable from each
# other, so every verdict built on that flag is void. This recomputes the comparison on
# the DIFFERENCE between models.
#
# The exact paired test needs the per-molecule deltas, which scripts/36 computed and then
# discarded. When they are absent this reports a CONSERVATIVE bound (common noise makes
# the models positively correlated, so treating them as independent overstates the
# standard error) and labels a non-significant outcome INCONCLUSIVE rather than null.
#
# Minutes of CPU. Because it loads no model and writes only into ${OUT}, running the
# python directly on a login node would not breach the no-training rule; SLURM is used
# for consistency with the other wrappers.
#
#     sbatch slurm/37_final_consolidation.sh
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

EVAL="${EVAL:-outputs/metrics/relabel_evaluation/relabel_evaluation.json}"
TUNE_DIR="${TUNE_DIR:-outputs/checkpoints/flow_nqed_relabel_finetune/seed_3}"
OUT="${OUT:-outputs/metrics/final_consolidation}"
METRICS="${METRICS:-outputs/metrics}"

mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Final verification and consolidation (READ-ONLY)"
echo "Start:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:      ${SLURM_JOB_ID}  Node: $(hostname)"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Eval:     ${EVAL}"
echo "Tune dir: ${TUNE_DIR}"
echo "Output:   ${OUT}"
echo "=============================================="

[[ -f "${EVAL}" ]] || { echo "ERROR: ${EVAL} not found." >&2; exit 66; }

ARTIFACT_ARGS=()
for pair in "qed_gap_decomposition=${METRICS}/qed_gap_decomposition/qed_gap_decomposition.json" \
            "acp_gate=${METRICS}/acp_gate/acp_gate.json" \
            "guidance_pilot=${METRICS}/decoder_guidance_pilot/guidance_pilot.json" \
            "qed_support_rerank=${METRICS}/qed_support_rerank/qed_support_rerank.json" \
            "finetune_report=${TUNE_DIR}/finetune_report.json"; do
    ARTIFACT_ARGS+=(--artifact "${pair}")
done

srun "${PYTHON}" -u scripts/37_final_consolidation.py \
    --evaluation "${EVAL}" \
    --output-dir "${OUT}" \
    --checkpoint "best=${TUNE_DIR}/best.pt" \
    --checkpoint "last=${TUNE_DIR}/last.pt" \
    "${ARTIFACT_ARGS[@]}" \
    --candidate finetuned \
    --reference final \
    --decoder degree

echo
echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Report:   ${OUT}/final_consolidation.txt"
echo "Table:    ${OUT}/definitive_table.md   (paste straight into the report)"
echo "Data:     ${OUT}/final_consolidation.json"
echo "=============================================="
echo "READ THE CORRECTED COMPARISON, NOT THE OLD VERDICT. A difference that is"
echo "significant under the conservative bound is a real finding. One that is not is"
echo "INCONCLUSIVE: the saved artefacts lack the per-molecule deltas for the exact test,"
echo "and absence of evidence is not evidence of equality."
echo "=============================================="
