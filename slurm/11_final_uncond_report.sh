#!/bin/bash
#SBATCH --job-name=zinc_final_uncond
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# ---------------------------------------------------------------------------
# THE FINAL UNCONDITIONAL FIGURE. Read-only, CPU only, no training, no sampling.
#
# Compares Arm B (node-autoregressive) at 10,000 and 50,000 training steps at the SAME
# sampling temperature (T=0.2), from generation output that already exists:
#
#   10k:  outputs/samples/flow_uncond_ar_lowtemp/seed_1/T0.2/
#   50k:  outputs/samples/flow_uncond_ar_50k/seed_1/T0.2/
#
# All chemistry (QED, QED >= 0.7, rings, aromatic rings, aromatic/all rings, validity,
# uniqueness, fallback, internal diversity) is recomputed from each run's samples.jsonl
# with RDKit, so the figure does not depend on trusting metrics.json; a disagreement
# between the two is printed rather than smoothed over. Novelty is taken from
# metrics.json, because it needs the training SMILES.
#
# Raw validity (the model's own bond predictions) and degree-decoded validity (an
# integer program plus a greedy fallback) are reported as SEPARATE quantities, together
# with the fallback rate, and the summary has an explicit section for every metric that
# got WORSE between the two budgets.
#
#   EARLY=... LATE=...        override the two input directories
#   REFERENCE=path.json       real-ZINC reference values from an earlier diagnostics run
#                             (default: the values measured in jobs 332841/332840, which
#                             the report labels as such)
#   OUT=outputs/figures/arm_b_final
#   GRID=12                   molecules in the grid
#   SELECT=spread|best|random how those molecules are chosen (spread = not cherry-picked)
#
#     sbatch slurm/11_final_uncond_report.sh
#
# Never run this file with `bash` directly on the master node.
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
    echo "rc: 2=missing input, 4=bad args or mismatched temperatures, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/11_final_uncond_report.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg

EARLY="${EARLY:-outputs/samples/flow_uncond_ar_lowtemp/seed_1/T0.2}"
LATE="${LATE:-outputs/samples/flow_uncond_ar_50k/seed_1/T0.2}"
OUT="${OUT:-outputs/figures/arm_b_final}"
GRID="${GRID:-12}"
SELECT="${SELECT:-spread}"
REFERENCE="${REFERENCE:-outputs/metrics/embedding_populations/embedding_populations.json}"
REFERENCE_ARGS=()
if [[ -f "${REFERENCE}" ]]; then
    REFERENCE_ARGS=(--reference "${REFERENCE}")
fi
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Final unconditional report: Arm B 10k vs 50k"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU on purpose)"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "10k samples:      ${EARLY}"
echo "50k samples:      ${LATE}"
echo "Real reference:   ${REFERENCE_ARGS[*]:-built-in measured constants (jobs 332841/332840)}"
echo "Output:           ${OUT}   grid ${GRID} molecules, selection ${SELECT}"
echo "=============================================="
"${PYTHON}" -V

for directory in "${EARLY}" "${LATE}"; do
    if [[ ! -f "${directory}/metrics.json" ]]; then
        echo "ERROR: ${directory}/metrics.json not found." >&2
        echo "Generate it first with slurm/05_generate_uncond.sh, or set EARLY/LATE." >&2
        exit 66
    fi
    if [[ ! -f "${directory}/samples.jsonl" ]]; then
        echo "WARNING: ${directory}/samples.jsonl missing; the chemistry cannot be"
        echo "         recomputed for that run and no molecule grid can be drawn." >&2
    fi
done

echo
echo "---- final report ----"
srun "${PYTHON}" -u scripts/11_final_uncond_report.py \
    --early "${EARLY}" \
    --late "${LATE}" \
    --early-label 10k \
    --late-label 50k \
    --output-dir "${OUT}" \
    --grid-molecules "${GRID}" \
    --select "${SELECT}" \
    "${REFERENCE_ARGS[@]+${REFERENCE_ARGS[@]}}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo
echo "Figure:   ${OUT}/arm_b_10k_vs_50k.png (.pdf for slides)"
echo "Molecules: ${OUT}/arm_b_50k_molecules.png (+ .svg, + _caption.txt)"
echo "Summary:  ${OUT}/final_summary.md   (read the 'What got WORSE' section too)"
echo "Data:     ${OUT}/figure_data.json"
echo
echo "Note: 'degree' validity includes greedy-fallback molecules; raw validity is the only"
echo "column that is purely the flow's own bond prediction. Aromaticity is not enforced by"
echo "any decoder, so the aromatic-ring gap is a model result, not a decoding artefact."
echo "=============================================="
