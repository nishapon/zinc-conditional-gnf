#!/bin/bash
#SBATCH --job-name=zinc_arm_figures
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

# ---------------------------------------------------------------------------
# Presentation figures and a results summary from ALREADY SAVED generation outputs.
# Reads only; trains nothing, samples nothing, recomputes no experiment.
#
# Produces, under outputs/figures/arm_comparison/:
#   arm_comparison.png/.pdf    validity by decoder, mean QED, degree fallback, diversity
#   arm_b_molecules.png/.svg   a 4x3 grid of valid Arm B molecules with their QED values
#   results_summary.md         the numbers in prose, generated from the data
#   figure_data.json           every plotted value plus provenance
#
# The validity panel keeps the three validities apart on purpose: raw (argmax, no
# chemistry), greedy valence-aware, and degree-constrained with its greedy fallback.
# Only the raw number reflects the model's own bond predictions.
#
#   ARM_A=... ARM_B=...   directories holding each arm's metrics.json (and samples.jsonl)
#   REFERENCE=...         comma-separated earlier diagnostic JSONs for the real and
#                         reconstruction reference values
#   SELECT=spread|best|random   how the 12 molecules are chosen (default: spread, i.e.
#                         evenly spaced across the QED range, so nothing is cherry-picked)
#
#     sbatch slurm/09_arm_comparison_figures.sh
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
    echo "rc: 2=missing input, 4=bad arguments, 1=traceback" >&2
    echo "Check that both arms have been generated from:" >&2
    echo "  sbatch --export=ALL,CHECKPOINT_NAME=last.pt,COUNT=2000,SAMPLES_ROOT=outputs/samples/flow_uncond_pause slurm/05_generate_uncond.sh" >&2
    echo "  sbatch --export=ALL,ARM=b,CHECKPOINT_NAME=last.pt,COUNT=2000,SAMPLES_ROOT=outputs/samples/flow_uncond_ar_pause slurm/05_generate_uncond.sh" >&2
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
[[ -n "${PYTHON}" && -x "${PYTHON}" ]] || { echo "ERROR: zinc_gnf python not found" >&2; exit 65; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg

ARM_A="${ARM_A:-outputs/samples/flow_uncond_pause/seed_1/T0.85}"
ARM_B="${ARM_B:-outputs/samples/flow_uncond_ar_pause/seed_1/T0.85}"
REFERENCE="${REFERENCE:-outputs/metrics/latent_diagnostics/diagnostics.json,outputs/metrics/embedding_populations/embedding_populations.json}"
SELECT="${SELECT:-spread}"
OUT="outputs/figures/arm_comparison"
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Arm comparison figures"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU; reads saved files only)"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Arm A:            ${ARM_A}"
echo "Arm B:            ${ARM_B}"
echo "Output:           ${OUT}"
echo "=============================================="
"${PYTHON}" -V

for DIRECTORY in "${ARM_A}" "${ARM_B}"; do
    if [[ ! -f "${DIRECTORY}/metrics.json" ]]; then
        echo "ERROR: ${DIRECTORY}/metrics.json not found. Generate that arm first." >&2
        exit 66
    fi
done

# Only pass reference files that exist, so a missing diagnostic degrades to "no
# reference lines" instead of failing the job.
REFERENCE_ARGS=()
IFS=',' read -r -a REFERENCE_PATHS <<< "${REFERENCE}"
for PATH_ENTRY in "${REFERENCE_PATHS[@]}"; do
    [[ -z "${PATH_ENTRY}" ]] && continue
    if [[ -f "${PATH_ENTRY}" ]]; then
        REFERENCE_ARGS+=(--reference "${PATH_ENTRY}")
        echo "reference values from: ${PATH_ENTRY}"
    else
        echo "note: ${PATH_ENTRY} not found, omitted from the reference lines"
    fi
done

echo
echo "---- figures ----"
srun "${PYTHON}" -u scripts/09_arm_comparison_figures.py \
    --arm-a "${ARM_A}" \
    --arm-b "${ARM_B}" \
    --output-dir "${OUT}" \
    --select "${SELECT}" \
    --grid-molecules 12 \
    "${REFERENCE_ARGS[@]+${REFERENCE_ARGS[@]}}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo
echo "Figures and the summary are under ${OUT} and are NOT committed to git"
echo "(outputs/figures/* is gitignored). Copy them out for the presentation."
echo "=============================================="
