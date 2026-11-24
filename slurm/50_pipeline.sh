#!/bin/bash
#SBATCH --job-name=zinc_qed_pipeline
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# ---------------------------------------------------------------------------
# One wrapper for the whole QED comparison pipeline. Choose a step with STEP=.
#
#   STEP=preflight    sbatch slurm/50_pipeline.sh    # CPU only, no GPU needed
#   STEP=smoke        sbatch slurm/50_pipeline.sh    # ~minutes, verifies the pipeline
#   STEP=conditional  sbatch slurm/50_pipeline.sh    # 2000 attempts x 5 reps per target
#   STEP=percentile   sbatch slurm/50_pipeline.sh    # training-percentile target set
#   STEP=compare      sbatch slurm/50_pipeline.sh    # 10,000 x 5 at t=0.85
#   STEP=summarize    sbatch slurm/50_pipeline.sh    # CPU only, CSVs + plots
#
# Every step is RESUMABLE: completed cells are skipped, so a timed-out job can be
# resubmitted unchanged. Add FORCE=1 to recompute.
#
# SCHEDULER NOTE. The SBATCH header follows this repository's existing convention
# (partition STUD, --gres=gpu:1), copied from slurm/44_train_arm.sh. If your cluster
# differs, every step below is a plain `$PYTHON scripts/NN_*.py ...` call and can be run
# under any scheduler, or directly on an interactive GPU node. Send me one working job
# script for your queue and I will finish matching wrappers.
# ---------------------------------------------------------------------------

set -euo pipefail
STEP="${STEP:?set STEP=preflight|smoke|conditional|percentile|compare|summarize}"
WORKDIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${WORKDIR}"

PYTHON="${ZINC_GNF_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    for candidate in "$HOME/miniconda3/envs/zinc_gnf/bin/python" \
                     "$HOME/anaconda3/envs/zinc_gnf/bin/python" \
                     "$HOME/miniforge3/envs/zinc_gnf/bin/python"; do
        if [[ -x "${candidate}" ]]; then PYTHON="${candidate}"; break; fi
    done
fi
[[ -n "${PYTHON}" && -x "${PYTHON}" ]] || { echo "ERROR: set ZINC_GNF_PYTHON=/path/to/python" >&2; exit 65; }

export PYTHONUNBUFFERED=1 PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg

CONFIG="${CONFIG:-configs/cluster.json}"
FORCE_FLAG=""; [[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"
RUNNER=""; command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]] && RUNNER="srun"
mkdir -p outputs/logs

echo "=============================================="
echo "step=${STEP}  config=${CONFIG}"
echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)  job=${SLURM_JOB_ID:-none}"
echo "commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "python=${PYTHON}"
echo "=============================================="

case "${STEP}" in
  preflight)
    ${RUNNER} "${PYTHON}" -u scripts/47_cluster_preflight.py --config "${CONFIG}"
    ;;
  smoke)
    # small budget: proves loading, generation, decoding, metrics and saved outputs
    ${RUNNER} "${PYTHON}" -u scripts/48_conditional_qed_eval.py \
        --config "${CONFIG}" --target-set grid --preset smoke ${FORCE_FLAG}
    ${RUNNER} "${PYTHON}" -u scripts/49_summarize.py \
        --config "${CONFIG}" --target-set grid
    ;;
  conditional)
    ${RUNNER} "${PYTHON}" -u scripts/48_conditional_qed_eval.py \
        --config "${CONFIG}" --target-set grid --preset full ${FORCE_FLAG}
    ;;
  percentile)
    ${RUNNER} "${PYTHON}" -u scripts/48_conditional_qed_eval.py \
        --config "${CONFIG}" --target-set percentile --preset full ${FORCE_FLAG}
    ;;
  compare)
    ${RUNNER} "${PYTHON}" -u scripts/46_uncond_vs_cond.py \
        --config "${CONFIG}" --moflow-protocol \
        --output-dir "$("${PYTHON}" - "${CONFIG}" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1]))["paths"]["results_dir"] + "/uncond_vs_cond")
EOF
)"
    ;;
  summarize)
    ${RUNNER} "${PYTHON}" -u scripts/49_summarize.py \
        --config "${CONFIG}" --target-set grid
    ${RUNNER} "${PYTHON}" -u scripts/49_summarize.py \
        --config "${CONFIG}" --target-set percentile || true
    ;;
  *)
    echo "ERROR: unknown STEP=${STEP}" >&2; exit 64 ;;
esac

echo
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "results under: $("${PYTHON}" - "${CONFIG}" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1]))["paths"]["results_dir"])
EOF
)"
