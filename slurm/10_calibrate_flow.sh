#!/bin/bash
#SBATCH --job-name=zinc_calibrate
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# LATENT RECALIBRATION of an already-trained flow. CPU only, no training.
#
# Measures, per latent dimension, the mean and std of z = f(x + smoothing noise) on
# TRAINING molecules through the frozen autoencoder recorded in the flow checkpoint,
# and writes a NEW checkpoint whose prior is that measured Gaussian. The input
# checkpoint is opened read-only and is never modified; the output path is refused if
# it already exists (pass FORCE=1 to replace it).
#
# Why: the flow is sampled from its prior, but nothing guarantees the latents it
# produces have the prior's spread. The teammates measured exactly this gap on QM9
# (their inspect_latent_calibration.py): the learned sigma underestimated the latent
# spread by about 1.5x, and widening sigma at sampling time recovered uniqueness at
# N=5. Here the fix is done properly rather than as a sampling hack: the empirical
# per-dimension mean and std ARE the maximum-likelihood parameters of a diagonal
# Gaussian prior, and the trained prior is in that same family, so the measured
# likelihood cannot get worse - and both log_prob and sample use the new prior, so the
# model stays a scoreable density. The job prints the before/after NLL per node on
# training AND validation molecules, so the effect is a number, not a hope.
# A perfectly calibrated flow gives a gain of 0.0: that is a valid outcome, and it
# would say the mismatch is NOT where the problem is.
#
#   FLOW_CHECKPOINT=...   which flow to calibrate (default: arm B's paused 10k last.pt)
#   OUT_CHECKPOINT=...    default: <input stem>_calibrated.pt next to the input
#   MOLECULES=20000       training molecules used for the measurement
#   VAL_MOLECULES=5000    validation molecules for the before/after likelihood check
#   MEASURE_ONLY=1        report only, write no checkpoint
#   FORCE=1               allow replacing an existing output checkpoint
#
#     sbatch slurm/10_calibrate_flow.sh
#     sbatch --export=ALL,FLOW_CHECKPOINT=outputs/checkpoints/flow_uncond/seed_1/last.pt slurm/10_calibrate_flow.sh
#     sbatch --export=ALL,MEASURE_ONLY=1 slurm/10_calibrate_flow.sh
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
    echo "rc: 2=missing checkpoint/data, 3=autoencoder mismatch, 4=bad args or refused" >&2
    echo "    overwrite, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT_DIR:-.}/calibrate.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/10_calibrate_flow.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-outputs/checkpoints/flow_uncond_ar/seed_1/last.pt}"
OUT_CHECKPOINT="${OUT_CHECKPOINT:-}"
OUT_DIR="$(dirname "${FLOW_CHECKPOINT}")"
MOLECULES="${MOLECULES:-20000}"
VAL_MOLECULES="${VAL_MOLECULES:-5000}"
EXTRA=()
if [[ -n "${OUT_CHECKPOINT}" ]]; then EXTRA+=(--output-checkpoint "${OUT_CHECKPOINT}"); fi
if [[ "${MEASURE_ONLY:-0}" == "1" ]]; then EXTRA+=(--measure-only); fi
if [[ "${FORCE:-0}" == "1" ]]; then EXTRA+=(--force); fi
mkdir -p outputs/logs

echo "=============================================="
echo "Latent recalibration of a trained flow"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU on purpose)"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Flow:             ${FLOW_CHECKPOINT} (read-only)"
echo "Output:           ${OUT_CHECKPOINT:-<input stem>_calibrated.pt}"
echo "Measurement:      ${MOLECULES} training molecules; check on ${VAL_MOLECULES} validation"
echo "Measure only:     ${MEASURE_ONLY:-0}    Force overwrite: ${FORCE:-0}"
echo "=============================================="
"${PYTHON}" -V

if [[ ! -f "${DATA_DIR}/splits.pkl" ]]; then
    echo "ERROR: ${DATA_DIR}/splits.pkl not found." >&2
    exit 66
fi
if [[ ! -f "${FLOW_CHECKPOINT}" ]]; then
    echo "ERROR: flow checkpoint ${FLOW_CHECKPOINT} not found." >&2
    echo "Train one first (sbatch slurm/04_flow_uncond_gpu.sh) or set FLOW_CHECKPOINT." >&2
    exit 66
fi

echo
echo "---- calibration ----"
srun "${PYTHON}" -u scripts/10_calibrate_flow.py \
    --flow-checkpoint "${FLOW_CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --molecules "${MOLECULES}" \
    --val-molecules "${VAL_MOLECULES}" \
    --batch-size 128 \
    --device cpu \
    "${EXTRA[@]+${EXTRA[@]}}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Report:  ${OUT_DIR}/calibration.txt"
echo "Data:    ${OUT_DIR}/calibration.json"
echo "Log:     ${OUT_DIR}/calibrate.log"
echo
echo "Next: sample the CALIBRATED checkpoint over a temperature sweep and compare it with"
echo "the uncalibrated samples you already have (the teammates' ZINC run used T=0.2):"
echo "  sbatch --export=ALL,ARM=b,CHECKPOINT_NAME=last_calibrated.pt,COUNT=2000,\\"
echo "TEMPERATURES=0.2,0.5,0.7,0.85,1.0,SAMPLES_ROOT=outputs/samples/flow_uncond_ar_calibrated \\"
echo "    slurm/05_generate_uncond.sh"
echo "A zero gain here is informative: it rules the prior mismatch OUT as the cause."
echo "=============================================="
