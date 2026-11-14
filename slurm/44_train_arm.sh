#!/bin/bash
#SBATCH --job-name=zinc_arm
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00

# ---------------------------------------------------------------------------
# ONE EXPERIMENT ARM. Shared by the four thin wrappers 44a-44d, which set ARM.
#
# Every arm: same starting checkpoint, same 3,000 screening steps, same lr 1e-5, same
# seed, same objective. The ONLY difference is the named intervention.
#
# The shared warm start is COPIED into the arm's own directory and resumed from there,
# so the original checkpoint is never opened for writing.
#
#   ARM=control        sbatch slurm/44_train_arm.sh      (or use 44a_..44d_)
#   ARM=balanced       ...
#   ARM=prior_frozen   ...
#   ARM=prior_learned  ...
#
# Run MODE=inspect first: it measures the joint N-QED imbalance and exits without
# training. If it reports NEARLY BALANCED ALREADY, do not run the balanced arm.
# ---------------------------------------------------------------------------

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: training must not run on the login node. Submit with sbatch." >&2; exit 64
fi
WORKDIR="${SLURM_SUBMIT_DIR:-$HOME/zinc-conditional-gnf}"; cd "${WORKDIR}"
PYTHON="${ZINC_GNF_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    for c in "$HOME/miniconda3/envs/zinc_gnf/bin/python" \
             "$HOME/anaconda3/envs/zinc_gnf/bin/python" \
             "$HOME/miniforge3/envs/zinc_gnf/bin/python"; do
        if [[ -x "${c}" ]]; then PYTHON="${c}"; break; fi
    done
fi
[[ -n "${PYTHON}" && -x "${PYTHON}" ]] || { echo "ERROR: set ZINC_GNF_PYTHON=" >&2; exit 65; }
export PYTHONUNBUFFERED=1 PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg

ARM="${ARM:?set ARM=control|balanced|prior_frozen|prior_learned}"
DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
START="${START:-outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/best.pt}"
AE="${AE:-outputs/ablation/ae_degree_128d/seed_3/last.pt}"
OUT="${OUT:-outputs/checkpoints/arms/${ARM}}"
STEPS="${STEPS:-3000}"
LR="${LR:-1e-5}"
ALPHA="${ALPHA:-0.5}"
RHO="${RHO:-0.01}"
SEED="${SEED:-3}"
MODE="${MODE:-train}"
EXTRA=""; [[ "${MODE}" == "inspect" ]] && EXTRA="--inspect-only"
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Arm: ${ARM}   mode=${MODE}"
echo "Start:  $(date -u +%Y-%m-%dT%H:%M:%SZ)  Job: ${SLURM_JOB_ID}"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Warm start: ${START}  (copied in, never written)"
echo "Budget: +${STEPS} steps  lr=${LR}  seed=${SEED}"
echo "Output: ${OUT}"
echo "=============================================="
for req in "${DATA_DIR}/splits.pkl" "${START}" "${AE}"; do
    [[ -f "${req}" ]] || { echo "ERROR: ${req} not found." >&2; exit 66; }
done

srun "${PYTHON}" -u scripts/44_train_arm.py \
    --arm "${ARM}" --start-checkpoint "${START}" --output-dir "${OUT}" \
    --data-dir "${DATA_DIR}" --ae-checkpoint "${AE}" \
    --screening-steps "${STEPS}" --learning-rate "${LR}" \
    --balanced-alpha "${ALPHA}" --coupled-init-rho "${RHO}" \
    --seed "${SEED}" ${EXTRA} --device auto

echo
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Summary:  ${OUT}/arm_summary.json"
echo "NEXT: evaluate all arms with scripts/42_checkpoint_eval.py under shared noise,"
echo "      passing --checkpoint control=... --checkpoint ${ARM}=... so the comparison"
echo "      is paired by noise id. Keep best-of-10 reporting separate."
