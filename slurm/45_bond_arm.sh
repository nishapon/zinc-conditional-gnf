#!/bin/bash
#SBATCH --job-name=zinc_bond_arm
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# EXPERIMENT C: one arm of the bond-aware pairwise coupling conditioner.
# Shared by the three thin wrappers 45a-45c, which set ARM.
#
#   ARM=control              the existing conditioner, continued training
#   ARM=pairwise             the new pathway, LIKELIHOOD TRAINING ALONE
#   ARM=pairwise_supervised  the same pathway plus the bond-prediction loss
#
# All three: same parent checkpoint, same 3,000 steps, same lr 1e-5, same seed, same
# data, same decoder. The two pairwise arms are architecturally identical and differ
# only in the bond loss weight, so the architecture and the training signal can be
# told apart.
#
# RUN MODE=inspect FIRST. It builds the arm's flow, prints the measured parameter and
# activation cost and the bond-target imbalance on one real batch, and exits WITHOUT
# training. That is the memory and runtime estimate; do not queue the GPU arms until
# the reported activation total is comfortable on the node.
#
#   MODE=inspect ARM=pairwise sbatch slurm/45_bond_arm.sh
#   MODE=smoke   ARM=pairwise_supervised sbatch slurm/45_bond_arm.sh
#   ARM=pairwise_supervised sbatch slurm/45_bond_arm.sh
#
# MODE=smoke runs 20 steps into a throwaway directory. It is the first thing to run on
# a GPU: it exercises the warm-start translation, the auxiliary loss, both logged loss
# terms and a checkpoint write, in about a minute. Memory headroom expected at batch
# 128 / N=38 / bond_pair_dim 16: about 355 MB above the control arm. Levers if it does
# not fit: PAIR_DIM=8 (about 213 MB) or BATCH=64 (about 177 MB).
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

ARM="${ARM:?set ARM=control|pairwise|pairwise_supervised}"
DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
START="${START:-outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/best.pt}"
AE="${AE:-outputs/ablation/ae_degree_128d/seed_3/last.pt}"
STEPS="${STEPS:-3000}"
LR="${LR:-1e-5}"
BATCH="${BATCH:-128}"
PAIR_DIM="${PAIR_DIM:-16}"
SEED="${SEED:-3}"
MODE="${MODE:-train}"

# the supervised arm is the only one with a nonzero weight; the script enforces this
# too, so a mistake here is refused rather than silently relabelling an arm
case "${ARM}" in
    pairwise_supervised) WEIGHT="${WEIGHT:-0.1}" ;;
    *)                   WEIGHT="0.0" ;;
esac

OUT="${OUT:-outputs/checkpoints/bond_arms/${ARM}}"
EXTRA=""
case "${MODE}" in
    inspect) EXTRA="--inspect-only" ;;
    smoke)   STEPS=20; OUT="outputs/checkpoints/bond_arms/smoke_${ARM}"
             EXTRA="--eval-every 10 --save-every 10 --val-molecules 256" ;;
    train)   ;;
    *) echo "ERROR: MODE must be inspect, smoke or train (got ${MODE})" >&2; exit 67 ;;
esac
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Experiment C arm: ${ARM}   mode=${MODE}"
echo "Start:  $(date -u +%Y-%m-%dT%H:%M:%SZ)  Job: ${SLURM_JOB_ID}"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Parent: ${START}  (translated into ${OUT}, never written in place)"
echo "Budget: +${STEPS} steps  lr=${LR}  batch=${BATCH}  seed=${SEED}"
echo "Pathway: pair_dim=${PAIR_DIM}  bond_loss_weight=${WEIGHT}"
echo "Output: ${OUT}"
echo "=============================================="
for req in "${DATA_DIR}/splits.pkl" "${START}" "${AE}"; do
    [[ -f "${req}" ]] || { echo "ERROR: ${req} not found." >&2; exit 66; }
done
if [[ "${MODE}" == "smoke" && -e "${OUT}/last.pt" ]]; then
    echo "Removing the previous smoke checkpoint so PATH 1 is exercised, not PATH 2."
    rm -f "${OUT}/last.pt" "${OUT}/best.pt"
fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

srun "${PYTHON}" -u scripts/45_train_bond_arm.py \
    --arm "${ARM}" --start-checkpoint "${START}" --output-dir "${OUT}" \
    --data-dir "${DATA_DIR}" --ae-checkpoint "${AE}" \
    --screening-steps "${STEPS}" --learning-rate "${LR}" \
    --batch-size "${BATCH}" --bond-pair-dim "${PAIR_DIM}" \
    --bond-loss-weight "${WEIGHT}" \
    --seed "${SEED}" ${EXTRA} --device auto

echo
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Plan:     ${OUT}/bond_arm_plan.json   (measured cost + bond imbalance)"
echo "Summary:  ${OUT}/bond_arm_summary.json"
echo
echo "WHAT TO CHECK IN train.log BEFORE TRUSTING A RUN:"
echo "  nll_per_node          the likelihood term, separate from the total loss"
echo "  bond_loss             the auxiliary term, logged in BOTH pairwise arms"
echo "  bond_f1 / bond_recall must be read instead of bond_accuracy; the all-negative"
echo "                        predictor's score is logged as bond_trivial_accuracy"
echo "  bond_bridge_norm      zero-initialised. If it never leaves 0 the new pathway"
echo "                        never engaged and the arm says nothing about the"
echo "                        architecture."
echo
echo "NEXT (generation decides this, not bond accuracy):"
echo "  scripts/42_checkpoint_eval.py with --checkpoint control=... --checkpoint"
echo "  pairwise=... --checkpoint pairwise_supervised=... so draws are paired by noise"
echo "  id, then native QED MAE, valid target-hit yield over ALL attempts, and raw"
echo "  validity by N (15/23/30) with corrected validity, ring distributions against"
echo "  the matched reference, uniqueness and diversity alongside. Use FRESH draws for"
echo "  confirmation and keep best-of-10 reporting separate from native."
