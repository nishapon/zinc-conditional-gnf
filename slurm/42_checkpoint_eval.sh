#!/bin/bash
#SBATCH --job-name=zinc_ckpt_eval
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

# ---------------------------------------------------------------------------
# CHECKPOINT SELECTION ON DECODED QED + CHEMISTRY, WITH MOLECULE GRIDS.
# READ-ONLY: no training, no checkpoint written, outputs to a NEW directory.
#
# STEP 1 IS FREE AND GATES EVERYTHING. Run with MODE=inventory first. It lists every
# checkpoint with its step and validation metrics and estimates the evaluation cost.
# If it reports TOO FEW CANDIDATES, STOP. Do not run extra training to manufacture
# candidates without a separate decision.
#
#     MODE=inventory sbatch slurm/42_checkpoint_eval.sh
#
# STEP 2, only after reading the inventory: pass the checkpoints explicitly.
#
#     CKPTS="best=<dir>/best.pt last=<dir>/last.pt" sbatch slurm/42_checkpoint_eval.sh
#
# Candidates are ranked on the VALIDATION conditions. Re-run the winner with
# SPLIT=test for the final number; the test split has never been read by a flow script.
# Molecule grids are rendered from the saved sample records, so they can be redrawn
# later at no cost.
# ---------------------------------------------------------------------------

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch." >&2; exit 64
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
[[ -n "${PYTHON}" && -x "${PYTHON}" ]] || { echo "ERROR: no zinc_gnf python; set ZINC_GNF_PYTHON=" >&2; exit 65; }
export PYTHONUNBUFFERED=1 PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
CKPT_DIR="${CKPT_DIR:-outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3}"
AE="${AE:-outputs/ablation/ae_degree_128d/seed_3/last.pt}"
OUT="${OUT:-outputs/metrics/checkpoint_eval}"
SAMPLES="${SAMPLES:-50}"
SPLIT="${SPLIT:-validation}"
MODE="${MODE:-inventory}"
GRID_SAMPLES="${GRID_SAMPLES:-6}"
BEST_OF_K="${BEST_OF_K:-10}"
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Checkpoint eval  mode=${MODE}  split=${SPLIT}"
echo "Start:  $(date -u +%Y-%m-%dT%H:%M:%SZ)   Job: ${SLURM_JOB_ID}"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Ckpts:  ${CKPT_DIR}"
echo "Output: ${OUT}"
echo "=============================================="
[[ -d "${CKPT_DIR}" ]] || { echo "ERROR: ${CKPT_DIR} not found." >&2; exit 66; }

if [[ "${MODE}" == "inventory" ]]; then
    srun "${PYTHON}" -u scripts/42_checkpoint_eval.py \
        --checkpoint-dir "${CKPT_DIR}" --output-dir "${OUT}" \
        --samples "${SAMPLES}" --inventory-only
    echo
    echo "READ THE VERDICT. If it says TOO FEW CANDIDATES, stop here and tell me."
    exit 0
fi

[[ -n "${CKPTS:-}" ]] || { echo "ERROR: set CKPTS=\"label=path label=path\"" >&2; exit 67; }
ARGS=(); for pair in ${CKPTS}; do ARGS+=(--checkpoint "${pair}"); done
for req in "${DATA_DIR}/scaler.json" "${AE}"; do
    [[ -f "${req}" ]] || { echo "ERROR: ${req} not found." >&2; exit 66; }
done

srun "${PYTHON}" -u scripts/42_checkpoint_eval.py \
    --checkpoint-dir "${CKPT_DIR}" "${ARGS[@]}" \
    --ae-checkpoint "${AE}" --data-dir "${DATA_DIR}" --output-dir "${OUT}" \
    --target-n 15,23,30 --qed-values 0.4,0.6,0.8 \
    --samples "${SAMPLES}" --split "${SPLIT}" \
    --grid-samples "${GRID_SAMPLES}" --best-of-k "${BEST_OF_K}" \
    --decoder degree --device cpu

echo
echo "Report: ${OUT}/checkpoint_eval.txt"
echo "Grids:  ${OUT}/figures/native_*.png   (baseline vs candidate, shared noise ids)"
echo "        ${OUT}/figures/bestof*_*.png  (SELECTION, separate figures)"
echo "Records:${OUT}/samples_*.jsonl        (noise ids kept for re-pairing)"
