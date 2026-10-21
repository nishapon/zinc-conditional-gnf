#!/bin/bash
#SBATCH --job-name=zinc_num_qed
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

# ---------------------------------------------------------------------------
# STAGE 15: (a) invertibility audit of the existing 50k joint checkpoint, then
#           (b) native vs reranked vs latent-guided QED at equal decoded budget.
# CPU only, no training, no new 50k run. Every checkpoint is opened READ-ONLY and
# nothing under outputs/checkpoints is written.
#
# (a) explains the stage-14 asymmetry. z->x->z computes F(G(z)) with the same
#     arithmetic in both directions, so the inverse's error is largely cancelled by the
#     forward pass: 1.41e-05 is a self-CONSISTENCY number, not evidence that inversion
#     is accurate. x->z->x computes G(F(x)), where nothing cancels - that is the 4.92e-03.
#     The audit reports both, in float32 AND float64, per component (ActNorm, +invertible
#     linear, +affine couplings, +node-autoregressive), by N, across the requested
#     (N, QED) grid, and finally decodes the round-tripped embeddings to ask whether the
#     error changes the MOLECULE - the only question that decides whether the existing
#     50k results stand.
#
# (b) runs the three arms with the SAME number of decoded candidates. The latent->QED
#     predictor is fitted on TRAINING molecules only and its held-out accuracy gates the
#     guided arm: above GATE it is skipped rather than reported. Reranking is labelled as
#     RDKit post-selection everywhere it appears and is never presented as conditioning.
#     If (a) says float32 is not trustworthy, set DOUBLE=1 and the inverse runs in
#     float64 - no retraining, no weight change.
#
#   FLOW=outputs/checkpoints/flow_nqed/seed_1/best.pt
#   OUT=outputs/metrics/stage15
#   TARGET_N=15,23,30     TARGET_QED=0.4,0.6,0.8
#   COUNT=200             kept molecules per cell
#   GROUP=5               candidates per kept molecule (the equal budget is COUNT*GROUP)
#   TEMPERATURE=0.2       matches the 50k evaluation
#   STEPS=10              guidance steps
#   GATE=0.08             held-out predictor MAE above which guidance is skipped
#   DOUBLE=1              run the flow inverse in float64
#   AUDIT_ONLY=1          stop after (a)
#
#     sbatch slurm/15_numerics_and_guided_qed.sh
#
# Never run this file with `bash` directly on the login node.
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
    echo "rc: 2=missing checkpoint/data, 3=autoencoder mismatch, 4=bad args, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT:-outputs/metrics/stage15}/*/*.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/15_numerics_and_guided_qed.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
FLOW="${FLOW:-outputs/checkpoints/flow_nqed/seed_1/best.pt}"
OUT="${OUT:-outputs/metrics/stage15}"
TARGET_N="${TARGET_N:-15,23,30}"
TARGET_QED="${TARGET_QED:-0.4,0.6,0.8}"
COUNT="${COUNT:-200}"
GROUP="${GROUP:-5}"
TEMPERATURE="${TEMPERATURE:-0.2}"
STEPS="${STEPS:-10}"
GATE="${GATE:-0.08}"
GUIDED_EXTRA=()
if [[ "${DOUBLE:-0}" == "1" ]]; then GUIDED_EXTRA+=(--double-inverse); fi
mkdir -p outputs/logs "${OUT}/invertibility" "${OUT}/guided_qed"

echo "=============================================="
echo "Stage 15: invertibility audit + guided QED experiment"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU on purpose)"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Flow:             ${FLOW} (READ-ONLY, never rewritten)"
echo "Grid:             N=${TARGET_N}  QED=${TARGET_QED}  T=${TEMPERATURE}"
echo "Budget:           ${COUNT} kept x ${GROUP} candidates = $((COUNT * GROUP)) decoded per arm"
echo "Guidance:         ${STEPS} steps, held-out MAE gate ${GATE}, float64 inverse ${DOUBLE:-0}"
echo "Output:           ${OUT}"
echo "=============================================="
"${PYTHON}" -V

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${FLOW}"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: ${required} not found." >&2
        exit 66
    fi
done

"${PYTHON}" - <<'PYCHECK'
import sys
import torch, rdkit, numpy, scipy
try:
    from scipy.optimize import milp
except ImportError as error:
    sys.exit(f"ERROR: scipy.optimize.milp unavailable ({error}); install scipy >= 1.9.")
print("torch", torch.__version__, "| rdkit", rdkit.__version__, "| scipy", scipy.__version__)
PYCHECK

echo
echo "---- 1/2 invertibility audit ----"
srun "${PYTHON}" -u scripts/15_invertibility_audit.py \
    --flow-checkpoint "${FLOW}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}/invertibility" \
    --molecules 512 \
    --batch-size 64 \
    --target-n "${TARGET_N}" \
    --target-qed "${TARGET_QED}" \
    --samples-per-cell 64 \
    --device cpu

if [[ "${AUDIT_ONLY:-0}" == "1" ]]; then
    echo "AUDIT_ONLY=1: stopping after the audit."
    echo "Report: ${OUT}/invertibility/invertibility.txt"
    exit 0
fi

echo
echo "---- 2/2 native vs reranked vs latent-guided QED (equal decoded budget) ----"
srun "${PYTHON}" -u scripts/16_qed_guided_experiment.py \
    --flow-checkpoint "${FLOW}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}/guided_qed" \
    --target-n "${TARGET_N}" \
    --target-qed "${TARGET_QED}" \
    --count "${COUNT}" \
    --rerank-group "${GROUP}" \
    --temperature "${TEMPERATURE}" \
    --guidance-steps "${STEPS}" \
    --predictor-molecules 20000 \
    --predictor-mae-gate "${GATE}" \
    --batch-size 128 \
    --device cpu \
    "${GUIDED_EXTRA[@]+${GUIDED_EXTRA[@]}}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "READ IN THIS ORDER:"
echo "  1. ${OUT}/invertibility/invertibility.txt"
echo "     - the float32 vs float64 x->z->x lines: is precision the whole story?"
echo "     - the per-component lines: which piece loses it?"
echo "     - the error-vs-N correlation: positive = the SEQUENTIAL node-autoregressive"
echo "       inverse accumulating error along the node order."
echo "     - the chemistry lines: if identical SMILES is ~100%, the error is below the"
echo "       decoder's resolution and the existing 50k results are unaffected."
echo "  2. ${OUT}/guided_qed/guided_qed.txt"
echo "     - compare ONLY native rows with each other to judge the GNF."
echo "     - rerank rows are RDKit post-selection; guided rows use an external latent"
echo "       predictor. Neither is native conditioning, and the table says so."
echo "     - if the predictor's held-out MAE exceeded the gate, the guided arm is absent"
echo "       by design rather than reported as a weak result."
echo "=============================================="
