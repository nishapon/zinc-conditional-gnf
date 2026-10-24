#!/bin/bash
#SBATCH --job-name=zinc_decinfo
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# STAGE 19: where is the requested QED lost? CPU only, read-only, no flow retraining.
#
# Stage 18 ruled out a weak conditional signal: generated embeddings move 0.665x (median,
# 1.217x at N=23/T=0.2) of the real populations' displacement, cosine +0.926, all samples
# finite. A matching MEAN displacement is NOT proof that the decoder is to blame, so this
# job measures each stage separately and names the hypothesis the data support:
#
#   H1  the information is in the generated embeddings and DECODING loses it
#   H2  the embeddings move, but not along a QED-PREDICTIVE direction
#   H3  the information reaches the decoder and graph constraints / validity selection
#       erase it afterwards
#
# What it runs:
#   A  decoder QED-TRANSFER CONTROL - encode REAL training molecules at QED 0.4 and 0.8
#      (same N) and decode them through the path generation uses. If the decoder cannot
#      preserve a difference it is GIVEN, no flow-side work can fix QED control. This is
#      the measurement that makes everything else interpretable.
#   B  paired intervention - identical latent noise, only the QED request changes;
#      displacement reported at the latent, embedding, bond-logit, bond-argmax, graph and
#      final-QED stages, paired with standard errors, separately for raw / constrained /
#      degree decoding, plus validity-survival rates so selection effects are visible.
#   C  a QED read-out trained on REAL TRAIN embeddings, validated on REAL VALIDATION
#      embeddings, applied diagnostically to generated embeddings. Its held-out MAE is
#      printed as its resolution limit and Mahalanobis statistics quantify how far the
#      generated embeddings lie outside its fitting distribution. It never guides
#      generation and is never reported as achieved QED.
#   E  support with every count and denominator stated (the stage-18 'support' column was
#      the MINIMUM of the two QED groups at that N; both are printed here).
#
# Nothing is written outside outputs/metrics/decoder_information; every checkpoint is
# opened read-only and the test split is never loaded.
#
#   FLOW=outputs/checkpoints/warm_start_stageA/last.pt   (the stage-17 warm start)
#   OUT=outputs/metrics/decoder_information
#   TARGET_N=15,23,30   QED_PAIR=0.4,0.8   COUNT=256   TEMPERATURES=0.2,1.0
#   PROBE_EPOCHS=10
#
#     sbatch slurm/19_decoder_information.sh
#
# Runtime estimate: the probe fits on 20,000 encoded training molecules (a few minutes on
# CPU), the transfer control decodes 3 x 2 x 256 real molecules, and the intervention
# decodes 2 x 3 x 2 x 256 generated ones with all three decoders. The degree decoder's
# integer program dominates: roughly 25-50 ms per molecule, so about 60-90 minutes in
# total. The 4 h limit leaves margin; memory stays under ~8 GB.
#
# Never run this file with `bash` directly on the login node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; this encodes 20,000" >&2
    echo "molecules and runs thousands of integer programs - not on the login node." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "rc: 2=missing checkpoint/data, 4=bad args, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT:-outputs/metrics/decoder_information}/audit.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/19_decoder_information.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
FLOW="${FLOW:-outputs/checkpoints/warm_start_stageA/last.pt}"
OUT="${OUT:-outputs/metrics/decoder_information}"
TARGET_N="${TARGET_N:-15,23,30}"
QED_PAIR="${QED_PAIR:-0.4,0.8}"
COUNT="${COUNT:-256}"
TEMPERATURES="${TEMPERATURES:-0.2,1.0}"
PROBE_EPOCHS="${PROBE_EPOCHS:-10}"
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Stage 19: decoder-information audit"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU on purpose)"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Flow:             ${FLOW} (READ-ONLY)"
echo "Grid:             N=${TARGET_N}  QED ${QED_PAIR}  T=${TEMPERATURES}  x ${COUNT}"
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
    sys.exit(f"ERROR: scipy.optimize.milp unavailable ({error}); the degree decoder needs it.")
print("torch", torch.__version__, "| rdkit", rdkit.__version__, "| scipy", scipy.__version__)
PYCHECK

echo
echo "---- audit ----"
srun "${PYTHON}" -u scripts/20_decoder_information_audit.py \
    --flow-checkpoint "${FLOW}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --target-n "${TARGET_N}" \
    --qed-pair "${QED_PAIR}" \
    --count "${COUNT}" \
    --temperature "${TEMPERATURES}" \
    --probe-train-molecules 20000 \
    --probe-val-molecules 4000 \
    --probe-epochs "${PROBE_EPOCHS}" \
    --transfer-molecules 256 \
    --seed 17 \
    --device cpu

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo
echo "READ IN THIS ORDER:"
echo "  1. table A (decoder QED transfer). This decides the rest. If the decoder preserves"
echo "     less than about half of a QED difference it is GIVEN on real embeddings, then QED"
echo "     control is impossible through this decoder and the next step is the decoder or"
echo "     the representation - not the flow."
echo "  2. the probe column in table B against the probe's own held-out MAE. A change the"
echo "     probe cannot resolve is NOT evidence that the embeddings carry QED; it is"
echo "     evidence that the displacement is not QED-predictive (hypothesis H2)."
echo "  3. raw paired QED versus degree paired QED. If raw moves and degree does not, graph"
echo "     constraints and validity selection are erasing the effect (hypothesis H3)."
echo "  4. the OOD column: generated embeddings far outside the real distribution mean the"
echo "     decoder and the probe are both extrapolating, which bounds what either can say."
echo "  5. table E for the exact support denominators."
echo
echo "The verdict line in ${OUT}/decoder_information.txt names the supported hypothesis,"
echo "but the tables are the evidence: read them before acting."
echo "=============================================="
