#!/bin/bash
#SBATCH --job-name=zinc_qedpath
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
# STAGE 20: resolve the conditional-signal and decoder-constraint contributions.
# CPU only. READ-ONLY. No flow retraining. No architecture change.
#
# WHY THIS JOB EXISTS
# Stage 19 (job 340255) printed one verdict, H2-direction, from a ladder that tests H2
# before H3 - so the degree-decoder attenuation its OWN tables showed could not reach the
# verdict. Three measurements also need fixing before that verdict can be trusted:
#
#   * the probe's single-prediction held-out MAE (0.0885) was used as the detection
#     threshold for the MEAN of a PAIRED difference over hundreds of pairs. Those are
#     different quantities and the comparison is biased towards "no signal".
#   * the transfer control (94-99% preservation) ran on TRAINING molecules, which the
#     autoencoder was fitted on. It does not establish held-out behaviour.
#   * the N=23 raw figure (+0.1444) rested on 6 valid pairs and must not be read as a
#     general response.
#
# WHAT IT RUNS (one python process, five measurements)
#   A  decoder QED transfer on HELD-OUT real molecules (validation; the test split is
#      never loaded), per-molecule so the preservation ratio carries a bootstrap
#      interval, beside aromaticity and raw/constrained/degree validity. The train split
#      is measured too, only so the generalisation gap is visible.
#   B  two read-outs. The stage-19 MLP probe is carried over UNMODIFIED, and a
#      closed-form within-N ridge read-out is fitted on TRAIN only (alpha chosen on an
#      inner split of train). For each: validation MAE and correlation, stratified by N
#      and QED band, calibration, the within-N GAIN (what a REAL QED gap looks like
#      through that instrument at fixed N) and the NULL (what it reports for real pairs in
#      the SAME band). The null, not the MAE, is the detection threshold.
#   C  the generated embeddings read through both instruments: paired bootstrap intervals,
#      a resolution decision against the real-pair null, and the implied real QED
#      difference after dividing by the measured gain.
#   D  constrained vs degree-constrained decoding of IDENTICAL embeddings on ONE common
#      valid-pair subset, with the attenuation decomposed over fallback / solved / mixed
#      strata. That decomposition is exact: on a fallback the degree decoder returns
#      precisely the constrained decoder's bonds, so the fallback stratum contributes
#      exactly zero. The job also measures whether the predicted DEGREE SEQUENCE responds
#      to the request at all - if it does not, the integer program's constraint is
#      identical under both requests, the bond set is frozen, and only bond orders can
#      carry QED.
#   E  support, every count with its denominator, on both splits.
#
# Then FOUR INDEPENDENT findings - E1 generated direction, E2 decoder transfer,
# E3 degree attenuation, E4 distribution shift - each graded supported / not_supported /
# inconclusive / needs_further_test, and ONE recommendation among A (flow), B (decoder),
# C (representation) with benefits, risks, costs and preconditions. No axis suppresses
# another, and "needs_further_test" is reported as a fact about the INSTRUMENT, never as
# evidence about the model.
#
# The read-outs are diagnostic: they never guide generation, never rerank a sample and are
# never reported as achieved QED.
#
# PRESERVATION
#   Writes ONLY under outputs/metrics/qed_pathway. Stage 17-19 outputs
#   (outputs/metrics/decoder_information, warm_start, signal_*) and every checkpoint are
#   opened read-only and never modified. No stage is re-run.
#
#   FLOW=outputs/checkpoints/warm_start_stageA/last.pt   (the stage-17 warm start, READ-ONLY)
#   OUT=outputs/metrics/qed_pathway
#   TARGET_N=15,23,30   QED_PAIR=0.4,0.8   COUNT=384   TEMPERATURES=0.2,0.6,1.0
#
#     sbatch slurm/20_qed_pathway_audit.sh
#
# Runtime estimate: stage 19 did 12 generated cells in 7 min 20 s. This job has 9 cells
# (3 N x 3 T) at 384 samples instead of 256, decodes both conditions with all three
# decoders, and adds the held-out and train transfer controls (6 cells x 2 x 256 real
# molecules) plus two read-out fits on 20,000 encoded molecules. The degree decoder's
# integer programs dominate at roughly 25-50 ms per molecule. Expect 45-90 minutes; the
# 4 h limit leaves margin. Bootstraps are vectorised numpy over at most a few thousand
# values and cost seconds.
# Memory estimate: peak is the 24,000 encoded pooled feature vectors plus one cell's
# paired embeddings, under ~8 GB. 24 G is requested for headroom on the probe fit.
#
# Never run this file with `bash` directly on the login node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; this encodes 24,000" >&2
    echo "molecules and runs thousands of integer programs - not on the login node." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "rc: 2=missing checkpoint/data, 4=bad args, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT:-outputs/metrics/qed_pathway}/audit.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/20_qed_pathway_audit.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
FLOW="${FLOW:-outputs/checkpoints/warm_start_stageA/last.pt}"
OUT="${OUT:-outputs/metrics/qed_pathway}"
TARGET_N="${TARGET_N:-15,23,30}"
QED_PAIR="${QED_PAIR:-0.4,0.8}"
COUNT="${COUNT:-384}"
TEMPERATURES="${TEMPERATURES:-0.2,0.6,1.0}"
QED_WINDOW="${QED_WINDOW:-0.05}"
PROBE_EPOCHS="${PROBE_EPOCHS:-10}"
REFERENCE_PAIRS="${REFERENCE_PAIRS:-256}"
RESAMPLES="${RESAMPLES:-2000}"
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Stage 20: QED pathway audit (signal vs decoder constraints)"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU on purpose)"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Flow:             ${FLOW} (READ-ONLY)"
echo "Grid:             N=${TARGET_N}  QED ${QED_PAIR}  T=${TEMPERATURES}  x ${COUNT}"
echo "Output:           ${OUT}  (the only directory written)"
echo "Preserved:        stage 17-19 outputs and all checkpoints are read-only"
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
try:
    from scipy.stats import rankdata                      # noqa: F401
except ImportError as error:
    sys.exit(f"ERROR: scipy.stats.rankdata unavailable ({error}); the AUC needs it.")
print("torch", torch.__version__, "| rdkit", rdkit.__version__, "| scipy", scipy.__version__)
PYCHECK

echo
echo "---- audit ----"
srun "${PYTHON}" -u scripts/21_qed_pathway_audit.py \
    --flow-checkpoint "${FLOW}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --target-n "${TARGET_N}" \
    --qed-pair "${QED_PAIR}" \
    --count "${COUNT}" \
    --temperature "${TEMPERATURES}" \
    --qed-window "${QED_WINDOW}" \
    --probe-train-molecules 20000 \
    --probe-val-molecules 4000 \
    --probe-epochs "${PROBE_EPOCHS}" \
    --transfer-molecules 256 \
    --reference-pairs "${REFERENCE_PAIRS}" \
    --bootstrap-resamples "${RESAMPLES}" \
    --max-examples 5 \
    --seed 17 \
    --device cpu

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo
echo "READ IN THIS ORDER (${OUT}/qed_pathway.txt):"
echo "  1. table B, the gain and null columns. They decide what every later number MEANS."
echo "     If NEITHER read-out has a within-N gain whose interval excludes zero, then the"
echo "     embedding axis is UNMEASURED, not cleared: E1 will say needs_further_test and no"
echo "     flow-side or representation-level decision is justified on this run."
echo "  2. table A, validation rows. If held-out preservation is below ~0.5 the decoder"
echo "     cannot transmit QED and the next step is the decoder or the representation."
echo "     Compare with the train rows: a large gap means stage 19's 94-99% was memorised."
echo "  3. table C. 'resolved' plus an implied real difference far below the request is"
echo "     the evidence for a weak QED-predictive direction. 'not_resolved' is a statement"
echo "     about the INSTRUMENT and must not be read as absence of signal."
echo "  4. table D. The gap column is constrained minus degree on ONE common valid subset,"
echo "     so it cannot be a validity artefact. Check fallback_identity_holds first: if it"
echo "     is false the decomposition is invalid. Then read the strata - the fallback"
echo "     stratum contributes exactly zero by construction, so a positive gap comes from"
echo "     the solved stratum (the integer program), the mixed stratum (fallback"
echo "     switching), a frozen bond set, or validity selection. Read it per temperature:"
echo "     stage 19 showed T=1.0 behaving differently from T=0.2."
echo "  5. table E for the exact denominators. N=15 and N=23 have thin low-QED support."
echo
echo "The FINDINGS block grades four axes independently and more than one can be true."
echo "The RECOMMENDATION names one of A / B / C with its preconditions - it is a"
echo "recommendation only: this job changes no architecture and trains no flow."
echo "=============================================="
