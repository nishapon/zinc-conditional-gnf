#!/bin/bash
#SBATCH --job-name=zinc_decexp
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00

# ---------------------------------------------------------------------------
# STAGE 21: matched four-decoder comparison. CPU only. READ-ONLY. No flow retraining.
#
# WHAT STAGE 20 ESTABLISHED (job 340259, commit 44ebe76)
#   At N=23, T=0.2, requested QED 0.4 -> 0.8:
#     constrained decoder      +0.0557
#     degree decoder           -0.0041
#     attenuation              +0.0598   CI [+0.0348, +0.0844]
#       integer programming    +0.0212
#       solver / fallback      +0.0386   <- the LARGER half
#   So the dominant cause is the DISCONTINUITY around the integer program, not the
#   program. `degree_constrained_bond_decode` abandons its program and runs a different
#   algorithm (the greedy valence-aware decoder) on five exits, two of which fire for
#   reasons that have nothing to do with chemistry:
#
#     PARITY. The requested degrees are degree_logits.argmax(-1), taken INDEPENDENTLY per
#     atom. Nothing makes their sum even, no graph can realise an odd degree sum, so the
#     decoder tests `wanted.sum() % 2 != 0` and gives up. Moving ONE atom's degree by +-1
#     flips that parity with certainty, so a QED request that shifts a single degree
#     switches the whole assembly algorithm - and the measured response between two
#     conditions is then dominated by which algorithm ran.
#
#     ISOLATED ATOMS. A predicted degree of 0 is representable and guarantees a
#     disconnected graph, so the program is solved and then thrown away by `not_connected`.
#
# THE MECHANISM UNDER TEST (zinc_gnf/assembly_total.py, decoder name "total")
#   One mechanism: make the degree-constrained path TOTAL, so no input can make it switch
#   algorithms. Four steps, none of which can fail and none of which is another algorithm:
#     1. repair the degree sequence to the nearest one a CONNECTED molecule can realise -
#        1 <= d_i <= min(n-1, cap_i), sum(d) >= 2(n-1), even sum, Erdos-Gallai - at the
#        minimum cost under the degree head's own softmax. A parity problem costs one atom
#        one step instead of the whole sequence.
#     2. ONE integer program for the bond SET under those exact degrees. Feasible by the
#        Erdos-Gallai theorem once step 1 holds, so the totality is proved, not hoped for.
#     3. connectivity by degree-preserving 2-swaps, no solver: remove (a,b) and (c,d) in
#        different components, add (a,c) and (b,d). Every degree is preserved. At least
#        one removed edge must be a NON-BRIDGE, and step 1 guarantees >= n-1 edges, so a
#        merging swap always exists.
#     4. ONE small integer program for the bond ORDERS under valence - which replaces the
#        existing decoder's argmax-then-DEMOTE, the step that silently destroys the order
#        information where aromaticity, and much of QED, lives.
#
#   No fallback, no algorithm switch, no demotion, no failure path.
#
# WHAT IS *NOT* DONE, because it would make any improvement meaningless:
#   * NO SELECTION. One embedding yields exactly one molecule per decoder,
#     deterministically. Nothing generates a batch and keeps the best QED.
#   * NO ORACLE. The target QED conditions the FLOW only. The assemblers see the decoder's
#     own bond and degree distributions and never a property value.
#   * NO INVALID CHEMISTRY. Valence and connectivity are enforced; an assembly that cannot
#     be completed is reported FAILED and counts as invalid.
#   * NO TEST DATA. Novelty is measured against TRAIN only; the test split is never loaded.
#
# WHAT IT MEASURES
#   All four assemblers decode THE SAME embeddings - one decode call per cell - so they
#   differ only in how they turn one fixed set of logits into a graph.
#     grid        N={15,23,30} x QED={0.4,0.6,0.8} x T={0.2,0.6,1.0}, one shared seed
#     per cell    raw native validity SEPARATELY, strict validity, exact heavy-atom count,
#                 QED MAE / hit rate / bias, aromatic and total rings, connectivity and
#                 fragments, fallback frequency, relaxation level, and degree/valence
#                 consistency measured FROM THE GRAPH for every decoder alike
#     paired      at fixed noise, QED 0.4 -> 0.8, on a COMMON subset valid under both
#                 requests for EVERY decoder, with bootstrap intervals and the
#                 total-minus-degree improvement
#     population  uniqueness, novelty vs TRAIN, internal diversity
#     cost        seconds per molecule per decoder on identical embeddings
#     support     every cell marked well-supported or low-support from TRAIN counts
#   Then the step 4 criteria as pass / partial / fail / inconclusive, and one step 5
#   recommendation among A (flow conditioning), B (direct atom/bond flow), C (more decoder).
#
# PRESERVATION
#   Writes ONLY under outputs/metrics/decoder_experiment. Stage 1-20 outputs
#   (decoder_information, qed_pathway, warm_start, signal_*) and every checkpoint are
#   opened read-only and never modified. The raw, constrained and degree decoders are
#   unchanged; "total" is an additional option kept out of the default decoder set.
#
#   FLOW=outputs/checkpoints/warm_start_stageA/last.pt   (the stage-17 warm start, READ-ONLY)
#   OUT=outputs/metrics/decoder_experiment
#   TARGET_N=15,23,30  TARGET_QED=0.4,0.6,0.8  TEMPERATURES=0.2,0.6,1.0  COUNT=256
#
#     sbatch slurm/21_decoder_experiment.sh
#
# Runtime estimate: 27 grid cells x 256 samples = 6,912 embeddings, each decoded by four
# assemblers, plus 9 paired cells x 2 conditions x 256 = 4,608 more. Measured cost of the
# experimental assembler is 12-24 ms per molecule (two small HiGHS solves plus the swap
# repair) against 1-10 ms for the degree decoder, and RDKit sanitisation dominates the
# rest. Expect 2.5-5 hours; the 8 h limit leaves margin. Set COUNT=128 to halve it.
# Memory estimate: one cell's records with their bond matrices plus the accumulated
# population SMILES, under ~6 GB. 24 G is requested for headroom.
#
# Never run this file with `bash` directly on the login node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; this decodes over 11,000" >&2
    echo "molecules four ways and runs thousands of integer programs." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "rc: 2=missing checkpoint/data, 4=bad args, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT:-outputs/metrics/decoder_experiment}/experiment.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/21_decoder_experiment.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
FLOW="${FLOW:-outputs/checkpoints/warm_start_stageA/last.pt}"
OUT="${OUT:-outputs/metrics/decoder_experiment}"
TARGET_N="${TARGET_N:-15,23,30}"
TARGET_QED="${TARGET_QED:-0.4,0.6,0.8}"
TEMPERATURES="${TEMPERATURES:-0.2,0.6,1.0}"
COUNT="${COUNT:-256}"
PAIRED_COUNT="${PAIRED_COUNT:-256}"
PAIRED_QED="${PAIRED_QED:-0.4,0.8}"
QED_TOLERANCE="${QED_TOLERANCE:-0.1}"
QED_WINDOW="${QED_WINDOW:-0.05}"
RESAMPLES="${RESAMPLES:-2000}"
# stage 20's held-out real-vs-generated AUC, used only by the step 5 decision
SHIFT_AUC="${SHIFT_AUC:-0.979}"
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Stage 21: matched four-decoder comparison"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU on purpose)"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Flow:             ${FLOW} (READ-ONLY, the stage-17 warm start)"
echo "Grid:             N=${TARGET_N} x QED=${TARGET_QED} x T=${TEMPERATURES} x ${COUNT}"
echo "Decoders:         raw, constrained, degree (all unchanged) + total (experimental)"
echo "Output:           ${OUT}  (the only directory written)"
echo "Preserved:        stage 1-20 outputs and all checkpoints are read-only"
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
    sys.exit(f"ERROR: scipy.optimize.milp unavailable ({error}); both assemblers need it.")
from scipy.sparse import coo_matrix                       # noqa: F401
print("torch", torch.__version__, "| rdkit", rdkit.__version__, "| scipy", scipy.__version__)
PYCHECK

echo
echo "---- experiment ----"
srun "${PYTHON}" -u scripts/22_decoder_experiment.py \
    --flow-checkpoint "${FLOW}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --target-n "${TARGET_N}" \
    --target-qed "${TARGET_QED}" \
    --temperature "${TEMPERATURES}" \
    --count "${COUNT}" \
    --paired-count "${PAIRED_COUNT}" \
    --paired-qed "${PAIRED_QED}" \
    --qed-tolerance "${QED_TOLERANCE}" \
    --qed-window "${QED_WINDOW}" \
    --timing-samples 32 \
    --bootstrap-resamples "${RESAMPLES}" \
    --shift-auc "${SHIFT_AUC}" \
    --seed 17 \
    --device cpu

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo
echo "READ IN THIS ORDER (${OUT}/decoder_experiment.txt):"
echo "  1. table C, the PAIRED table. It is the only measurement that isolates the"
echo "     decoder: same embeddings, same noise, one common subset valid under both QED"
echo "     requests for EVERY decoder. A positive total-degree gap whose interval excludes"
echo "     zero is QED response the degree decoder was losing. Compare it with the"
echo "     'constrained' column, which is the response a decoder with NO degree"
echo "     constraint already delivered - that is the practical ceiling for this mechanism."
echo "  2. table D, assembly health. 'fallback_rate' on the degree decoder is the"
echo "     discontinuity itself; 'requested_sum_was_odd_rate' is how often the predicted"
echo "     degree sum could not be realised by ANY graph. If those are near zero, the"
echo "     mechanism this stage fixes was not firing on real embeddings and a small"
echo "     improvement in table C is the expected result, not a disappointment."
echo "  3. table A, population totals. 'total' must not buy QED response with validity,"
echo "     aromaticity, uniqueness or diversity. Raw native validity is its OWN row."
echo "  4. table B, QED achievement, with the well-supported and low-support rows read"
echo "     separately: an improvement only visible where the training data is thin is not"
echo "     an improvement. QED MAE is dominated by where the generated population sits,"
echo "     so do not expect a decoder change alone to reach 0.15."
echo "  5. the STEP 4 criteria, then the STEP 5 recommendation."
echo
echo "Honest reading: if the recovered response is a small fraction of the requested"
echo "change, say so. This stage improves ASSEMBLY; it cannot create conditional signal"
echo "that the embeddings do not carry, and nothing here makes the model deployment-ready."
echo "=============================================="
