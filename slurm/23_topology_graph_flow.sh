#!/bin/bash
#SBATCH --job-name=zinc_topoflow
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00

# ---------------------------------------------------------------------------
# STAGE 23: topology-aware bond conditioner. Bounded. Resumable.
#
# WHAT STAGE 22c PROVED (job 340309). `BondSubnet` gives a pair only r_i + r_j and
# |r_i - r_j|, and a six-ring versus two three-rings produced a difference of EXACTLY
# 0.000e+00 on every pair they share. The model could not see topology at a fixed degree
# sequence, and connectivity and ring closure are precisely what needs that. The
# measured consequences at QED 0.8, T=0.85:
#
#     N   real bonds  generated  components  largest  exact N after correction
#    15        15.5        9.9        5.79    8.0/15                    1.6%
#    23        24.7       17.4        7.00   13.4/23                    0.8%
#    30        33.0       25.5        7.17   20.2/30                    0.4%
#
#   74.2% of graphs at N=23 were over-valent AND disconnected AT ONCE, and the carbon
#   fraction was 0.637 against 0.732 real - so the MARGINALS are wrong too and not
#   everything may be blamed on topology.
#
# WHY A CONVOLUTION AND NOT MESSAGE PASSING. Iterating the stage-22 pair -> atom -> pair
# aggregation does NOT fix it: both graphs are 2-regular and are the textbook
# counterexample to 1-WL, so any sum-aggregation scheme over the atom graph assigns them
# identical atom embeddings and therefore identical parameters to every shared pair. A
# masked 2-D convolution over the N x N adjacency grid is not 1-WL-bounded, because it
# reads the pattern as a function of atom INDEX and the positions where the two graphs
# differ fall inside its receptive field. That is the mechanism MoFlow used (10 coupling
# layers of 3x3 convolutions at 512 hidden, 82% validity without correction on ZINC250K).
#
#   THE TRADEOFF, STATED: using the index grid means the density is over CANONICALLY
#   ORDERED graphs rather than being permutation equivariant. That is already true of this
#   project's representation - the atom order is fixed to canonical SMILES order - and of
#   MoFlow and GraphNVP, so it adds no new limitation.
#
#   VERIFIED BEFORE PROPOSING THIS RUN: on the 1-WL pair the stage-22 conditioner gives
#   0.0000e+00 and the stage-23 conditioner a non-zero difference, and it does so with
#   FEWER parameters (1,309,690 against 1,500,538) - so an improvement cannot be
#   explained by capacity. A test asserts both halves of that comparison.
#
# WHAT IS PRESERVED: invertibility (round trip 2.4e-07 at initialisation), exact log
# determinants (checked against the true Jacobian), padding masks, undirected bond
# symmetry (structural - only the strict upper triangle is modelled), variable N, the
# charge-aware 17-type atom vocabulary, native N and QED conditioning through both the
# couplings and the prior, and the existing molecular evaluation. The conditioner reads
# ONLY the coupling's pass-through channels, so the inverse recomputes the same scale and
# shift - the transformed dimensions never leak into it.
#
# NUMERICS: max_log_scale 0.75 and max_shift 3.0, and they are NOT re-stated here - every
# script derives them from GraphFlowConfig, which is what stage 22a got wrong (the script
# hard-coded 2.0 while the dataclass said 0.75 and the checkpoint trained at 2.0). The
# inverse is now checked AT INITIALISATION and the run REFUSES to start if it is
# ill-conditioned, re-checked at every evaluation, and reported on every generation call.
# Resuming with a different configuration is refused rather than silently accepted.
#
# THE ARMS:
#   TOPOLOGY  the new conditioner on the true (N, QED) labels
#   NULL      the SAME architecture, budget and seed with QED PERMUTED WITHIN N. Every
#             marginal identical, the association destroyed. If the real arm's QED
#             response does not separate from this, it learned nothing about QED.
#   PREVIOUS  the stage-22 pairwise checkpoint, evaluated under identical requests. It is
#             opened READ-ONLY and is the thing stage 23 must beat.
#   BASELINE  the stage-17 embedding GNF, also READ-ONLY.
#
# A more expressive conditioner is NOT automatically a valid molecular generator. The
# acceptance criteria, in order:
#   1. fragmentation falls and RAW validity rises against the stage-22 arm;
#   2. this is NOT bought by pushing bond density up into over-valence - 'bonds' must move
#      towards 'real' while 'over-val' does not climb with it;
#   3. QED responsiveness survives on structurally valid, exact-N molecules.
# Raw validity is reported separately from post-correction validity throughout, and
# correction can drop atoms, so exact N is reported both ways.
#
# PRESERVATION: writes ONLY under outputs/checkpoints/graph_flow_topology*,
# outputs/checkpoints/graph_flow_topology_null and outputs/metrics/graph_flow_topology.
# Stage 1-22 checkpoints, outputs and code are untouched.
#
#     sbatch slurm/23_topology_graph_flow.sh
#     ARM=real_only sbatch slurm/23_topology_graph_flow.sh     # skip the null arm
#     STEPS=18000 sbatch slurm/23_topology_graph_flow.sh       # extend after a rising trend
#
# Resumable: each arm reloads its own last.pt and continues at the exact next batch, so
# re-submitting after a timeout costs nothing and a finished arm is a no-op.
#
# Runtime estimate: ~1.5M parameters. The conditioner replaces a per-pair MLP with 3
# masked 3x3 convolutions at 64 channels on a 38x38 grid, so a step costs roughly 1.5-2x
# stage 22a's. Stage 22a ran 6,000 steps in about 2 h, so expect 3-4 h per arm for 9,000
# steps; both arms plus the CPU evaluation fit the 12 h limit. If it times out, resubmit.
# Memory estimate: activations are dominated by [batch, 64, 38, 38] per conv layer -
# about 12 MB per layer at batch 32, so well under 4 GB including the backward graph.
# 24 G host RAM and one GPU are requested for headroom.
#
# Never run this file with `bash` directly on the login node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; this trains two flows." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "rc: 2=missing data, 4=bad args or config mismatch, 1=traceback/non-finite loss" >&2
    echo "    rc=1 with 'ill_conditioned_inverse_at_initialisation' means the numerics" >&2
    echo "    were refused BEFORE training - lower --max-log-scale, do not override it." >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${REAL_DIR:-outputs/checkpoints/graph_flow_topology}/train.log" >&2
    echo "This job is RESUMABLE: re-submitting continues each arm at its next batch." >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/23_topology_graph_flow.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
REAL_DIR="${REAL_DIR:-outputs/checkpoints/graph_flow_topology}"
NULL_DIR="${NULL_DIR:-outputs/checkpoints/graph_flow_topology_null}"
OUT="${OUT:-outputs/metrics/graph_flow_topology}"
PREVIOUS="${PREVIOUS:-outputs/checkpoints/graph_flow_real/last.pt}"
BASELINE="${BASELINE:-outputs/checkpoints/warm_start_stageA/last.pt}"
STEPS="${STEPS:-9000}"
BATCH="${BATCH:-32}"
LR="${LR:-1e-3}"
BOND_BLOCKS="${BOND_BLOCKS:-6}"
ATOM_BLOCKS="${ATOM_BLOCKS:-6}"
HIDDEN="${HIDDEN:-256}"
CONV_HIDDEN="${CONV_HIDDEN:-48}"
CONV_LAYERS="${CONV_LAYERS:-3}"
EVAL_EVERY="${EVAL_EVERY:-750}"
TARGET_N="${TARGET_N:-15,23,30}"
TARGET_QED="${TARGET_QED:-0.4,0.6,0.8}"
# T=1.0 first: reduced-temperature sampling on a dequantized one-hot representation biases
# every variable towards its MAJORITY class, which for bonds is "no bond" - part of stage
# 22c's bond deficit is that bias rather than the model. T=1.0 is the unbiased reading.
TEMPERATURES="${TEMPERATURES:-1.0,0.85,0.6}"
COUNT="${COUNT:-256}"
ARM="${ARM:-both}"
mkdir -p outputs/logs "${REAL_DIR}" "${NULL_DIR}" "${OUT}"

echo "=============================================="
echo "Stage 23: topology-aware bond conditioner"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Budget:           ${STEPS} steps, batch ${BATCH}, lr ${LR} (arms: ${ARM})"
echo "Conditioner:      topology (masked 2-D conv, ${CONV_LAYERS} layers, ${CONV_HIDDEN} channels)"
echo "Numerics:         from GraphFlowConfig - NOT overridden here"
echo "Topology arm:     ${REAL_DIR}"
echo "Null arm:         ${NULL_DIR}  (QED permuted WITHIN N)"
echo "Stage-22 arm:     ${PREVIOUS} (READ-ONLY, pairwise conditioner)"
echo "Stage-17 arm:     ${BASELINE} (READ-ONLY, embedding GNF)"
echo "Evaluation:       ${OUT}"
echo "Preserved:        stage 1-22 checkpoints and outputs are untouched"
echo "=============================================="
"${PYTHON}" -V
"${PYTHON}" -c "import torch; print('cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${PYTHON}" -c "from zinc_gnf.graph_flow import GraphFlowConfig as C; c=C(); print(f'numerical source of truth: max_log_scale={c.max_log_scale} max_shift={c.max_shift} noise_scale={c.noise_scale} bond_conditioner={c.bond_conditioner}')"

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: ${required} not found." >&2
        exit 66
    fi
done

train_arm() {
    local directory="$1"; shift
    echo
    echo "---- training: ${directory} ----"
    srun "${PYTHON}" -u scripts/23_train_graph_flow.py \
        --output-dir "${directory}" \
        --data-dir "${DATA_DIR}" \
        --max-steps "${STEPS}" \
        --batch-size "${BATCH}" \
        --learning-rate "${LR}" \
        --bond-conditioner topology \
        --bond-blocks "${BOND_BLOCKS}" \
        --atom-blocks "${ATOM_BLOCKS}" \
        --hidden "${HIDDEN}" \
        --conv-hidden "${CONV_HIDDEN}" \
        --conv-layers "${CONV_LAYERS}" \
        --eval-every "${EVAL_EVERY}" \
        --save-every "${EVAL_EVERY}" \
        --seed 17 \
        --device auto \
        "$@"
}

if [[ "${ARM}" == "both" || "${ARM}" == "real_only" ]]; then
    train_arm "${REAL_DIR}"
fi
if [[ "${ARM}" == "both" || "${ARM}" == "null_only" ]]; then
    train_arm "${NULL_DIR}" --shuffle-qed --shuffle-seed 99
fi

echo
echo "---- matched evaluation (CPU) ----"
EVAL_ARGS=(--graph-flow-checkpoint "${REAL_DIR}/last.pt"
           --output-dir "${OUT}"
           --data-dir "${DATA_DIR}"
           --target-n "${TARGET_N}"
           --target-qed "${TARGET_QED}"
           --temperature "${TEMPERATURES}"
           --count "${COUNT}"
           --seed 17
           --device cpu)
if [[ -f "${NULL_DIR}/last.pt" ]]; then
    EVAL_ARGS+=(--null-checkpoint "${NULL_DIR}/last.pt")
fi
if [[ -f "${PREVIOUS}" ]]; then
    EVAL_ARGS+=(--previous-graph-checkpoint "${PREVIOUS}")
fi
if [[ -f "${BASELINE}" ]]; then
    EVAL_ARGS+=(--baseline-flow-checkpoint "${BASELINE}")
fi
srun "${PYTHON}" -u scripts/24_evaluate_graph_flow.py "${EVAL_ARGS[@]}"

echo
echo "---- failure attribution on the new arm (CPU, read-only) ----"
srun "${PYTHON}" -u scripts/25_graph_flow_diagnostics.py \
    --checkpoint "${REAL_DIR}/last.pt" \
    --output-dir "${OUT}/diagnostics" \
    --data-dir "${DATA_DIR}" \
    --target-n "${TARGET_N}" \
    --target-qed 0.8 \
    --temperature 1.0 \
    --count "${COUNT}" \
    --seed 17

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo
echo "READ IN THIS ORDER:"
echo "  0. ${REAL_DIR}/train.log -> 'inverse at initialisation'. If the run refused to"
echo "     start, the numerics were rejected before any GPU time was spent; fix those"
echo "     first. Then check inverse_round_trip_worst in the history: above 1e-3 the"
echo "     samples are NOT from the trained density and no generation metric is valid."
echo "  1. table C of ${OUT}/graph_flow_evaluation.txt - THE STRUCTURE TABLE. This is"
echo "     what stage 23 exists to move. Stage 22 gave 17.4 bonds against 24.7 real at"
echo "     N=23 with 7.00 components and 74.2% over-valent-and-disconnected. 'bonds'"
echo "     must rise TOWARDS 'real' while 'over-val' does NOT rise with it: buying"
echo "     validity by adding bonds indiscriminately shows up as both columns climbing"
echo "     together, and that is a FAILURE, not a pass. Compare the graph_flow and"
echo "     graph_flow_previous rows directly - same grid, same seeds."
echo "  2. 'raw valid' in the same table, against stage 22's 0.0012 and the embedding"
echo "     pipeline's ~0.09. This is the architectural number, with NO chemistry applied."
echo "  3. ${REAL_DIR}/training_summary.json -> condition_gap_trend.status. If it is"
echo "     'rising_undertrained' the budget was too small: resubmit with STEPS=18000"
echo "     (it resumes) and do not conclude anything about QED yet."
echo "  4. table A, the QED response slope, read AGAINST the graph_flow_null row. The"
echo "     null arm saw QED permuted within N, so its slope is the floor that means"
echo "     nothing. Separation from it is the only evidence of QED learning - and it only"
echo "     counts if exact N is no longer ~1%, because a slope over fragments is a slope"
echo "     over the wrong molecules."
echo "  5. the graph_flow_prior row. If it matches the full arm, the model is only"
echo "     shifting its conditional prior rather than controlling QED."
echo "  6. ${OUT}/diagnostics/graph_flow_diagnostics.json for the per-cause breakdown and"
echo "     the atom/bond marginals against real data. Stage 22c's carbon fraction was"
echo "     0.637 against 0.732 real; if that gap persists the ATOM block is still"
echo "     under-fit and topology was not the only problem."
echo
echo "A more expressive conditioner is not automatically a valid molecular generator."
echo "This is a bounded proof of concept: a positive result licenses a longer run, not a"
echo "deployment claim."
echo "=============================================="
