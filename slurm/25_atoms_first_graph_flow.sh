#!/bin/bash
#SBATCH --job-name=zinc_atomsfirst
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
#SBATCH --time=16:00:00

# ---------------------------------------------------------------------------
# STAGE 25: the ATOMS-FIRST factorisation. Bounded. Resumable. Two arms.
#
# WHAT STAGE 24 ESTABLISHED (commit 54b7ed4, job over graph_flow_topology/last.pt).
# Stage 23's bond counts were right and its molecules were not: at N=23, QED 0.8, T=1.0,
# 24.45 bonds against 24.65 real and 1.20 components, with 89.5% of graphs carrying an
# over-valent atom and raw validity 7.81%. The audit then separated the two candidate
# explanations:
#
#   * best-case atom relabelling made 98.4% of those graphs valence-FEASIBLE by changing
#     about 10.5% of atoms. The generated bond patterns were realisable; the atom choice
#     was not. (Feasibility under the audit's allowed-valence rules - not sanitisation,
#     and not achieved validity.)
#   * real atoms with generated bonds gave 2.34% raw validity; generated atoms with real
#     bonds gave 14.84%. Substituting the BONDS helps more than substituting the atoms.
#   * aromaticity: real N=23 molecules carry ~1.848 aromatic rings each and the
#     representation preserves aromaticity on 100% of sampled real molecules, while
#     generated graphs have essentially none and only 2.64% of their 5-/6-rings carry the
#     Kekule alternation against 45.6% in the real reference.
#
# And the mechanism, from the code: `encode` ran the bond stack with `extra=None`, so the
# block choosing bond orders never saw an atom type, while valence is a constraint between
# an atom's TYPE and the sum of its incident bond orders.
#
# WHAT THIS JOB CHANGES. The factorisation, from
#
#     bonds_first   log p(G|c) = log p_bond(A|c) + log p_atom(X|A,c)
# to
#     atoms_first   log p(G|c) = log p_atom(X|c) + log p_bond(A|X,c)
#
# so the bond conditioner receives both endpoints' identity, formal charge and allowed
# valence through `AtomChemistryContext` (20 per-pair channels at the default settings).
#
#   BOTH ARE EXACT FACTORISATIONS OF THE SAME JOINT. This job does NOT claim that
#   bonds-first is incapable of representing chemically valid molecules. The hypothesis is
#   about learnability, inductive bias and sampling quality, and it is falsifiable: the
#   comparison script applies stage 25's acceptance criteria and can return "no".
#
# WHAT IS PRESERVED, and tested: exact log-determinants against the true Jacobian, exact
# inverses (2.4e-07 at initialisation), padding masks, undirected bond symmetry
# (structural - only the strict upper triangle is modelled), variable N, the charge-aware
# 17-type vocabulary, native (N, QED) conditioning through both the couplings and the
# prior, and conditioning dropout. The conditioners still read ONLY the couplings'
# pass-through channels, so the inverse recomputes the same scale and shift.
#
#   CONTINUOUS, NOT DISCRETE, CONDITIONING. The default `atom_context_mode=dequantized`
#   feeds the bond conditioner the same continuous dequantized atom channels the
#   likelihood sees, so nothing non-differentiable and nothing non-invertible enters the
#   likelihood path, and training and sampling condition on the same quantity. The
#   `onehot` variant (hard argmax identities) is available and LABELLED: its conditional
#   is still a normalised density in the bond variable, so the dequantization bound
#   stands, but the density becomes piecewise in the atom channels. It is not used here.
#
#   NOT A CAPACITY CHANGE. 1,258,778 parameters against the stage-23 baseline's
#   1,309,690 - 0.961x, strictly FEWER. A win therefore cannot be attributed to capacity;
#   a loss would leave the question open and a capacity-matched rerun owed. A test pins
#   both numbers.
#
#   NO VALENCE PENALTY AND NO CORRECTION IN THE OBJECTIVE. Training stays pure maximum
#   likelihood. Post-hoc correction is reported separately, never merged with the native
#   numbers, and always beside the atoms it dropped. An auxiliary valence loss remains a
#   SEPARATE experimental variant and is not part of this job.
#
# THE ARMS:
#   ATOMS_FIRST  the new factorisation on the true (N, QED) labels
#   NULL         the SAME architecture, budget and seed with QED PERMUTED WITHIN N. Every
#                marginal identical, the association destroyed. If the real arm's QED
#                response does not separate from this, it learned nothing about QED.
#   BONDS_FIRST  the stage-23 18,000-step checkpoint, opened READ-ONLY. The thing stage 25
#                must beat, evaluated under identical requests, seeds and temperatures.
#
# BUDGET. 9,000 steps per arm by default, with an evaluation every 750. That is an INTERIM
# read and NOT the matched comparison: the baseline has 18,000 steps, so at 9,000 the new
# arm is disadvantaged. If the interim read is promising - or ambiguous - resubmit with
# STEPS=18000 and both arms continue from their own last.pt to the matched budget. Read
# condition_gap_trend.status before concluding anything: 'rising_undertrained' means the
# budget was too small and no architectural conclusion is warranted.
#
# PRESERVATION. Writes ONLY under outputs/checkpoints/graph_flow_atoms_first,
# outputs/checkpoints/graph_flow_atoms_first_null and
# outputs/metrics/graph_flow_atoms_first. Stage 1-24 checkpoints, outputs and code are
# untouched; the stage-23 checkpoint is opened read-only.
#
#     sbatch slurm/25_atoms_first_graph_flow.sh
#     STEPS=18000 sbatch slurm/25_atoms_first_graph_flow.sh   # the MATCHED comparison
#     ARM=real_only sbatch slurm/25_atoms_first_graph_flow.sh # skip the null arm
#     ARM=eval_only sbatch slurm/25_atoms_first_graph_flow.sh # re-run the comparison only
#
# Resumable: each arm reloads its own last.pt and continues at the exact next batch, so
# re-submitting after a timeout costs nothing and a finished arm is a no-op.
#
# Runtime estimate: 1.26M parameters. The atom stack is now 1-D convolutional rather than
# a per-atom MLP with a relational term, and the bond conditioner's stem grew by 20 input
# channels, so a step costs about the same as stage 23's - which ran 9,000 steps in
# roughly 3-4 h. Expect 3-5 h per arm, plus about 40 min for the CPU evaluation, audit and
# comparison. Two arms at 9,000 steps fit the 16 h limit; at STEPS=18000 submit twice or
# use ARM to split them.
# Memory estimate: activations are dominated by the bond conditioner's [batch, 48, 38, 38]
# maps plus the atom stack's [batch, 128, 38], under 4 GB at batch 32 including the
# backward graph. 24 G host RAM and one GPU are requested for headroom.
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
    echo "rc: 2=missing data or checkpoint, 4=bad args or config mismatch," >&2
    echo "    1=traceback/non-finite loss/ill-conditioned inverse" >&2
    echo "    rc=4 from training with 'DIFFERENT model configuration' means the output" >&2
    echo "    directory holds a checkpoint of another architecture - use a fresh one." >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${REAL_DIR:-outputs/checkpoints/graph_flow_atoms_first}/train.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/25_atoms_first_graph_flow.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
REAL_DIR="${REAL_DIR:-outputs/checkpoints/graph_flow_atoms_first}"
NULL_DIR="${NULL_DIR:-outputs/checkpoints/graph_flow_atoms_first_null}"
OUT="${OUT:-outputs/metrics/graph_flow_atoms_first}"
# the stage-23 baseline and its own null arm: READ-ONLY, never written to
BONDS_FIRST="${BONDS_FIRST:-outputs/checkpoints/graph_flow_topology/last.pt}"
BONDS_FIRST_NULL="${BONDS_FIRST_NULL:-outputs/checkpoints/graph_flow_topology_null/last.pt}"
STEPS="${STEPS:-9000}"
BATCH="${BATCH:-32}"
LR="${LR:-1e-3}"
BOND_BLOCKS="${BOND_BLOCKS:-6}"
ATOM_BLOCKS="${ATOM_BLOCKS:-6}"
HIDDEN="${HIDDEN:-256}"
CONV_HIDDEN="${CONV_HIDDEN:-48}"
ATOM_CONV_HIDDEN="${ATOM_CONV_HIDDEN:-128}"
ATOM_CONTEXT_DIM="${ATOM_CONTEXT_DIM:-8}"
ATOM_CONTEXT_MODE="${ATOM_CONTEXT_MODE:-dequantized}"
EVAL_EVERY="${EVAL_EVERY:-750}"
TARGET_N="${TARGET_N:-15,23,30}"
TARGET_QED="${TARGET_QED:-0.4,0.6,0.8}"
TEMPERATURES="${TEMPERATURES:-1.0,0.85}"
COUNT="${COUNT:-256}"
ARM="${ARM:-both}"
mkdir -p outputs/logs "${REAL_DIR}" "${NULL_DIR}" "${OUT}"

echo "=============================================="
echo "Stage 25: atoms-first conditional graph flow"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Factorisation:    atoms_first - p(X|c) p(A|X,c)"
echo "Atom context:     ${ATOM_CONTEXT_MODE} channels, embed ${ATOM_CONTEXT_DIM}"
echo "Budget:           ${STEPS} steps, batch ${BATCH}, lr ${LR} (arms: ${ARM})"
echo "Numerics:         from GraphFlowConfig - NOT overridden here"
echo "Atoms-first arm:  ${REAL_DIR}"
echo "Null arm:         ${NULL_DIR}  (QED permuted WITHIN N)"
echo "Stage-23 arm:     ${BONDS_FIRST} (READ-ONLY, bonds-first, 18,000 steps)"
echo "Evaluation:       ${OUT}"
echo "Preserved:        stage 1-24 checkpoints and outputs are untouched"
echo "=============================================="
"${PYTHON}" -V
"${PYTHON}" -c "import torch; print('cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${PYTHON}" -c "
from zinc_gnf.graph_flow import ConditionalGraphFlow as M, GraphFlowConfig as C
c = C()
print(f'numerical source of truth: max_log_scale={c.max_log_scale} max_shift={c.max_shift} noise_scale={c.noise_scale} default factorisation={c.factorisation}')
a = M(C(factorisation='atoms_first')).parameter_report()
b = M(C()).parameter_report()
print(f'parameters: atoms_first {a[\"total\"]:,} vs bonds_first {b[\"total\"]:,} = {a[\"total\"] / b[\"total\"]:.3f}x (a win cannot be capacity)')
"

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: ${required} not found." >&2
        exit 66
    fi
done
if [[ ! -f "${BONDS_FIRST}" ]]; then
    echo "ERROR: the stage-23 baseline ${BONDS_FIRST} was not found; stage 25 is a" >&2
    echo "       COMPARISON and must not be run without the arm it is compared against." >&2
    exit 66
fi

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
        --factorisation atoms_first \
        --atom-conv-hidden "${ATOM_CONV_HIDDEN}" \
        --atom-context-dim "${ATOM_CONTEXT_DIM}" \
        --atom-context-mode "${ATOM_CONTEXT_MODE}" \
        --bond-conditioner topology \
        --bond-blocks "${BOND_BLOCKS}" \
        --atom-blocks "${ATOM_BLOCKS}" \
        --hidden "${HIDDEN}" \
        --conv-hidden "${CONV_HIDDEN}" \
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
echo "---- THE comparison: atoms-first against bonds-first (CPU) ----"
COMPARE_ARGS=(--atoms-first-checkpoint "${REAL_DIR}/last.pt"
              --bonds-first-checkpoint "${BONDS_FIRST}"
              --output-dir "${OUT}/comparison"
              --data-dir "${DATA_DIR}"
              --target-n "${TARGET_N}"
              --target-qed "${TARGET_QED}"
              --temperature "${TEMPERATURES}"
              --count "${COUNT}"
              --seed 17
              --device cpu)
if [[ -f "${NULL_DIR}/last.pt" ]]; then
    COMPARE_ARGS+=(--null-checkpoint "${NULL_DIR}/last.pt")
fi
srun "${PYTHON}" -u scripts/27_compare_factorisations.py "${COMPARE_ARGS[@]}"

echo
echo "---- the standing matched evaluation, both arms under one protocol (CPU) ----"
EVAL_ARGS=(--graph-flow-checkpoint "${REAL_DIR}/last.pt"
           --previous-graph-checkpoint "${BONDS_FIRST}"
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
srun "${PYTHON}" -u scripts/24_evaluate_graph_flow.py "${EVAL_ARGS[@]}"

echo
echo "---- the stage-24 valence and aromaticity audit, on the NEW arm (CPU) ----"
srun "${PYTHON}" -u scripts/26_valence_aromaticity_audit.py \
    --checkpoint "${REAL_DIR}/last.pt" \
    --output-dir "${OUT}/valence_audit" \
    --data-dir "${DATA_DIR}" \
    --target-n "${TARGET_N}" \
    --target-qed 0.8 \
    --temperature 1.0 \
    --count "${COUNT}" \
    --seed 17

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}" "${OUT}/comparison"
echo
echo "READ IN THIS ORDER:"
echo "  0. ${REAL_DIR}/train.log -> 'inverse at initialisation', then"
echo "     inverse_round_trip_worst in the history. Above 1e-3 the samples are NOT from"
echo "     the trained density and no generation number below is valid."
echo "  1. ${OUT}/comparison/factorisation_comparison.txt - THE table. Section A is the"
echo "     native, no-chemistry comparison against the stage-23 arm under identical"
echo "     requests. Raw validity is the architectural number; everything in section C"
echo "     happened AFTER the model, and 'atoms LOST to correction' is the price of it."
echo "  2. section B. Aromaticity is a SEPARATE acceptance criterion: stage 24 measured"
echo "     ~1.848 aromatic rings per real N=23 molecule and 45.6% alternating 5-/6-rings"
echo "     against the old arm's 2.64%. Better valence with flat aromaticity is the"
echo "     'validity_only' verdict, not a pass."
echo "  3. section D. QED restricted to VALID, EXACT-N molecules, with its support count."
echo "     A slope over corrected fragments is a slope over the wrong molecules. Read it"
echo "     against the atoms_first_null column - the shuffled-QED arm is the floor that"
echo "     means nothing."
echo "  4. the VERDICT block. The criteria are conjunctive and it can return 'no'."
echo "     'capacity_not_the_explanation' tells you which way a null result may be read."
echo "  5. ${REAL_DIR}/training_summary.json -> condition_gap_trend.status. If it is"
echo "     'rising_undertrained' the budget was too small: resubmit with STEPS=18000"
echo "     (it resumes) and conclude nothing about QED yet."
echo "  6. ${OUT}/valence_audit/audit.log for the per-element valence breakdown and the"
echo "     best-case relabelling counterfactual on the NEW arm. If best-case feasibility"
echo "     is still far above achieved validity, the atom choice is still the gap."
echo
echo "Both factorisations are exact factorisations of the same joint. A positive result"
echo "here licenses a longer run, not a deployment claim; and 100% corrected validity is"
echo "not native generation success."
echo "=============================================="
