#!/bin/bash
#SBATCH --job-name=zinc_gflow
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
#SBATCH --time=08:00:00

# ---------------------------------------------------------------------------
# STAGE 22a/b: direct conditional graph-flow PROOF OF CONCEPT. Bounded. Resumable.
#
# WHY: stages 17-21 closed off decoder work. The conditional signal is present in
# embedding space (stage 18: delivered/required 0.665, cosine +0.926) and the decoder
# preserves QED differences on REAL held-out embeddings (stage 20: 97.8-99.1%), yet
# realised QED does not follow the request. Stage 20 located the gap - a real-versus-
# generated embedding classifier reaches AUC 0.979 - and stage 21 showed that removing the
# decoder's assembly discontinuity makes conditioning WORSE (slope +0.0311 -> -0.0346),
# because pinning the bond set to the degree sequence everywhere removes the channel the
# signal travelled through. The common factor is a separately trained decoder standing
# between the conditional model and the molecule. This stage removes it.
#
# WHAT IS NOT ASSUMED: no published molecular flow established native target-property
# conditional generation. MoFlow optimises QED by latent gradient ascent with a separately
# trained MLP regressor and reports the BEST QED discovered, not achievement of a request;
# GraphAF, GCPN, MolecularRNN and GraphDF use RL. So nothing licenses the assumption that
# attaching a QED embedding yields control, and this job is built to falsify it.
#
# THE TWO ARMS - this is the whole design:
#   REAL      trained on the true (N, QED) labels
#   NULL      the SAME architecture, budget and seed, trained on QED PERMUTED WITHIN N.
#             Every marginal is identical and the association is destroyed. If the real
#             arm's QED response does not separate from this arm's, the model learned
#             nothing about QED - and every stage from 13 to 21 produced plausible
#             molecules with no QED control, so this is the only honest defence.
#
# THE PRE-REGISTERED UNDERTRAINING TEST: every evaluation scores the same validation
# molecules under their true QED and under QED permuted within N. That conditional
# likelihood gap is zero exactly when the density ignores QED. Its TREND decides what may
# be concluded:
#   rising_undertrained  -> the budget was too small. CONCLUDE NOTHING about the
#                           architecture; extend and re-run.
#   plateaued_zero       -> the density is not using QED; a longer run will not help.
#   plateaued_positive   -> the density notices QED. Whether SAMPLING follows is separate,
#                           and stage 17 found exactly that dissociation in embedding space.
#
# ARCHITECTURE: MoFlow-style one-shot flow over the dataset's own tensors. Bonds through a
# Glow-style stack over the STRICT UPPER TRIANGLE (so symmetry is structural, not repaired),
# then atoms through a stack whose coupling conditioner does one relational
# message-passing step over the bonds. Conditioning enters BOTH every coupling (cINN-style,
# Ardizzone et al.) AND the prior, and the two are reported separately - a prior-only
# conditional flow merely shifts the latent cloud and can raise average QED without
# controlling it, which is precisely the artefact stage 21 produced.
#
# CHEMISTRY IS DELIBERATELY OUTSIDE THE MODEL. A valence constraint is not an invertible
# map and cannot live in a flow without destroying the density. Reported separately:
#   raw        argmax of the generated tensors, NO chemistry - the architectural number
#   corrected  MoFlow's post-hoc valency correction, which keeps the largest connected
#              component and can therefore drop atoms, so exact N is measured both ways
#
# NO oracle QED reranking, NO candidate selection, NO guidance: one latent draw yields one
# molecule. TRAIN split for fitting, VALIDATION for model selection; the test split cannot
# be returned by `load_training_splits` and is never loaded.
#
# PRESERVATION: writes ONLY under outputs/checkpoints/graph_flow_* and
# outputs/metrics/graph_flow. Stage 1-21 checkpoints, outputs and code are untouched; the
# stage-17 warm start is opened READ-ONLY as the baseline.
#
#     sbatch slurm/22_graph_flow_poc.sh
#     ARM=real_only sbatch slurm/22_graph_flow_poc.sh      # skip the null arm
#     STEPS=12000 sbatch slurm/22_graph_flow_poc.sh        # extend after a rising trend
#
# Resumable: each arm reloads its own last.pt and continues at the exact next batch
# (DeterministicBatcher state is checkpointed), so re-submitting after a timeout costs
# nothing. Re-submitting a FINISHED arm is a no-op.
#
# Runtime estimate: ~6 M parameters, batch 32, bond tensor [32, 703, 4] and atom tensor
# [32, 38, 17]. About 1.5-3 h per arm for 6,000 steps on one GPU, so both arms plus the
# CPU-side evaluation fit the 8 h limit with margin. If it times out, resubmit.
# Memory estimate: activations are dominated by the 703-position bond stack at hidden 256;
# measured well under 4 GB. 24 G host RAM and one GPU requested for headroom.
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
    echo "rc: 2=missing data, 4=bad args, 1=traceback/non-finite loss" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${REAL_DIR:-outputs/checkpoints/graph_flow_real}/train.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/22_graph_flow_poc.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
REAL_DIR="${REAL_DIR:-outputs/checkpoints/graph_flow_real}"
NULL_DIR="${NULL_DIR:-outputs/checkpoints/graph_flow_null}"
OUT="${OUT:-outputs/metrics/graph_flow}"
BASELINE="${BASELINE:-outputs/checkpoints/warm_start_stageA/last.pt}"
STEPS="${STEPS:-6000}"
BATCH="${BATCH:-32}"
LR="${LR:-1e-3}"
BOND_BLOCKS="${BOND_BLOCKS:-6}"
ATOM_BLOCKS="${ATOM_BLOCKS:-6}"
HIDDEN="${HIDDEN:-256}"
CONDITION_DROPOUT="${CONDITION_DROPOUT:-0.1}"
NOISE_SCALE="${NOISE_SCALE:-0.6}"
EVAL_EVERY="${EVAL_EVERY:-500}"
TARGET_N="${TARGET_N:-15,23,30}"
TARGET_QED="${TARGET_QED:-0.4,0.6,0.8}"
TEMPERATURES="${TEMPERATURES:-0.6,0.85,1.0}"
COUNT="${COUNT:-256}"
ARM="${ARM:-both}"
mkdir -p outputs/logs "${REAL_DIR}" "${NULL_DIR}" "${OUT}"

echo "=============================================="
echo "Stage 22: direct conditional graph flow (proof of concept)"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Budget:           ${STEPS} steps, batch ${BATCH}, lr ${LR} (arms: ${ARM})"
echo "Real arm:         ${REAL_DIR}"
echo "Null arm:         ${NULL_DIR}  (QED permuted WITHIN N)"
echo "Baseline:         ${BASELINE} (READ-ONLY, stage-17 warm start)"
echo "Evaluation:       ${OUT}"
echo "Preserved:        stage 1-21 checkpoints and outputs are untouched"
echo "=============================================="
"${PYTHON}" -V
"${PYTHON}" -c "import torch; print('cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

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
        --bond-blocks "${BOND_BLOCKS}" \
        --atom-blocks "${ATOM_BLOCKS}" \
        --hidden "${HIDDEN}" \
        --condition-dropout "${CONDITION_DROPOUT}" \
        --noise-scale "${NOISE_SCALE}" \
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
if [[ -f "${BASELINE}" ]]; then
    EVAL_ARGS+=(--baseline-flow-checkpoint "${BASELINE}")
fi
srun "${PYTHON}" -u scripts/24_evaluate_graph_flow.py "${EVAL_ARGS[@]}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo
echo "READ IN THIS ORDER:"
echo "  1. ${REAL_DIR}/training_summary.json -> condition_gap_trend.status."
echo "     If it is 'rising_undertrained', STOP: the budget was too small and the"
echo "     evaluation describes an undertrained model, not the architecture. Resubmit"
echo "     with STEPS=12000 (it resumes; nothing is recomputed)."
echo "     If it is 'plateaued_zero', the density is not using QED and a longer run at"
echo "     this size will not change that - which is itself a decisive negative result."
echo "  2. table A of ${OUT}/graph_flow_evaluation.txt, the response slope, read AGAINST"
echo "     the graph_flow_null row. The null arm saw QED permuted within N, so whatever"
echo "     slope it shows is the floor that means nothing. Separation from it is the"
echo "     entire claim of this stage."
echo "  3. the graph_flow_prior row in the same table. If it matches the full arm, the"
echo "     model is only shifting its conditional prior - average QED moves, control does"
echo "     not - which is the artefact stage 21's 'total' decoder produced."
echo "  4. table B. 'raw' is the architectural validity, with NO chemistry: compare it"
echo "     with our embedding pipeline's ~14% (unconditional arm B) and ~9%"
echo "     (conditional), and with the literature's validity-without-check figures"
echo "     (MoFlow 82%, GraphAF 68% on ZINC250K). 'corrected' is post-hoc and can drop"
echo "     atoms, which is why exact N appears for both."
echo "  5. table D. The two likelihoods are over DIFFERENT representations and are"
echo "     reported side by side deliberately compared to nothing."
echo
echo "This is a proof of concept at a bounded budget. A positive result licenses stage 22b"
echo "(a longer single-arm run), not a deployment claim."
echo "=============================================="
