#!/bin/bash
#SBATCH --job-name=zinc_stageb
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
# STAGE 18: measure the conditional signal FIRST, and run stage B only if the
# measurement says stage B is the right experiment.
#
# 1  SIGNAL AUDIT (minutes, read-only) on the warm-start stage A checkpoint and, for
#    comparison, on the +2k MLE control. For each (N, QED 0.4 -> 0.8) and T in {0.2, 1.0}:
#      delivered  how far the generated embeddings move when ONLY the QED request
#                 changes, with the SAME noise draws
#      required   how far the REAL embedding populations at those two QED values are
#                 apart, from TRAINING molecules within +-0.05 of each target
#      ratio      delivered / required, the cosine between the two directions, and
#                 delivered / within-condition sampling spread
#    plus the SUPPORT at each (N, QED) so a rare target is not read as a model failure,
#    and a non-finite count per cell so a NaN is never read as a displacement.
#
#    This is the hypothesis test. Stage 17 showed the likelihood gap is the same in every
#    model (+0.074 to +0.087) while the sampled slope differs 4.5x, so the question is
#    whether the conditioning physically moves the samples. If the ratio is far below 1
#    with a positive cosine, the conditional path is CAPACITY-bound and stage B follows.
#    If the cosine is poor, more capacity in the same mechanism is the wrong answer and
#    the job stops rather than burning GPU hours.
#
# 2  STAGE B (only if the audit says capacity-bound): from the stage A checkpoint,
#    unfreeze the couplings of the LAST ${UNFREEZE} blocks - those applied FIRST when
#    sampling, so they shape global structure - at a 10x lower learning rate, with the
#    adapters still at the normal rate. ActNorm and the invertible linears stay frozen:
#    they are volume/gauge, not structure. Pure likelihood; the failed structural
#    objective is not used.
#
# 3  CONTROL: the same number of steps from the same stage A checkpoint with the backbone
#    still FULLY FROZEN. The only difference between arms is which tensors can move.
#
# 4  MATCHED EVALUATION of stage B, the control and stage A at identical N, QED,
#    temperatures, counts and decoders.
#
# Writes only to outputs/metrics/signal_*, outputs/checkpoints/stageb_*,
# outputs/samples/stageb_*. Every existing checkpoint is opened read-only.
#
#   STAGEA=outputs/checkpoints/warm_start_stageA/last.pt
#   CONTROL_SRC=outputs/checkpoints/warm_start_control/last.pt
#   UNFREEZE=3   STEPS=2000   LR=1e-4   BACKBONE_LR=1e-5
#   EVAL_T=0.2,0.6   COUNT=500   AUDIT_ONLY=1
#
#     sbatch slurm/18_signal_and_stageb.sh
#
# Never run this file with `bash` directly on the login node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; never train on the login node." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "rc: 2=missing input, 4=bad args, 5=non-finite loss, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 outputs/metrics/signal_stageA/signal.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/18_signal_and_stageb.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
STAGEA="${STAGEA:-outputs/checkpoints/warm_start_stageA/last.pt}"
CONTROL_SRC="${CONTROL_SRC:-outputs/checkpoints/warm_start_control/last.pt}"
UNFREEZE="${UNFREEZE:-3}"
STEPS="${STEPS:-2000}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
EVAL_T="${EVAL_T:-0.2,0.6}"
COUNT="${COUNT:-500}"
TARGET_N="${TARGET_N:-15,23,30}"
TARGET_QED="${TARGET_QED:-0.4,0.6,0.8}"
mkdir -p outputs/logs

echo "=============================================="
echo "Stage 18: conditional-signal audit, then gated stage B"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Stage A source:   ${STAGEA} (READ-ONLY)"
echo "Unfreeze blocks:  ${UNFREEZE} (sampled first)  backbone lr ${BACKBONE_LR}, adapters ${LR}"
echo "=============================================="
"${PYTHON}" -V
nvidia-smi || { echo "ERROR: no GPU visible to this job" >&2; exit 67; }

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${STAGEA}"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: ${required} not found." >&2
        exit 66
    fi
done

echo
echo "---- 1/4 conditional-signal audit ----"
srun "${PYTHON}" -u scripts/19_conditional_signal_audit.py \
    --flow-checkpoint "${STAGEA}" \
    --label stageA \
    --data-dir "${DATA_DIR}" \
    --output-dir outputs/metrics/signal_stageA \
    --target-n "${TARGET_N}" \
    --qed-pair 0.4,0.8 \
    --count 256 \
    --temperature 0.2,1.0 \
    --device cuda

if [[ -f "${CONTROL_SRC}" ]]; then
    srun "${PYTHON}" -u scripts/19_conditional_signal_audit.py \
        --flow-checkpoint "${CONTROL_SRC}" \
        --label mle_control \
        --data-dir "${DATA_DIR}" \
        --output-dir outputs/metrics/signal_control \
        --target-n "${TARGET_N}" \
        --qed-pair 0.4,0.8 \
        --count 256 \
        --temperature 0.2,1.0 \
        --device cuda
fi

JUSTIFIED=$("${PYTHON}" -c "import json;print(json.load(open('outputs/metrics/signal_stageA/signal.json'))['decision']['stage_b_justified'])")
echo
echo "audit verdict: stage_b_justified=${JUSTIFIED}"
if [[ "${AUDIT_ONLY:-0}" == "1" ]]; then
    echo "AUDIT_ONLY=1: stopping. Report: outputs/metrics/signal_stageA/signal.txt"
    exit 0
fi
if [[ "${JUSTIFIED}" != "True" ]]; then
    echo "The audit does NOT support stage B: the conditional displacement is not simply"
    echo "too small in the right direction. Stopping instead of spending GPU hours on the"
    echo "wrong experiment. Read outputs/metrics/signal_stageA/signal.txt for the reading"
    echo "and the recommended alternative."
    exit 0
fi

echo
echo "---- 2/4 stage B: unfreeze ${UNFREEZE} sampled-first blocks' couplings ----"
srun "${PYTHON}" -u scripts/17_structure_finetune.py \
    --flow-checkpoint "${STAGEA}" \
    --data-dir "${DATA_DIR}" \
    --output-dir outputs/checkpoints/stageb_arm \
    --lambda-structure 0 \
    --freeze-backbone \
    --unfreeze-blocks "${UNFREEZE}" \
    --backbone-lr "${BACKBONE_LR}" \
    --learning-rate "${LR}" \
    --max-steps "${STEPS}" \
    --batch-size 64 \
    --val-molecules 2000 \
    --eval-every 250 \
    --save-every 250 \
    --seed 1 \
    --device cuda \
    --resume auto

echo
echo "---- 3/4 control: same steps from the SAME checkpoint, backbone still frozen ----"
srun "${PYTHON}" -u scripts/17_structure_finetune.py \
    --flow-checkpoint "${STAGEA}" \
    --data-dir "${DATA_DIR}" \
    --output-dir outputs/checkpoints/stageb_control \
    --lambda-structure 0 \
    --freeze-backbone \
    --learning-rate "${LR}" \
    --max-steps "${STEPS}" \
    --batch-size 64 \
    --val-molecules 2000 \
    --eval-every 250 \
    --save-every 250 \
    --seed 1 \
    --device cuda \
    --resume auto

echo
echo "---- 4/4 matched evaluation ----"
for pair in "stageb_arm:outputs/checkpoints/stageb_arm/last.pt" \
            "stageb_control:outputs/checkpoints/stageb_control/last.pt" \
            "stageA:${STAGEA}"; do
    name="${pair%%:*}"; checkpoint="${pair#*:}"
    echo
    echo "   [${name}] ${checkpoint}"
    srun "${PYTHON}" -u scripts/13_generate_nqed.py \
        --flow-checkpoint "${checkpoint}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "outputs/samples/stageb_${name}" \
        --target-n "${TARGET_N}" \
        --target-qed "${TARGET_QED}" \
        --temperature "${EVAL_T}" \
        --count "${COUNT}" \
        --batch-size 128 \
        --check-molecules 2000 \
        --ceiling-molecules 1000 \
        --device cuda \
        --allow-weak-conditioning
done

srun "${PYTHON}" -u scripts/19_conditional_signal_audit.py \
    --flow-checkpoint outputs/checkpoints/stageb_arm/last.pt \
    --label stageb_arm \
    --data-dir "${DATA_DIR}" \
    --output-dir outputs/metrics/signal_stageb_arm \
    --target-n "${TARGET_N}" \
    --qed-pair 0.4,0.8 \
    --count 256 \
    --temperature 0.2,1.0 \
    --device cuda

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "GO / NO-GO (stage B arm vs the FROZEN control, matched temperature and counts):"
echo "  GO       QED MAE <= 0.15 AND within-N correlation >= 0.45 AND raw validity >= 15%"
echo "           AND aromatic rings >= 1.0. Both control and QED move the right way, so"
echo "           the conditional path was capacity-bound and the guardrail held."
echo "  PARTIAL  QED improves but structure drops below raw 15% / aromatic 1.0: the"
echo "           trade-off is real and the next question is where on it to sit, not"
echo "           whether to unfreeze more."
echo "  NO-GO    QED MAE still > 0.19 or within-N correlation < 0.3 while structure"
echo "           degrades: FiLM + prior + late couplings cannot express QED in this"
echo "           embedding space. The recommendation then is the conditional"
echo "           MoFlow-style direct atom/bond flow, where the property conditions a"
echo "           DISCRETE graph variable instead of a learned embedding geometry."
echo "  Also compare outputs/metrics/signal_stageb_arm/signal.txt with"
echo "  outputs/metrics/signal_stageA/signal.txt: the ratio must rise, otherwise the"
echo "  extra capacity did not change what it was supposed to change."
echo "=============================================="
