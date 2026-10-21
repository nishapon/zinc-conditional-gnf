#!/bin/bash
#SBATCH --job-name=zinc_warmstart
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
# STAGE 17: can conditioning be ADDED to the good unconditional generator instead of
# learned from scratch alongside it?
#
#   1  AUDIT (fails the job if the transfer is not exact)
#      - every tensor of both state dicts categorised; the whitening buffers must match
#      - the warm-started conditional model is compared with the unconditional one on
#        real embeddings for five different conditions, on latents, log-determinant,
#        log-likelihood, inverse and samples. FiLM is `h*(1+gamma)+beta` with zeroed
#        weights (exactly the identity) and the prior heads' weights are zeroed with
#        their biases set to the unconditional mu / pre_sigma, so the expected result is
#        BIT-EXACT. Anything else aborts with rc 3.
#      - ablation on the EXISTING conditional checkpoint, no training: as trained, FiLM
#        neutralised, prior condition-dependence frozen, both. This is what separates
#        "FiLM destroys structure" from "the prior does" from "the backbone weights do".
#
#   2  STAGE A: warm start + FROZEN BACKBONE, pure likelihood (lambda_structure 0 - the
#      failed structural objective is NOT used). Only FiLM, the condition embedding and
#      the prior heads train, so the pretrained cross-node structure cannot be
#      overwritten. Short run, resumable.
#
#   3  CONTROL: the same number of steps, same learning rate and same seed, continued on
#      the EXISTING conditional checkpoint with all parameters trainable. That is the
#      honest comparison: "more training of what we have" versus "warm start + adapters".
#
#   4  MATCHED EVALUATION of four models with identical N, QED, temperatures, sample
#      counts and decoders:
#        unconditional 50k (QED-blind baseline), existing conditional 50k,
#        warm-start stage A, control.
#      The unconditional model is evaluated with scripts/05_generate.py at the same N and
#      candidate budget, since it cannot be given a QED target.
#
# NOTHING IS OVERWRITTEN. Every existing checkpoint is opened read-only; this job writes
# only to outputs/checkpoints/warm_start*, outputs/samples/warm_start*,
# outputs/metrics/warm_start.
#
#   UNCOND=outputs/checkpoints/flow_uncond_ar/seed_1/last.pt   (Arm B 50k)
#   COND=outputs/checkpoints/flow_nqed/seed_1/best.pt          (joint 50k)
#   STEPS=2000   LR=1e-4   EVAL_T=0.2,0.6   COUNT=500
#   TARGET_N=15,23,30   TARGET_QED=0.4,0.6,0.8
#   AUDIT_ONLY=1   stop after step 1
#
#     sbatch slurm/17_warm_start_experiment.sh
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
    echo "rc: 2=missing input, 3=INCOMPATIBLE or NOT EQUIVALENT transfer (read" >&2
    echo "    outputs/metrics/warm_start/warm_start.txt - do not proceed), 4=bad args," >&2
    echo "    5=non-finite loss, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/17_warm_start_experiment.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
UNCOND="${UNCOND:-outputs/checkpoints/flow_uncond_ar/seed_1/last.pt}"
COND="${COND:-outputs/checkpoints/flow_nqed/seed_1/best.pt}"
AUDIT="outputs/metrics/warm_start"
WARM="outputs/checkpoints/warm_start_stageA"
CONTROL="outputs/checkpoints/warm_start_control"
STEPS="${STEPS:-2000}"
LR="${LR:-1e-4}"
EVAL_T="${EVAL_T:-0.2,0.6}"
COUNT="${COUNT:-500}"
TARGET_N="${TARGET_N:-15,23,30}"
TARGET_QED="${TARGET_QED:-0.4,0.6,0.8}"
mkdir -p outputs/logs "${AUDIT}"

echo "=============================================="
echo "Stage 17: warm-started conditional flow, with a matched control"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Unconditional:    ${UNCOND} (READ-ONLY)"
echo "Conditional:      ${COND} (READ-ONLY)"
echo "Steps per arm:    ${STEPS}  lr ${LR}"
echo "Evaluation:       N=${TARGET_N} QED=${TARGET_QED} T=${EVAL_T}, ${COUNT} per cell"
echo "=============================================="
"${PYTHON}" -V
nvidia-smi || { echo "ERROR: no GPU visible to this job" >&2; exit 67; }

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${UNCOND}" "${COND}"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: ${required} not found." >&2
        exit 66
    fi
done

echo
echo "---- 1/4 warm-start audit (compatibility, equivalence, component ablation) ----"
if [[ -f "${AUDIT}/warm_start.pt" ]]; then
    echo "note: ${AUDIT}/warm_start.pt already exists; the audit refuses to overwrite it."
    echo "      Remove that single file by hand if you want to redo the audit."
else
    srun "${PYTHON}" -u scripts/18_warm_start_audit.py \
        --unconditional-checkpoint "${UNCOND}" \
        --conditional-checkpoint "${COND}" \
        --output-dir "${AUDIT}" \
        --data-dir "${DATA_DIR}" \
        --molecules 64 \
        --ablation-molecules 256 \
        --ablation-n 23 \
        --ablation-qed 0.8 \
        --temperature 0.2 \
        --device cuda
fi

if [[ "${AUDIT_ONLY:-0}" == "1" ]]; then
    echo "AUDIT_ONLY=1: stopping after the audit. Report: ${AUDIT}/warm_start.txt"
    exit 0
fi

echo
echo "---- 2/4 stage A: warm start, FROZEN backbone, pure likelihood ----"
srun "${PYTHON}" -u scripts/17_structure_finetune.py \
    --flow-checkpoint "${AUDIT}/warm_start.pt" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${WARM}" \
    --lambda-structure 0 \
    --freeze-backbone \
    --max-steps "${STEPS}" \
    --batch-size 64 \
    --learning-rate "${LR}" \
    --val-molecules 2000 \
    --eval-every 250 \
    --save-every 250 \
    --seed 1 \
    --device cuda \
    --resume auto

echo
echo "---- 3/4 control: same budget, continued on the EXISTING conditional run ----"
srun "${PYTHON}" -u scripts/17_structure_finetune.py \
    --flow-checkpoint "${COND}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${CONTROL}" \
    --lambda-structure 0 \
    --max-steps "${STEPS}" \
    --batch-size 64 \
    --learning-rate "${LR}" \
    --val-molecules 2000 \
    --eval-every 250 \
    --save-every 250 \
    --seed 1 \
    --device cuda \
    --resume auto

echo
echo "---- 4/4 matched evaluation of all four models ----"
for pair in "warm_start_stageA:${WARM}/last.pt" "control:${CONTROL}/last.pt" \
            "conditional_50k:${COND}"; do
    name="${pair%%:*}"; checkpoint="${pair#*:}"
    echo
    echo "   [${name}] ${checkpoint}"
    srun "${PYTHON}" -u scripts/13_generate_nqed.py \
        --flow-checkpoint "${checkpoint}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "outputs/samples/warm_start_${name}" \
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

echo
echo "   [unconditional_50k] ${UNCOND} - QED-blind baseline at the same N and budget"
srun "${PYTHON}" -u scripts/05_generate.py \
    --flow-checkpoint "${UNCOND}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "outputs/samples/warm_start_unconditional_50k" \
    --count "$(( COUNT * 3 ))" \
    --temperature "${EVAL_T}" \
    --batch-size 128 \
    --device cuda

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "READ IN THIS ORDER:"
echo "  1. ${AUDIT}/warm_start.txt"
echo "     - the equivalence block MUST be bit-exact (or ~1e-7). If it is not, stop:"
echo "       everything downstream is invalid."
echo "     - the ablation table says WHICH conditional component costs structure. If"
echo "       neutralising FiLM or freezing the prior's condition-dependence restores"
echo "       aromaticity, conditioning is the proximate cause; if neither does, the"
echo "       damage is baked into the conditional backbone weights, which is precisely"
echo "       what the warm start replaces."
echo "  2. outputs/samples/warm_start_{warm_start_stageA,control,conditional_50k}/nqed_report.txt"
echo "     and outputs/samples/warm_start_unconditional_50k/*/metrics.txt"
echo
echo "GO / NO-GO (stage A vs CONTROL, same temperature, same counts):"
echo "  GO   raw validity >= 15%, aromatic rings >= 1.0, fallback below the control, and"
echo "       QED MAE at or below the control's with a non-zero QED-only permutation gap."
echo "       Then stage B (unfreeze a limited subset at a lower learning rate) is"
echo "       justified - one factor at a time."
echo "  PARTIAL  structure retained (aromatic rings >= 1.0) but QED MAE no better than"
echo "       0.19: the adapters preserve the generator but cannot express QED control."
echo "       That is informative, not a failure, and stage B is the next test."
echo "  NO-GO  structure collapses to the conditional model's level within ~2000 steps"
echo "       even with the backbone frozen: conditioning is then incompatible with this"
echo "       embedding space, and the recommendation is the conditional MoFlow-style"
echo "       direct atom/bond flow instead of further adaptation."
echo "=============================================="
