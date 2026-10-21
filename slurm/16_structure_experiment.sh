#!/bin/bash
#SBATCH --job-name=zinc_structure
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
# STAGE 16: the decisive bounded experiment. TWO short fine-tunes from the SAME 50k
# joint checkpoint, differing in ONE factor, then the SAME evaluation on both, then a
# matched-temperature sweep of the untouched 50k checkpoint as the third reference.
#
#   control  lambda_structure = 0  -> continued pure maximum likelihood
#   arm      lambda_structure = L  -> the same plus a moment-matching penalty on what
#                                     the FROZEN decoder says about the flow's OWN
#                                     samples (bonds per atom, cycle rank, double-bond
#                                     fraction, isolated atoms, valence overflow),
#                                     measured against TRAINING moments
#
# Same seed, steps, batch, learning rate, temperature, architecture, autoencoder and
# split in both. A difference between them is attributable to the structural term.
#
# WHY THIS EXPERIMENT. Our likelihood covers 64*N per-node dimensions and is dominated
# by per-node marginals, while the chemistry is carried by O(N^2) pairwise relations
# that barely enter it. The shuffled-node control showed the consequence directly: real
# embeddings regrouped at random reproduce the flow's failure (P(no bond) 0.998 vs 0.904
# real, aromatic rings 0.03 vs 1.83). More MLE steps cannot re-weight the objective
# towards the part that is missing; this term is the smallest change that does, and it
# reuses the existing checkpoint instead of retraining.
#
# NOTHING IS OVERWRITTEN: the 50k checkpoints are opened read-only and the runs write to
# outputs/checkpoints/structure_{control,arm} and outputs/samples/structure_*.
#
#   STEPS=3000       fine-tune steps per arm
#   LAMBDA=1.0       structural weight for the arm
#   SAMPLE_T=1.0     temperature of the samples the structural term sees
#   EVAL_T=0.2,0.6,1.0   temperatures evaluated for every model (matched across arms)
#   COUNT=500        samples per (N, QED, T) cell at evaluation
#   SOURCE=outputs/checkpoints/flow_nqed/seed_1/best.pt
#   SKIP_BASELINE=1  skip the untouched-checkpoint sweep (if already measured)
#
#     sbatch slurm/16_structure_experiment.sh
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
    echo "rc: 2=missing input, 3=checkpoint/autoencoder mismatch, 4=bad args," >&2
    echo "    5=non-finite loss (lower LAMBDA or the learning rate), 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 outputs/checkpoints/structure_*/train.log" >&2
    echo "A walltime kill is not a lost run: both fine-tunes resume from last.pt." >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/16_structure_experiment.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
SOURCE="${SOURCE:-outputs/checkpoints/flow_nqed/seed_1/best.pt}"
STEPS="${STEPS:-3000}"
LAMBDA="${LAMBDA:-1.0}"
SAMPLE_T="${SAMPLE_T:-1.0}"
EVAL_T="${EVAL_T:-0.2,0.6,1.0}"
COUNT="${COUNT:-500}"
TARGET_N="${TARGET_N:-15,23,30}"
TARGET_QED="${TARGET_QED:-0.4,0.6,0.8}"
mkdir -p outputs/logs

echo "=============================================="
echo "Stage 16: structure-aligned fine-tune, with its own control"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Source:           ${SOURCE} (READ-ONLY, never modified)"
echo "Steps per arm:    ${STEPS}   lambda (arm) ${LAMBDA}   sample T ${SAMPLE_T}"
echo "Evaluation:       N=${TARGET_N} QED=${TARGET_QED} T=${EVAL_T}, ${COUNT} per cell"
echo "=============================================="
"${PYTHON}" -V
nvidia-smi || { echo "ERROR: no GPU visible to this job" >&2; exit 67; }

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${SOURCE}"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: ${required} not found." >&2
        exit 66
    fi
done
"${PYTHON}" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 'ERROR: no CUDA')"

run_arm () {                       # $1 = name, $2 = lambda
    local name="$1" lambda="$2"
    local out="outputs/checkpoints/structure_${name}"
    echo
    echo "---- fine-tune [${name}] lambda=${lambda} ($(date -u +%H:%M:%SZ)) ----"
    srun "${PYTHON}" -u scripts/17_structure_finetune.py \
        --flow-checkpoint "${SOURCE}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${out}" \
        --lambda-structure "${lambda}" \
        --max-steps "${STEPS}" \
        --batch-size 64 \
        --sample-batch 16 \
        --sample-temperature "${SAMPLE_T}" \
        --learning-rate 1e-4 \
        --val-molecules 2000 \
        --eval-every 250 \
        --save-every 250 \
        --seed 1 \
        --device cuda \
        --resume auto
    echo
    echo "---- evaluate [${name}] (identical settings for every arm) ----"
    srun "${PYTHON}" -u scripts/13_generate_nqed.py \
        --flow-checkpoint "${out}/last.pt" \
        --data-dir "${DATA_DIR}" \
        --output-dir "outputs/samples/structure_${name}" \
        --target-n "${TARGET_N}" \
        --target-qed "${TARGET_QED}" \
        --temperature "${EVAL_T}" \
        --count "${COUNT}" \
        --batch-size 128 \
        --check-molecules 2000 \
        --ceiling-molecules 1000 \
        --device cuda \
        --allow-weak-conditioning
}

# The control runs FIRST: if the structural arm later looks better, the control's numbers
# already exist and cannot be quietly skipped.
run_arm control 0.0
run_arm arm "${LAMBDA}"

if [[ "${SKIP_BASELINE:-0}" != "1" ]]; then
    echo
    echo "---- reference: the UNTOUCHED 50k checkpoint at the same temperatures ----"
    srun "${PYTHON}" -u scripts/13_generate_nqed.py \
        --flow-checkpoint "${SOURCE}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "outputs/samples/structure_baseline_50k" \
        --target-n "${TARGET_N}" \
        --target-qed "${TARGET_QED}" \
        --temperature "${EVAL_T}" \
        --count "${COUNT}" \
        --batch-size 128 \
        --check-molecules 2000 \
        --ceiling-molecules 1000 \
        --device cuda \
        --allow-weak-conditioning
fi

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "THREE comparable evaluations, same decoders, same temperatures, same counts:"
echo "  outputs/samples/structure_baseline_50k/nqed_report.txt   untouched 50k"
echo "  outputs/samples/structure_control/nqed_report.txt        + ${STEPS} MLE steps"
echo "  outputs/samples/structure_arm/nqed_report.txt            + ${STEPS} steps with the"
echo "                                                           structural term"
echo
echo "GO / NO-GO (arm vs CONTROL, at the same temperature):"
echo "  GO if raw validity rises materially above the control AND aromatic rings per"
echo "     molecule move substantially towards the real 1.82 AND fallback falls, with"
echo "     uniqueness/novelty/diversity not collapsing (>90% / >90% / no mode collapse)."
echo "  NO-GO if raw validity and aromaticity are unchanged within noise: the pairwise"
echo "     information is then not reachable from this embedding space, and the next step"
echo "     is a direct atom/bond flow (conditional MoFlow-style), not more tuning here."
echo "  Read the validation NLL in outputs/checkpoints/structure_arm/train.log too: the"
echo "  objective is no longer pure MLE and the likelihood cost must be stated."
echo "=============================================="
