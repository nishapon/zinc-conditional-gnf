#!/bin/bash
#SBATCH --job-name=zinc_qedcontrast_activation
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# ---------------------------------------------------------------------------
# ACTIVATION-STRESS VALIDATION (not a 150-step smoke test).
#
# The previous objective's failures appeared ONLY after activation, ~400 steps past step
# 5,000, on a flow that had already drifted. A short smoke test therefore proved nothing.
# This job is built to reproduce that danger window on purpose, on REAL data and the REAL
# frozen 128D autoencoder, in about 25-40 minutes:
#
#   warmup 300, ramp 200  ->  the objective reaches FULL weight at step 500
#   max_steps 3000        ->  then runs ~2,500 further steps AT FULL WEIGHT
#
# It therefore crosses activation on a partially trained flow and sustains full weight for
# five times as many steps as the window in which every previous attempt died.
#
# PASS CRITERIA (all must hold; read them off the log):
#   1. the run reaches step 3000 and summary.json has completed=true
#   2. NO FloatingPointError, and "skipped=0/<n>" in every QED-contrastive line
#   3. nll/node stays finite and does not diverge after step 500
#   4. qed_contrast loss drops below the printed chance value (ln 3 = 1.0986)
#   5. prior_mu_gap grows away from 0 (the conditional prior IS separating across QED)
#   6. val_linear_condition_number_max stays bounded (root-cause diagnostic)
# Record the reported it/s: full-run hours ~= 50000 / it_per_s / 3600.
#
# Writes to a THROWAWAY directory; no baseline or failed-experiment output is touched.
#
#     sbatch slurm/13c_qedcontrast_activation_gpu.sh
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; never train on the login node." >&2
    exit 64
fi

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
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
AE_CHECKPOINT="${AE:-outputs/ablation/ae_degree_128d/seed_3/last.pt}"
OUT="${OUT:-outputs/checkpoints/_activation_qedcontrast_ae128}"
STEPS="${STEPS:-3000}"

echo "=============================================="
echo "QED-contrastive ACTIVATION-STRESS validation (128D, forward-only objective)"
echo "Job:         ${SLURM_JOB_ID}   Node: $(hostname)   Partition: ${SLURM_JOB_PARTITION:-?}"
echo "Autoencoder: ${AE_CHECKPOINT} (frozen, read-only)"
echo "Output:      ${OUT}  (throwaway; baselines untouched)"
echo "Plan:        full weight from step 500, then ~$((STEPS-500)) steps at full weight"
echo "=============================================="
nvidia-smi || { echo "ERROR: no GPU visible" >&2; exit 67; }

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${AE_CHECKPOINT}"; do
    [[ -f "${required}" ]] || { echo "ERROR: ${required} not found." >&2; exit 66; }
done

srun "${PYTHON}" -u scripts/04_train_flow.py \
    --config configs/zinc250k_flow_nqed_ae128_qedcontrast.yaml \
    --data-dir "${DATA_DIR}" \
    --ae-checkpoint "${AE_CHECKPOINT}" \
    --output-dir "${OUT}" \
    --device cuda \
    --seed 3 \
    --max-steps "${STEPS}" \
    --qed-contrast-warmup 300 \
    --qed-contrast-ramp 200 \
    --eval-every 500 \
    --save-every 500 \
    --log-every 50 \
    --resume none

echo
echo "=============================================="
echo "ACTIVATION-STRESS RESULT"
echo "---- throughput (use the last it/s for the runtime estimate) ----"
grep -o "it/s=[0-9.]*" "${OUT}/train.log" | tail -5
echo "---- QED-contrastive behaviour after activation (want: loss < 1.0986, mu_gap growing, skipped=0) ----"
grep "QED-contrastive" "${OUT}/train.log" | tail -12
echo "---- validation, incl. the invertible-linear conditioning diagnostic ----"
grep "VAL step" "${OUT}/train.log" | tail -6
"${PYTHON}" -c "
import json,sys,math
p='${OUT}/metrics.jsonl'
rows=[json.loads(l) for l in open(p)]
tr=[r for r in rows if r['kind']=='train']
va=[r for r in rows if r['kind']=='val']
act=[r for r in tr if r.get('qed_contrast_weight',0)>0]
print()
print('steps logged          :', tr[-1]['step'] if tr else 0)
print('all train losses finite:', all(math.isfinite(r['loss']) for r in tr))
print('applications at full w :', len(act))
if act:
    chance=math.log(3)
    print('contrast loss first/last: %.4f -> %.4f   (chance %.4f)'%(act[0]['qed_contrast_loss'],act[-1]['qed_contrast_loss'],chance))
    print('below chance at end     :', act[-1]['qed_contrast_loss'] < chance)
    print('prior_mu_gap first/last : %.4f -> %.4f'%(act[0]['prior_mu_gap'],act[-1]['prior_mu_gap']))
    print('accuracy first/last     : %.3f -> %.3f'%(act[0]['qed_contrast_accuracy'],act[-1]['qed_contrast_accuracy']))
if va:
    k='val_linear_condition_number_max'
    print('max linear cond number  :', max(r.get(k,float('nan')) for r in va))
    print('final val nll/node      : %.4f'%va[-1]['val_loss'])
" || true
echo
echo "If all six PASS CRITERIA hold, launch the full run:"
echo "  sbatch slurm/13c_flow_nqed_qedcontrast_gpu.sh"
echo "Then remove this throwaway: rm -rf ${OUT}"
echo "=============================================="
