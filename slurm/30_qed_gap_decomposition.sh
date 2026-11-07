#!/bin/bash
#SBATCH --job-name=zinc_qed_gap_decomp
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

# ---------------------------------------------------------------------------
# READ-ONLY diagnostic. No training, no GPU, no checkpoint is written, no existing
# output directory is touched. The test split is never loaded.
#
# Writes ONLY to $OUT (default outputs/metrics/qed_gap_decomposition), which is new.
#
# Compares the baseline, early and final checkpoints on:
#   A  the exact three-way likelihood-gap decomposition (float64, identity asserted)
#   B  route-separated inverse-sampling geometry at fixed N with common Gaussian eps
#   C  decoder/descriptor coherence, plus a REAL-embedding decoder control
#
# Override any of these:
#   BASELINE=outputs/checkpoints/flow_nqed_ae128/seed_3/best.pt
#   EARLY=outputs/diagnostics/qedcontrast_seed3_early_best.pt
#   FINAL=outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/best.pt
#   AE=outputs/ablation/ae_degree_128d/seed_3/last.pt
#   SAMPLES=32   MOLECULES=512   OUT=...
#
#     sbatch slurm/30_qed_gap_decomposition.sh
#
# A checkpoint that does not exist is skipped with a warning rather than failing the job.
# Never run this on the login node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; do not run on the login node." >&2
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
[[ -n "${PYTHON}" && -x "${PYTHON}" ]] || {
    echo "ERROR: could not find the zinc_gnf env python; resubmit with ZINC_GNF_PYTHON=..." >&2
    exit 65; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
AE="${AE:-outputs/ablation/ae_degree_128d/seed_3/last.pt}"
BASELINE="${BASELINE:-outputs/checkpoints/flow_nqed_ae128/seed_3/best.pt}"
EARLY="${EARLY:-outputs/diagnostics/qedcontrast_seed3_early_best.pt}"
FINAL="${FINAL:-outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/best.pt}"
OUT="${OUT:-outputs/metrics/qed_gap_decomposition}"
SAMPLES="${SAMPLES:-32}"
MOLECULES="${MOLECULES:-512}"

# Autoencoder provenance is CHECKED, not waived. The SHA256 recorded in each flow
# checkpoint must match the autoencoder on disk; set ALLOW_AE_MISMATCH=1 only if you
# intend to override it, and the override is then recorded in the output JSON.
EXTRA_ARGS=""
if [[ "${ALLOW_AE_MISMATCH:-0}" == "1" ]]; then
    echo "WARNING: ALLOW_AE_MISMATCH=1 - autoencoder SHA256 mismatches will be tolerated"
    EXTRA_ARGS="${EXTRA_ARGS} --allow-ae-mismatch"
fi
# The out-of-support 'offset' protocol is OFF by default: it was the cause of the
# job 366591 failure. Set INCLUDE_EXTRAPOLATION=1 to run it as a labelled probe.
if [[ "${INCLUDE_EXTRAPOLATION:-0}" == "1" ]]; then
    echo "including the EXTRAPOLATION offset protocol (explicitly labelled in the report)"
    EXTRA_ARGS="${EXTRA_ARGS} --include-extrapolation"
fi
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "QED likelihood-gap decomposition (READ-ONLY)"
echo "Start:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:       ${SLURM_JOB_ID}   Node: $(hostname)   Partition: ${SLURM_JOB_PARTITION:-?}"
echo "Commit:    $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "AE:        ${AE}"
echo "Output:    ${OUT}  (new directory; nothing else is touched)"
echo "=============================================="

for required in "${DATA_DIR}/splits.pkl" "${DATA_DIR}/scaler.json" "${AE}"; do
    [[ -f "${required}" ]] || { echo "ERROR: ${required} not found." >&2; exit 66; }
done

ARGS=()
for pair in "baseline=${BASELINE}" "early=${EARLY}" "final=${FINAL}"; do
    label="${pair%%=*}"; path="${pair#*=}"
    if [[ -f "${path}" ]]; then
        echo "including ${label}: ${path}"
        ARGS+=(--flow-checkpoint "${label}=${path}")
    else
        echo "WARNING: skipping ${label}: ${path} not found" >&2
    fi
done
[[ ${#ARGS[@]} -gt 0 ]] || { echo "ERROR: no flow checkpoints found." >&2; exit 66; }

srun "${PYTHON}" -u scripts/30_qed_gap_decomposition.py \
    "${ARGS[@]}" \
    --ae-checkpoint "${AE}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --target-n 15,23,30 \
    --qed-values 0.4,0.6,0.8 \
    --decomposition-molecules "${MOLECULES}" \
    --samples-per-cell "${SAMPLES}" \
    --temperature 0.2 \
    --decoders constrained,degree \
    ${EXTRA_ARGS} \
    --device cpu

echo
echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Report:   ${OUT}/qed_gap_decomposition.txt"
echo "Data:     ${OUT}/qed_gap_decomposition.json"
echo
sed -n '1,80p' "${OUT}/qed_gap_decomposition.txt" 2>/dev/null || true
echo "=============================================="
