#!/bin/bash
#SBATCH --job-name=zinc_embed_pop
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# THE SHUFFLED-NODE CONTROL. Read-only, CPU only, no training.
#
# Compares three embedding populations over the SAME validation molecules, so the
# molecule sizes are identical everywhere and only the embeddings differ:
#
#   real        the frozen autoencoder's embeddings
#   shuffled    the same node vectors, globally permuted and re-partitioned into
#               molecules of the original sizes. Marginals, covariance and principal
#               components are preserved EXACTLY; only which nodes share a molecule
#               changes. This isolates the inter-node relationship.
#   flow_T*     embeddings sampled from the 300-step smoke checkpoint, same sizes
#
# For each: embedding geometry (norms, spectrum, participation ratio, variance inside
# the real TRAINING eigenbasis, within/between-molecule distances, near-identical
# pairs), the decoder's predictions BEFORE constrained assembly (atom types, charges,
# degrees and their parity/feasibility, bond-logit argmax, P(no bond)), and the
# assembled molecules (QED, QED >= 0.7, total rings, aromatic rings, aromatic/all
# rings, ring bond orders, validity, degree fallback).
#
# Reference distributions come from the TRAINING split; every controlled comparison is
# on VALIDATION molecules. The test split is never loaded. Nothing is modified: the
# autoencoder and the flow checkpoint are opened read-only and their hashes recorded.
#
#   MOLECULES=2000            validation molecules shared by all populations
#   REFERENCE_MOLECULES=20000 training molecules for the reference eigenbasis
#   TEMPERATURES="0.85,1.0"   flow sampling temperatures
#   FLOW_CHECKPOINT=...       default: the 300-step smoke run's best.pt
#
#     sbatch slurm/08_embedding_populations.sh
#
# Never run this file with `bash` directly on the master node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "rc: 2=missing checkpoint/data, 3=autoencoder/flow mismatch, 4=bad args, 1=traceback" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS" >&2
    echo "         tail -50 ${OUT:-outputs/metrics/embedding_populations}/populations.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/08_embedding_populations.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"
AE_CHECKPOINT="${ZINC_GNF_AE:-outputs/ablation/ae_degree/seed_3/last.pt}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-outputs/checkpoints/flow_smoke/best.pt}"
OUT="outputs/metrics/embedding_populations"
MOLECULES="${MOLECULES:-2000}"
REFERENCE_MOLECULES="${REFERENCE_MOLECULES:-20000}"
TEMPERATURES="${TEMPERATURES:-0.85,1.0}"
mkdir -p outputs/logs "${OUT}"

echo "=============================================="
echo "Embedding populations: real vs shuffled vs flow"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown} (CPU on purpose)"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Autoencoder:      ${AE_CHECKPOINT} (read-only, frozen)"
echo "Flow:             ${FLOW_CHECKPOINT} (read-only)"
echo "Molecules:        ${MOLECULES} validation, reference basis from ${REFERENCE_MOLECULES} training"
echo "Output:           ${OUT}"
echo "=============================================="
"${PYTHON}" -V

if [[ ! -f "${DATA_DIR}/splits.pkl" ]]; then
    echo "ERROR: ${DATA_DIR}/splits.pkl not found." >&2
    exit 66
fi
if [[ ! -f "${AE_CHECKPOINT}" ]]; then
    echo "ERROR: frozen autoencoder ${AE_CHECKPOINT} not found." >&2
    exit 66
fi

# The degree-constrained decoder needs scipy's MILP; without it every molecule would
# fall back to greedy assembly and the fallback column would be meaningless.
"${PYTHON}" - <<'PYCHECK'
import sys
import torch, rdkit, numpy, scipy
print("torch", torch.__version__, "| rdkit", rdkit.__version__, "| scipy", scipy.__version__)
try:
    from scipy.optimize import milp
except ImportError as error:
    sys.exit(f"ERROR: scipy.optimize.milp unavailable ({error}); install scipy >= 1.9.")
print("dependency check ok")
PYCHECK

# The shuffled control needs no flow at all, so a missing flow checkpoint degrades to
# "real vs shuffled" instead of failing the job.
FLOW_ARGS=()
if [[ -f "${FLOW_CHECKPOINT}" ]]; then
    FLOW_ARGS=(--flow-checkpoint "${FLOW_CHECKPOINT}" --temperature "${TEMPERATURES}")
    echo "will sample the flow at temperatures ${TEMPERATURES}"
else
    echo "note: ${FLOW_CHECKPOINT} not found; running the shuffled control without flow populations"
fi

echo
echo "---- populations ----"
srun "${PYTHON}" -u scripts/08_embedding_populations.py \
    --ae-checkpoint "${AE_CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --molecules "${MOLECULES}" \
    --reference-molecules "${REFERENCE_MOLECULES}" \
    --batch-size 128 \
    --seed 0 \
    --device cpu \
    "${FLOW_ARGS[@]+${FLOW_ARGS[@]}}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la "${OUT}"
echo "Report:  ${OUT}/embedding_populations.txt"
echo "Data:    ${OUT}/embedding_populations.json"
echo "Log:     ${OUT}/populations.log"
echo
echo "Reading it: if 'shuffled' reproduces the flow's QED, aromaticity and fallback, then"
echo "losing the inter-node relationship is a SUFFICIENT cause. That is not proof that the"
echo "flow does exactly this, and it cannot separate 'not trained enough' from 'cannot be"
echo "learned' - the flow's own embedding statistics in the same report speak to the first,"
echo "and only a longer run with the same measurements can settle the second."
echo "Total rings versus aromatic rings separates a missing ring skeleton from wrong orders."
echo "=============================================="
