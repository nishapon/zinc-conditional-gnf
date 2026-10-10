#!/bin/bash
#SBATCH --job-name=zinc_charge_prep
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:30:00

# ---------------------------------------------------------------------------
# Slice 1b: charge-aware ZINC250K preprocessing (representation v2).
#
# Encodes all 249,455 molecules with (element, formal charge) atom types,
# audits duplicates and split overlap, fits the condition scaler on the
# training split only, and writes to SEPARATE directories (the neutral-only
# baseline data in data/processed/zinc250k is never touched):
#
#   data/processed/zinc250k_charge_aware/   splits.pkl (~1.2 GB), scaler.json,
#                                           report.json, row_index.csv
#   outputs/metrics/preprocessing_charge_aware/   rejection / duplicate /
#                                           overlap reports, run.log
#
# Resources: validated end to end on the real dataset in a sandbox
# (8m45s serial at ~495 mol/s; peak RSS ~1.5 GB). 1 CPU, 6 GB, 90 min leaves
# wide margin on a slower node. CPU partition only, no GPU, no network.
#
# Submit from the repository root on the login node, after
# slurm/01b_prepare_charge_aware.sh:
#
#     sbatch slurm/01b_preprocess_charge_aware.sh
#
# Optional MoFlow split verification (report only; the split is NOT changed):
# if data/external/moflow/zinc250k.csv and valid_idx_zinc.json exist (copied
# on the login node), they are verified row-for-row against our CSV.
# To USE MoFlow's fixed held-out set as the test split, resubmit with:
#
#     sbatch --export=ALL,SPLIT_PROTOCOL=moflow_fixed slurm/01b_preprocess_charge_aware.sh
#
# Never run this file with `bash` directly on the master node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; never run" >&2
    echo "research workloads on the master node." >&2
    exit 64
fi

# ---- failure diagnostics -------------------------------------------------
on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "time: $(date -u +%Y-%m-%dT%H:%M:%SZ)  host: $(hostname)" >&2
    echo "rc 2=CSV missing, 3=SHA-256 mismatch, 4=bad config/args/MoFlow check," >&2
    echo "   130=interrupted (nothing written), 1=see traceback in .log" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,TotalCPU,MaxRSS,NodeList" >&2
    echo "         seff ${SLURM_JOB_ID}" >&2
    echo "         tail -50 outputs/metrics/preprocessing_charge_aware/run.log" >&2
    echo "==============================================" >&2
}
trap 'on_error $LINENO' ERR

# ---- environment (existing Conda env `zinc_gnf`) -----------------------------
WORKDIR="${SLURM_SUBMIT_DIR:-$HOME/zinc-conditional-gnf}"
cd "${WORKDIR}"

PYTHON="${ZINC_GNF_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    for candidate in \
        "$HOME/miniconda3/envs/zinc_gnf/bin/python" \
        "$HOME/anaconda3/envs/zinc_gnf/bin/python" \
        "$HOME/miniforge3/envs/zinc_gnf/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            PYTHON="${candidate}"
            break
        fi
    done
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "ERROR: could not find the python of the Conda env 'zinc_gnf'." >&2
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/01b_preprocess_charge_aware.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SPLIT_PROTOCOL="${SPLIT_PROTOCOL:-random_80_10_10}"
MOFLOW_CSV="data/external/moflow/zinc250k.csv"
MOFLOW_IDX="data/external/moflow/valid_idx_zinc.json"

mkdir -p outputs/logs data/processed/zinc250k_charge_aware outputs/metrics/preprocessing_charge_aware

echo "=============================================="
echo "Slice 1b: charge-aware preprocessing (v2)"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs allocated:   ${SLURM_CPUS_PER_TASK:-unknown}  (nproc: $(nproc))"
echo "Memory request:   ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "GPU variables:    CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (must be empty)"
echo "Workdir:          ${WORKDIR}"
echo "Python:           ${PYTHON}"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Split protocol:   ${SPLIT_PROTOCOL}"
echo "----------------------------------------------"
"${PYTHON}" -V
df -h "${WORKDIR}" 2>/dev/null || true
echo "=============================================="

# Fail fast: the dataset must already be on disk (no internet on compute nodes).
if [[ ! -f data/raw/zinc250k.csv ]]; then
    echo "ERROR: data/raw/zinc250k.csv not found." >&2
    echo "Compute nodes have no internet access (cluster rule); download on the login node." >&2
    exit 66
fi
ls -la data/raw/zinc250k.csv

"${PYTHON}" - <<'PYCHECK'
import importlib
missing = []
for name in ("rdkit", "numpy", "scipy", "yaml", "zinc_gnf"):
    try:
        importlib.import_module(name)
    except Exception as error:  # noqa: BLE001
        missing.append(f"{name}: {type(error).__name__}: {error}")
if missing:
    raise SystemExit("MISSING DEPENDENCIES in this Conda env:\n  " + "\n  ".join(missing))
import rdkit
print("dependency check ok; rdkit", rdkit.__version__)
PYCHECK

EXTRA_ARGS=()
if [[ -f "${MOFLOW_CSV}" && -f "${MOFLOW_IDX}" ]]; then
    echo "MoFlow files found: verifying row identity (report only unless SPLIT_PROTOCOL=moflow_fixed)"
    EXTRA_ARGS+=(--moflow-csv "${MOFLOW_CSV}" --moflow-valid-idx "${MOFLOW_IDX}")
elif [[ "${SPLIT_PROTOCOL}" == "moflow_fixed" ]]; then
    echo "ERROR: SPLIT_PROTOCOL=moflow_fixed needs ${MOFLOW_CSV} and ${MOFLOW_IDX}." >&2
    echo "Copy them on the login node first (see slurm/README_charge_aware.md)." >&2
    exit 67
else
    echo "MoFlow files not present: skipping MoFlow verification (optional)"
fi

echo
echo "---- charge-aware preprocessing ----"
srun "${PYTHON}" -u scripts/01b_preprocess_charge_aware.py \
    --config configs/zinc250k_charge_aware.yaml \
    --split-protocol "${SPLIT_PROTOCOL}" \
    --progress-every 10000 \
    "${EXTRA_ARGS[@]}"

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la data/processed/zinc250k_charge_aware
echo "Data:     data/processed/zinc250k_charge_aware/"
echo "Reports:  outputs/metrics/preprocessing_charge_aware/"
echo "Baseline: data/processed/zinc250k/ (untouched)"
echo "=============================================="
