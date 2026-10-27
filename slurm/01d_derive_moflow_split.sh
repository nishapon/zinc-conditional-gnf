#!/bin/bash
#SBATCH --job-name=zinc_moflow_split
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# ---------------------------------------------------------------------------
# Derive the PRIMARY-experiment dataset (MoFlow-fixed split) from the already
# encoded charge-aware records. No re-encoding. CPU partition, no GPU.
#
#   reads : data/processed/zinc250k_charge_aware/        (untouched)
#           data/raw/zinc250k.csv
#           data/external/moflow/zinc250k.csv, valid_idx_zinc.json
#   writes: data/processed/zinc250k_charge_aware_moflow_fixed/   (~1.2 GB)
#           outputs/metrics/split_moflow_fixed/
#
# Sandbox measurements on the real data: 19 s, peak 3.1 GB (8 GB requested for
# margin). Compute nodes have no internet: copy MoFlow's two files on the
# login node first (see slurm/README_ae.md).
#
#     sbatch slurm/01d_derive_moflow_split.sh
#
# Never run this file with `bash` directly on the master node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch; never run" >&2
    echo "research workloads on the master node." >&2
    exit 64
fi

on_error() {
    local rc=$?
    echo "==============================================" >&2
    echo "JOB FAILED  rc=${rc}  line=${1:-?}  command: ${BASH_COMMAND}" >&2
    echo "rc 2=input missing, 4=bad args/output collision, 5=verification failed (nothing written)" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList" >&2
    echo "         tail -50 outputs/metrics/split_moflow_fixed/derive.log" >&2
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
    echo "Resubmit with: sbatch --export=ALL,ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python slurm/01d_derive_moflow_split.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p outputs/logs data/processed/zinc250k_charge_aware_moflow_fixed outputs/metrics/split_moflow_fixed

echo "=============================================="
echo "Derive MoFlow-fixed split (no re-encoding)"
echo "Start time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Job:              ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})"
echo "Node:             $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs / memory:    ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "Workdir:          ${WORKDIR}"
echo "Python:           ${PYTHON}"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "=============================================="
"${PYTHON}" -V

missing=0
for required in data/raw/zinc250k.csv \
                data/processed/zinc250k_charge_aware/splits.pkl \
                data/processed/zinc250k_charge_aware/row_index.csv \
                data/external/moflow/zinc250k.csv \
                data/external/moflow/valid_idx_zinc.json; do
    if [[ -f "${required}" ]]; then
        ls -la "${required}"
    else
        echo "ERROR: missing ${required}" >&2
        missing=1
    fi
done
if [[ "${missing}" -ne 0 ]]; then
    echo "Compute nodes have no internet access. On the LOGIN node run:" >&2
    echo "  wget -O data/external/moflow/zinc250k.csv https://raw.githubusercontent.com/calvin-zcx/moflow/master/data/zinc250k.csv" >&2
    echo "  wget -O data/external/moflow/valid_idx_zinc.json https://raw.githubusercontent.com/calvin-zcx/moflow/master/data/valid_idx_zinc.json" >&2
    echo "The charge-aware random-split dataset comes from slurm/01b_preprocess_charge_aware.sh." >&2
    exit 66
fi

srun "${PYTHON}" -u scripts/01d_derive_moflow_split.py \
    --config configs/zinc250k_charge_aware.yaml \
    --moflow-csv data/external/moflow/zinc250k.csv \
    --moflow-valid-idx data/external/moflow/valid_idx_zinc.json

echo "=============================================="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -la data/processed/zinc250k_charge_aware_moflow_fixed
echo "Primary dataset: data/processed/zinc250k_charge_aware_moflow_fixed/"
echo "Reports:         outputs/metrics/split_moflow_fixed/"
echo "Random-split dataset data/processed/zinc250k_charge_aware/ (untouched)"
echo "=============================================="
