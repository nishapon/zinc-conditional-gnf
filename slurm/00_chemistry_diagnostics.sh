#!/bin/bash
#SBATCH --job-name=zinc_p0_chem
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# Phase 0b: ZINC250K molecular chemistry diagnostics (read-only on the data).
#
# Counts every one of the 249,455 records, reports why molecules are rejected
# by the existing neutral-only rules, the formal-charge census, charge-typed
# atom frequencies, charged vs neutral counts, CSV-vs-RDKit QED agreement, and
# all-molecules vs accepted-molecules distributions. No molecule is removed,
# the CSV is never modified, and nothing is written to data/processed.
#
# 04:00:00 is a generous ceiling (well under the 24 h limit); run the
# profiler (slurm/00_profile_preprocessing.sh) first and its report suggests
# the time this node actually needs. On SIGTERM (time limit / scancel) the
# script writes chemistry_report.partial.json instead of losing the run.
#
# CPU partition only, no GPU, single CPU, no network. Submit from the
# repository root on the login node (after slurm/00_prepare_phase0.sh):
#
#     sbatch slurm/00_chemistry_diagnostics.sh
#
# Optional quick check of the whole path on 2000 molecules (still via sbatch):
#
#     sbatch --export=ALL,MAX_MOLECULES=2000 --time=00:20:00 \
#            slurm/00_chemistry_diagnostics.sh
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
    echo "rc 2=CSV missing, 3=SHA-256 mismatch, 4=bad config/args," >&2
    echo "   130=stopped early (partial report written), 1=see traceback in .log" >&2
    echo "inspect: sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,TotalCPU,MaxRSS,NodeList" >&2
    echo "         seff ${SLURM_JOB_ID}" >&2
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
    echo "Set ZINC_GNF_PYTHON=/path/to/envs/zinc_gnf/bin/python and resubmit:" >&2
    echo "    sbatch --export=ALL,ZINC_GNF_PYTHON=... slurm/00_chemistry_diagnostics.sh" >&2
    exit 65
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

mkdir -p outputs/logs outputs/diagnostics/chemistry outputs/figures/diagnostics

echo "=============================================="
echo "Phase 0b: ZINC250K chemistry diagnostics"
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
echo "MAX_MOLECULES:    ${MAX_MOLECULES:-all}"
echo "----------------------------------------------"
"${PYTHON}" -V
df -h "${WORKDIR}" 2>/dev/null || true
echo "=============================================="

# Fail fast if the dataset is missing. Compute nodes have no internet access.
if [[ ! -f data/raw/zinc250k.csv ]]; then
    echo "ERROR: data/raw/zinc250k.csv not found." >&2
    echo "Compute nodes have no internet access (cluster rule)." >&2
    echo "On the login node (master.ismll.de or mirror.ismll.de) run:" >&2
    echo "    mkdir -p data/raw" >&2
    echo "    wget -O data/raw/zinc250k.csv https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/main/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv" >&2
    exit 66
fi
ls -la data/raw/zinc250k.csv

"${PYTHON}" - <<'PYCHECK'
import importlib
missing = []
for name in ("rdkit", "numpy", "scipy", "matplotlib", "yaml", "zinc_gnf"):
    try:
        importlib.import_module(name)
    except Exception as error:  # noqa: BLE001
        missing.append(f"{name}: {type(error).__name__}: {error}")
if missing:
    raise SystemExit("MISSING DEPENDENCIES in this Conda env:\n  " + "\n  ".join(missing))
print("dependency check ok")
PYCHECK

EXTRA_ARGS=()
if [[ -n "${MAX_MOLECULES:-}" ]]; then
    EXTRA_ARGS+=(--max-molecules "${MAX_MOLECULES}")
fi

echo
echo "---- chemistry diagnostics ----"
srun "${PYTHON}" -u scripts/00_chemistry_diagnostics.py \
    --config configs/zinc250k.yaml \
    --csv data/raw/zinc250k.csv \
    --expected-sha256 35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae \
    --output-dir outputs/diagnostics/chemistry \
    --figures-dir outputs/figures/diagnostics \
    --progress-every 5000 \
    "${EXTRA_ARGS[@]}"

echo "=============================================="
echo "Diagnostics finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Report:  outputs/diagnostics/chemistry/chemistry_report.json"
echo "Summary: outputs/diagnostics/chemistry/chemistry_summary.txt"
echo "Figures: outputs/figures/diagnostics/*.png"
echo "=============================================="
