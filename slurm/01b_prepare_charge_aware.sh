#!/bin/bash
# ---------------------------------------------------------------------------
# Preparation for the charge-aware preprocessing job. Run on the LOGIN node;
# it only creates directories and checks files (no research workload):
#
#     bash slurm/01b_prepare_charge_aware.sh
#
# SLURM does not create the directory of #SBATCH --output/--error, so every
# log and report directory is created BEFORE submission. It also reports the
# free disk space, because the output splits.pkl is about 1.2 GB.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p outputs/logs \
    data/raw \
    data/external/moflow \
    data/processed/zinc250k_charge_aware \
    outputs/metrics/preprocessing_charge_aware

echo "Repository:   ${REPO_DIR}"
echo "Git commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Git branch:   $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Created/verified:"
echo "  outputs/logs"
echo "  data/processed/zinc250k_charge_aware   (separate from the neutral baseline)"
echo "  outputs/metrics/preprocessing_charge_aware"
echo "  data/external/moflow                   (optional MoFlow files)"
echo "Free space here (need ~2 GB):"
df -h "${REPO_DIR}" | tail -1

if [[ -f data/raw/zinc250k.csv ]]; then
    echo "Dataset:      OK  $(ls -l data/raw/zinc250k.csv | awk '{print $5}') bytes"
    echo "              (SHA-256 is verified inside the job)"
else
    echo "Dataset:      MISSING data/raw/zinc250k.csv" >&2
    echo "Download it HERE on the login node (compute nodes have no internet):" >&2
    echo "  wget -O data/raw/zinc250k.csv https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/main/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv" >&2
    exit 1
fi

if [[ -d data/processed/zinc250k ]]; then
    echo "Baseline:     data/processed/zinc250k exists and will NOT be touched"
fi

if [[ -f data/external/moflow/zinc250k.csv && -f data/external/moflow/valid_idx_zinc.json ]]; then
    echo "MoFlow files: present (the job will verify them; split unchanged by default)"
else
    echo "MoFlow files: not present (optional). To enable the verification report, on the login node:"
    echo "  wget -O data/external/moflow/zinc250k.csv https://raw.githubusercontent.com/calvin-zcx/moflow/master/data/zinc250k.csv"
    echo "  wget -O data/external/moflow/valid_idx_zinc.json https://raw.githubusercontent.com/calvin-zcx/moflow/master/data/valid_idx_zinc.json"
fi

echo
echo "Next, from ${REPO_DIR}:"
echo "  sbatch slurm/01b_preprocess_charge_aware.sh"
