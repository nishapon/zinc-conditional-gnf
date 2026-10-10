#!/bin/bash
# ---------------------------------------------------------------------------
# Phase 0 preparation. Run on the LOGIN node from anywhere (it is only
# `mkdir` and file checks, no research workload):
#
#     bash slurm/00_prepare_phase0.sh
#
# SLURM does not create the directory of #SBATCH --output/--error, so a job
# whose log directory is missing fails without leaving any log. This script
# creates every log and report directory BEFORE submission and verifies the
# dataset file exists (compute nodes have no internet access).
#
# It deliberately does not hash the 24 MB CSV or import RDKit; the jobs
# verify the SHA-256 themselves on a compute node.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p outputs/logs \
    outputs/diagnostics/chemistry \
    outputs/diagnostics/profiling/scratch_shared \
    outputs/figures/diagnostics \
    data/raw

echo "Repository:   ${REPO_DIR}"
echo "Git commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Git branch:   $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Created/verified:"
echo "  outputs/logs"
echo "  outputs/diagnostics/chemistry"
echo "  outputs/diagnostics/profiling"
echo "  outputs/figures/diagnostics"

if [[ -f data/raw/zinc250k.csv ]]; then
    echo "Dataset:      OK  $(ls -l data/raw/zinc250k.csv | awk '{print $5}') bytes"
    echo "              (SHA-256 is verified inside the job; expected"
    echo "               35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae)"
else
    echo "Dataset:      MISSING data/raw/zinc250k.csv" >&2
    echo "Download it HERE on the login node (compute nodes have no internet):" >&2
    echo "  wget -O data/raw/zinc250k.csv https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/main/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv" >&2
    exit 1
fi

echo
echo "Next, from ${REPO_DIR}:"
echo "  sbatch slurm/00_profile_preprocessing.sh"
echo "  sbatch slurm/00_chemistry_diagnostics.sh"
