#!/bin/bash
#SBATCH --job-name=zinc_preprocess
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# ---------------------------------------------------------------------------
# Slice 1: preprocess ZINC250K + produce diagnostic figures.
#
# CPU-only. The ISML cluster's CPU partition is used explicitly; no GPU is
# requested or needed. Dataset must already exist at data/raw/zinc250k.csv
# (compute nodes have no internet; download on a login node first).
#
# Submit with:
#     sbatch slurm/01_preprocess.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON="${ZINC_GNF_PYTHON:-$HOME/miniconda3/envs/zinc_gnf/bin/python}"
WORKDIR="${SLURM_SUBMIT_DIR:-$HOME/zinc-conditional-gnf}"
cd "${WORKDIR}"
# outputs/logs/ is tracked via .gitkeep so it exists after a fresh clone;
# the mkdir below is a defensive no-op in case someone deleted it.
mkdir -p outputs/logs data/processed outputs/figures outputs/metrics/preprocessing

export WANDB_MODE=offline

echo "=============================================="
echo "ZINC250K preprocessing + QED histogram"
echo "Start time:        $(date)"
echo "Node:              $(hostname)"
echo "Partition:         ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs allocated:    ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Python:            ${PYTHON}"
echo "Workdir:           ${WORKDIR}"
echo "=============================================="

# Fail fast if the dataset is missing. Compute nodes have no internet access
# per cluster policy, so the file must be prepared on a login node first.
if [[ ! -f data/raw/zinc250k.csv ]]; then
    echo "ERROR: data/raw/zinc250k.csv not found." >&2
    echo "Compute nodes on the ISML cluster have no internet access." >&2
    echo "Download the ZINC250K CSV on master.ismll.de or mirror.ismll.de first:" >&2
    echo "" >&2
    echo "    mkdir -p data/raw" >&2
    echo "    wget -O data/raw/zinc250k.csv \\" >&2
    echo "      https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/main/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv" >&2
    exit 1
fi

echo
echo "---- Stage 1a: encode + split + fit scaler ----"
srun "${PYTHON}" -u scripts/01_preprocess.py \
    --config configs/zinc250k.yaml \
    --expected-sha256 35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae \
    --no-progress

echo
echo "---- Stage 1b: diagnostic figures ----"
srun "${PYTHON}" -u scripts/01_plot_qed_distribution.py \
    --config configs/zinc250k.yaml

echo "=============================================="
echo "Preprocessing complete"
echo "End time: $(date)"
echo "=============================================="
