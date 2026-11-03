#!/bin/bash
#SBATCH --job-name=zinc_ae_summary
#SBATCH --output=outputs/logs/%x_%j.log
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --mail-user=ponnambalam@uni-hildesheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

# ---------------------------------------------------------------------------
# Collects the per-seed result.json files of the same-budget ablation into one
# paired comparison (mean +- std across seeds, paired differences against the
# baseline arm). Reads small JSON files only. CPU, 1 CPU.
#
#     sbatch slurm/03c_summarize_ablation.sh
#
# Never run this file with `bash` directly on the master node.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not running under SLURM. Submit with sbatch." >&2
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
[[ -n "${PYTHON}" && -x "${PYTHON}" ]] || { echo "ERROR: zinc_gnf python not found" >&2; exit 65; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p outputs/logs outputs/ablation/summary

# The ablation arms live in sibling directories ae_<variant>/seed_<n>. Link them under one
# root (links only; nothing is copied, moved or modified) so the summariser can pair them.
STAGE="outputs/ablation/summary/_stage"
rm -rf "${STAGE}"
mkdir -p "${STAGE}"
for arm in baseline degree noise4; do
    if [[ -d "outputs/ablation/ae_${arm}" ]]; then
        ln -s "${WORKDIR}/outputs/ablation/ae_${arm}" "${STAGE}/${arm}"
    else
        echo "note: outputs/ablation/ae_${arm} not found, that arm is omitted"
    fi
done

srun "${PYTHON}" -u scripts/03c_summarize_ablation.py \
    --root "${STAGE}" \
    --reference baseline \
    --output outputs/ablation/summary/ablation_summary.json

rm -rf "${STAGE}"
echo "Report: outputs/ablation/summary/ablation_summary.json  (and .txt beside it)"
