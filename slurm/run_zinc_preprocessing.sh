#!/bin/bash
#SBATCH --job-name=ZINC_preprocess
#SBATCH --output=logs/zinc_preprocess_output.log
#SBATCH --error=logs/zinc_preprocess_error.log
#SBATCH --mail-user=nishauniupdates@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

PYTHON=/home/${USER}/miniconda3/envs/zinc_gnf/bin/python
WORKDIR=/home/${USER}/zinc-conditional-gnf

mkdir -p $WORKDIR/logs
cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "ZINC250K preprocessing"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u scripts/preprocess_zinc.py \
    --csv data/raw/zinc250k.csv \
    --output-dir data/processed/zinc250k \
    --target-molecules 0 \
    --candidate-count 0 \
    --seed 42 \
    --max-nodes 38 \
    --train-fraction 0.80 \
    --validation-fraction 0.10

echo "=============================================="
echo "ZINC250K preprocessing complete"
echo "End time: $(date)"
echo "=============================================="
