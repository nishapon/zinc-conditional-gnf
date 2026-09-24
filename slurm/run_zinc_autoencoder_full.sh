#!/bin/bash
#SBATCH --job-name=ZINC_molecular_ae
#SBATCH --output=logs/zinc_autoencoder_output.log
#SBATCH --error=logs/zinc_autoencoder_error.log
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
echo "ZINC molecular autoencoder: full dataset"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u scripts/train_autoencoder.py \
    --config configs/zinc250k.yaml \
    --splits data/processed/zinc250k/zinc_splits.pkl \
    --output-dir outputs/checkpoints/autoencoder \
    --device cuda \
    --epochs 30 \
    --gradient-clip 5.0

echo "=============================================="
echo "ZINC molecular autoencoder training complete"
echo "End time: $(date)"
echo "=============================================="
