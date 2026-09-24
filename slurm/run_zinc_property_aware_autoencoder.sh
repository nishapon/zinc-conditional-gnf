#!/bin/bash
#SBATCH --job-name=ZINC_property_ae
#SBATCH --output=logs/zinc_property_autoencoder_output.log
#SBATCH --error=logs/zinc_property_autoencoder_error.log
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
echo "ZINC property-aware autoencoder training"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u scripts/train_property_aware_autoencoder.py \
    --config configs/zinc250k.yaml \
    --splits data/processed/zinc250k/zinc_splits.pkl \
    --source-checkpoint outputs/checkpoints/autoencoder/best.pt \
    --output-dir outputs/checkpoints/property_aware_autoencoder \
    --device cuda \
    --warmup-epochs 5 \
    --joint-epochs 25 \
    --qed-loss-weight 1.0 \
    --gradient-clip 5.0

echo "=============================================="
echo "ZINC property-aware autoencoder training complete"
echo "End time: $(date)"
echo "=============================================="
