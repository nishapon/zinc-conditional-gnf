#!/bin/bash
#SBATCH --job-name=ZINC_embeddings
#SBATCH --output=logs/zinc_embeddings_output.log
#SBATCH --error=logs/zinc_embeddings_error.log
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
echo "ZINC property-aware embedding extraction"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u scripts/extract_embeddings.py \
    --config configs/zinc250k.yaml \
    --splits data/processed/zinc250k/zinc_splits.pkl \
    --checkpoint outputs/checkpoints/property_aware_autoencoder/best.pt \
    --output-dir outputs/embeddings/zinc250k \
    --device cuda \
    --batch-size 32 \
    --shard-size 2048 \
    --noise-seed 42 \
    --overwrite

echo "=============================================="
echo "ZINC embedding extraction complete"
echo "End time: $(date)"
echo "=============================================="
