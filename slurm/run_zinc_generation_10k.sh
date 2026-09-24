#!/bin/bash
#SBATCH --job-name=ZINC_generation_10k
#SBATCH --output=logs/zinc_generation_10k_output.log
#SBATCH --error=logs/zinc_generation_10k_error.log
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
echo "ZINC conditional generation: 10,000 molecules"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u scripts/generate_molecules.py \
    --config configs/zinc250k.yaml \
    --splits data/processed/zinc250k/zinc_splits.pkl \
    --autoencoder-checkpoint outputs/checkpoints/property_aware_autoencoder/best.pt \
    --flow-checkpoint outputs/checkpoints/conditional_flow/best.pt \
    --embeddings outputs/embeddings/zinc250k \
    --output-dir outputs/samples/zinc250k_seed_42 \
    --device cuda \
    --num-samples 10000 \
    --batch-size 64 \
    --temperature 1.0 \
    --seed 42 \
    --target-tolerance 0.05

echo "=============================================="
echo "ZINC conditional generation complete"
echo "End time: $(date)"
echo "=============================================="
