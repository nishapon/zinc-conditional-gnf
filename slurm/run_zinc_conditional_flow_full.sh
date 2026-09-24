#!/bin/bash
#SBATCH --job-name=ZINC_conditional_flow
#SBATCH --output=logs/zinc_conditional_flow_output.log
#SBATCH --error=logs/zinc_conditional_flow_error.log
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
echo "ZINC conditional GNF: full training"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u scripts/train_flow.py \
    --config configs/zinc250k.yaml \
    --embeddings outputs/embeddings/zinc250k \
    --output-dir outputs/checkpoints/conditional_flow \
    --device cuda \
    --epochs 40 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --gradient-clip 5.0

echo "=============================================="
echo "ZINC conditional GNF training complete"
echo "End time: $(date)"
echo "=============================================="
