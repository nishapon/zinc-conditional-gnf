#!/bin/bash
#SBATCH --job-name=ZINC_flow_smoke
#SBATCH --output=logs/zinc_flow_smoke_output.log
#SBATCH --error=logs/zinc_flow_smoke_error.log
#SBATCH --mail-type=NONE
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

PYTHON=/home/${USER}/miniconda3/envs/zinc_gnf/bin/python
WORKDIR=/home/${USER}/zinc-conditional-gnf

mkdir -p $WORKDIR/logs
cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "ZINC conditional GNF: smoke training"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u scripts/train_flow.py \
    --config configs/zinc250k.yaml \
    --embeddings outputs/embeddings/zinc250k \
    --output-dir outputs/checkpoints/conditional_flow_smoke \
    --device cuda \
    --epochs 2 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --gradient-clip 5.0

echo "=============================================="
echo "ZINC conditional GNF smoke training complete"
echo "End time: $(date)"
echo "=============================================="
