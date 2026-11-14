#!/bin/bash
# Arm "balanced". Thin wrapper so each arm is a separate, named submission.
# Identical budget to the other three; the only difference is ARM.
#     sbatch slurm/44b_balanced.sh
export ARM=balanced
exec sbatch "$(dirname "$0")/44_train_arm.sh"
