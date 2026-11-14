#!/bin/bash
# Arm "control". Thin wrapper so each arm is a separate, named submission.
# Identical budget to the other three; the only difference is ARM.
#     sbatch slurm/44a_control.sh
export ARM=control
exec sbatch "$(dirname "$0")/44_train_arm.sh"
