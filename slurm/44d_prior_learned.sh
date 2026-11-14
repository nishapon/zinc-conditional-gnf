#!/bin/bash
# Arm "prior_learned". Thin wrapper so each arm is a separate, named submission.
# Identical budget to the other three; the only difference is ARM.
#     sbatch slurm/44d_prior_learned.sh
export ARM=prior_learned
exec sbatch "$(dirname "$0")/44_train_arm.sh"
