#!/bin/bash
# Arm "prior_frozen". Thin wrapper so each arm is a separate, named submission.
# Identical budget to the other three; the only difference is ARM.
#     sbatch slurm/44c_prior_frozen.sh
export ARM=prior_frozen
exec sbatch "$(dirname "$0")/44_train_arm.sh"
