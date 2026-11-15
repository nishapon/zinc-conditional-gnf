#!/bin/bash
# Experiment C arm "pairwise". Thin wrapper so each arm is a separate, named submission.
# Identical parent, budget, lr, seed, data and decoder to the other two; only ARM differs.
#     MODE=inspect sbatch slurm/45b_pairwise.sh    # cost estimate, no training
#     MODE=smoke   sbatch slurm/45b_pairwise.sh    # 20 steps, throwaway directory
#     sbatch slurm/45b_pairwise.sh                 # the 3,000-step arm
export ARM=pairwise
exec sbatch "$(dirname "$0")/45_bond_arm.sh"
