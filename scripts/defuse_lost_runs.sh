#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --partition=regular
#SBATCH --job-name=defuse_lost_runs
#SBATCH --output=defuse_lost_runs-%j.out
#SBATCH --error=defuse_lost_runs-%j.error
#SBATCH --constraint=haswell

# Defuse the lost runs
python -c "from gaspy.fireworks_helper_scripts import defuse_lost_runs; defuse_lost_runs()"
