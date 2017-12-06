#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --partition=regular
#SBATCH --job-name=defuse_lost_runs
#SBATCH --output=defuse_lost_runs-%j.out
#SBATCH --error=defuse_lost_runs-%j.error
#SBATCH --constraint=haswell

# Go back to home directory, then go to GASpy
cd
cd GASpy/
# Get path information from the .gaspyrc.json file
conda_path="$(python .read_rc.py conda_path)"

# Load the appropriate environment, etc.
module load python
cd scripts/
source activate $conda_path

python defuse_lostruns.py
