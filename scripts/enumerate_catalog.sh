#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=2:00:00
#SBATCH --partition=regular
#SBATCH --job-name=vasp
#SBATCH --output=enumerate-%j.out
#SBATCH --error=enumerate-%j.error
#SBATCH --constraint=haswell

# Go back to home directory, then go to GASpy
cd
cd GASpy/
# Get path information from the .gaspyrc.json file
conda_path="$(python .read_rc.py conda_path)"
luigi_port="$(python .read_rc.py luigi_port)"

# Load the appropriate environment, etc.
module load python
cd gaspy/
source activate $conda_path

# Tell Luigi to do the enumeration
PYTHONPATH=$PYTHONPATH luigi \
    --module tasks EnumerateAlloys \
    --max-index 2 \
    --whitelist '["Mo", "Mn", "Cr", "Ti", "Zn", "Ge", "As", "Se", "Ru", "Pb", "S"]' \
    --scheduler-host $luigi_port \
    --workers=32 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
