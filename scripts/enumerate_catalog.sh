#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=regular
#SBATCH --job-name=enumerate_catalog
#SBATCH --output=enumerate_catalog-%j.out
#SBATCH --error=enumerate_catalog-%j.error
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
    --whitelist '["Pd", "Cu", "Au", "Ag", "Pt", "Rh", "Re", "Ni", "Co", "Ir", "W", "Al", "Ga", "In", "H", "N", "Os", "Fe", "V", "Si", "Sn", "Sb", "Mo", "Mn", "Cr", "Ti", "Zn", "Ge", "As", "Se", "Ru", "Pb", "S"]' \
    --scheduler-host $luigi_port \
    --workers=32 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
