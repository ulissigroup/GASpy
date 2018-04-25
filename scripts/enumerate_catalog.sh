#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=regular
#SBATCH --job-name=enumerate_catalog
#SBATCH --output=enumerate_catalog-%j.out
#SBATCH --error=enumerate_catalog-%j.error
#SBATCH --constraint=haswell

# Load GASpy environment & variables
. ~/GASpy/.load_env.sh


export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Tell Luigi to do the enumeration

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy.tasks EnumerateAlloys \
    --max-to-submit 3000 \
    --max-index 2 \
    --whitelist '["Pd", "Cu", "Au", "Ag", "Pt", "Rh", "Re", "Ni", "Co", "Ir", "W", "Al", "Ga", "In", "H", "N", "Os", "Fe", "V", "Si", "Sn", "Sb", "Mo", "Mn", "Cr", "Ti", "Zn", "Ge", "As",  "Ru", "Pb", "Nb", "Ca", "Na"]' \
    --scheduler-host $LUIGI_PORT \
    --workers=32 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
