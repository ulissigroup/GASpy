#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=regular
#SBATCH --qos=premium
#SBATCH --job-name=enumerate_catalog
#SBATCH --output=enumerate_catalog-%j.out
#SBATCH --error=enumerate_catalog-%j.error
#SBATCH --constraint=haswell
#SBATCH --volume="/global/project/projectdirs/m2755/GASpy_kt:/home/GASpy"
#SBATCH --image=docker:ulissigroup/gaspy:dev 

# Stop numpy/scipy from trying to parallelize over all the cores,
# because we are already parallelizing them ourselves in the Python script.
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

shifter python /home/GASpy/scripts/enumerate_dft_catalog_manually.py
