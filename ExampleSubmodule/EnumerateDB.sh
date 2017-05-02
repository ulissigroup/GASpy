#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=2:00:00
#SBATCH --partition=regular
#SBATCH --job-name=vasp
#SBATCH --output=enumerate-%j.out
#SBATCH --error=enumerate-%j.error
#SBATCH --constraint=haswell

module load python
source activate /project/projectdirs/m2755/GASpy_conda/

PYTHONPATH='.' luigi --module ExampleTargets EnumerateAlloys --scheduler-host gilgamesh.cheme.cmu.edu --max-index=1 --workers=32 --log-level=WARNING --parallel-scheduling --worker-timeout 300
