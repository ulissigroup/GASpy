#!/bin/bash -l

#SBATCH -N 1         #Use 2 nodes
#SBATCH -t 24:00:00  #Set 30 minute time limit
#SBATCH -p regular   #Submit to the regular 'partition'
#SBATCH -L SCRATCH   #Job requires $SCRATCH file system
#SBATCH -C haswell   #Use Haswell nodes

PYTHONPATH='../' python dump_images.py 
