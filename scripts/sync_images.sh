#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=regular
#SBATCH --job-name=sync_images
#SBATCH --output=sync_images-%j.out
#SBATCH --error=sync_images-%j.error
#SBATCH --constraint=haswell

# Sync the pictures
aws s3 sync $SCRATCH/GASpy_DB/images/adsorption s3://catalyst-thumbnails/
aws s3 sync $SCRATCH/GASpy_DB/images/catalog s3://catalyst-thumbnails/
