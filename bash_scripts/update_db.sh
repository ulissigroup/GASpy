#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=2:00:00
#SBATCH --partition=regular
#SBATCH --job-name=vasp
#SBATCH --output=updateDB-%j.out
#SBATCH --error=updateDB-%j.error
#SBATCH --constraint=haswell

module load python
cd /global/project/projectdirs/m2755/GASpy/bash_scripts
source activate /project/projectdirs/m2755/GASpy_conda/

rm /global/cscratch1/sd/zulissi/GASpy_DB/DumpToAuxDB.token
PYTHONPATH='..' luigi \
    --module tasks UpdateAllDB \
    --max-processes 10\
    --scheduler-host 128.55.144.133 \
    --workers=10 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
