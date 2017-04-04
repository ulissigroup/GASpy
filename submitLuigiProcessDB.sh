#!/bin/bash

# set the number of nodes and processes per node
#PBS -l nodes=1:ppn=16
#PBS -l walltime=24:00:00
#PBS -N luigi_enumerate

source ~/fireworks/fireworks_virtualenv/bin/activate
cd $PBS_O_WORKDIR
PYTHONPATH='.' luigi --module generate_database_luigi  Process_DB_calculations  --scheduler-host gilgamesh.cheme.cmu.edu  --workers=16 --log-level=WARNING --parallel-scheduling --worker-timeout 300

