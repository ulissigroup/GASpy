#!/bin/sh

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

# Remove the DB dumping token to make sure that we actually dump
rm ${gasdb_path}/DumpToAuxDB.token

# Tell Luigi to do the dumping
PYTHONPATH=$PYTHONPATH luigi \
    --module tasks UpdateAllDB \
    --max-processes 0 \
    --scheduler-host $luigi_port \
    --workers=1 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
