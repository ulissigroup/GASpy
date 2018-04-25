#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --job-name=update_db
#SBATCH --output=update_db-%j.out
#SBATCH --error=update_db-%j.error
#SBATCH --constraint=haswell

# Load GASpy environment and variables
. ~/GASpy/.load_env.sh

# Remove the DB dumping token to make sure that we actually dump
rm ${GASDB_PATH}/DumpToAuxDB.token
# Tell Luigi to do the dumping
PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy.tasks UpdateAllDB \
    --max-processes 0 \
    --scheduler-host $LUIGI_PORT \
    --workers=4 \
    --log-level=WARNING \
    --worker-timeout 300
