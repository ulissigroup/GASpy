#!/bin/sh -l
# Dump the calculations results from the FireWorks database
# into the Auxiliary database. We recommend cronning this to run
# three or four times a day.

# Load the appropriate variables from the .gaspyrc.json file
GASDB_PATH="$(python -c 'from gaspy.utils import read_rc; print(read_rc("gasdb_path"))')"
LUIGI_HOST="$(python -c 'from gaspy.utils import read_rc; print(read_rc("luigi_host"))')"

# Remove the DB dumping token to make sure that we actually dump
rm ${GASDB_PATH}/DumpToAuxDB.token

# Tell Luigi to do the dumping
PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy.tasks UpdateAllDB \
    --max-processes 0 \
    --scheduler-host $LUIGI_HOST \
    --workers=4 \
    --log-level=WARNING \
    --worker-timeout 300
