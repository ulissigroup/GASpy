#!/bin/sh
# This script will first execute `lpad admin maintain` to clean up some fizzled fireworks.
# It will also delete any broken fireworks and remove duplicate entries.
# You should probably run this daily to keep things clean.

# Load the FireWorks information from our .gaspyrc.json file
LPAD_PATH="$(python -c 'from gaspy.utils import read_rc; print(read_rc("lpad_path"))')"
FW_HOST="$(python -c 'from gaspy.utils import read_rc; print(read_rc("lpad.host"))')"
FW_DB="$(python -c 'from gaspy.utils import read_rc; print(read_rc("lpad.name"))')"
FW_PORT="$(python -c 'from gaspy.utils import read_rc; print(read_rc("lpad.port"))')"
FW_USER="$(python -c 'from gaspy.utils import read_rc; print(read_rc("lpad.username"))')"
FW_PW="$(python -c 'from gaspy.utils import read_rc; print(read_rc("lpad.password"))')"

# Execute maintenance and save any error. If the error happens with `detect_lostruns`,
# then do some clean up
error=$(lpad -l $LPAD_PATH admin maintain 2>&1 >/dev/null)
our_error=$(echo $error | grep "detect_lostruns")

# Clean up
while [ ! -z "$our_error" ]; do
    # Find the FWID in question
    fwid=$(lpad -l $LPAD_PATH admin maintain 2> /dev/null | tail -n 1)
    # Delete it
    mongo $FW_DB \
        --host $FW_HOST \
        --port $FW_PORT \
        --username $FW_USER \
        --password $FW_PW \
        --eval "db.launches.remove({'fw_id': $fwid})" && echo "Just removed FW number $fwid"

    # Reset the error query
    error=$(lpad -l $LPAD_PATH admin maintain 2>&1 >/dev/null)
    our_error=$(echo $error | grep "detect_lostruns")
done

# Remove duplicate entries from GASdb
python -c "from gaspy import gasdb; gasdb.remove_duplicates_in_adsorption_collection(); gasdb.remove_duplicates_in_atoms_collection()"

# Defuse any lost runs
python -c "from gaspy.fireworks_helper_scripts import defuse_lost_runs; defuse_lost_runs()"
