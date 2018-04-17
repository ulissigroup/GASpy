#!/bin/sh
# This script will first execute `lpad admin maintain` to clean up some fizzled fireworks.
# It will also delete any broken fireworks and remove duplicate entries

# Load the launchpad and mongo information
source ~/GASpy/.load_env.sh

# Execute maintenance and save any error. If the error happens with `detect_lostruns`,
# then do some clean up
error=$(lpad -l $LPAD_PATH admin maintain 2>&1 >/dev/null)
our_error=$(echo $error | grep "detect_lostruns")

# Clean up
while [[ ! -z "$our_error" ]]; do
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

# Now remove duplicate entries from GASdb
python -c "from gaspy.gasdb import remove_duplicates; remove_duplicates()"
