#!/bin/sh

# This function should be used to clear "bad" things from our databases.
# Input:
#     $1  The fireworks ID of the item that we want to clear

# Get path information from the .gaspyrc.json file
lpad_path="$(python ../.read_rc.py lpad_path)"

# Purge the bad firework from the Primary DB
lpad -l $lpad_path defuse_fws -i $1

# Call the python script to purge the Aux DB
python target_purge_fr_aux.py $1
