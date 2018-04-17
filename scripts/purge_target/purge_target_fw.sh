#!/bin/sh -l

# This function should be used to clear "bad" things from our databases.
# Input:
#     $1  The fireworks ID of the item that we want to clear

# Get path information from the .gaspyrc.json file

. ~/GASpy/.load_env.sh

# Purge the bad firework from the Primary DB
lpad -l $LPAD_PATH defuse_fws -i $1

# Call the python script to purge the Aux DB
python target_purge_fr_aux.py $1
