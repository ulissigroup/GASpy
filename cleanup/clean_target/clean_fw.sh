#!/bin/sh

# This function should be used to clear "bad" things from our databases.
# Input:
#     $1  The fireworks ID of the item that we want to clear

# Purge the bad firework from the Primary DB
lpad -l /global/project/projectdirs/m2755/zu_vaspsurfaces_files/my_launchpad.yaml defuse_fws -i $1

# Call the python script to purge the Aux DB
python aux_db_clean.py $1

# Purge the Local DB
ase-db /global/cscratch1/sd/zulissi/GASpy_DB/adsorption_energy_database.db fwid=$1 --delete -y
