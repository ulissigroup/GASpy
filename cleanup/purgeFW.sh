#!/bin/sh
'''
This function should be used to clear "bad" things from our databases.

Input:
    $1  The fireworks ID of the item that we want to clear
'''

# Purge the bad firework from the Primary DB
lpad -l ~/zu_vaspsurfaces_files/my_launchpad.yaml defuse_fws -i $1

# Call the python script to purge the Aux DB
python purgeAuxDB.py $1

# Purge the Local DB
ase-db $SCRATCH/GASpy_DB/adsorption_energy_database.db fwid=$1 --delete -y
