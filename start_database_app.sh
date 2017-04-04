#!/bin/sh


source /home-research/zulissi/virtualenvs/ase_webserver/bin/activate
export ASE_DB_APP_CONFIG='/home-research/zulissi/software/bulk-gas-slab-database/adsorption_app_settings.cfg'
nohup ase-db adsorption_energy_database.db -w &
#nohup ase-db 
#pg://ase:ase@asedb-94.cl2xmu5qpbam.us-east-2.rds.amazonaws.com:5432 -w &


#nohup ase-db adsorption_energy_database.db -w &
