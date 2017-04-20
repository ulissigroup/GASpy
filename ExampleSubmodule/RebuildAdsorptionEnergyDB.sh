#!/bin/sh
module load mongodb

#Delete the output files to tell luigi we need to re-add these outputs
rm $SCRATCH/GASpy_DB/pickles/DumpToLocalDB___bulk_____relax_* 

#Delete the final db
rm $SCRATCH/GASpy_DB/adsorption_energy_database.db

#Force a refresh of the auxdb by dropping the database (this regenerates very quickly)
mongo -u admin_zu_vaspsurfaces -p '$TPAHPmj' --host mongodb01.nersc.gov vasp_zu_vaspsurfaces --eval 'db.atoms.drop()' 

#Update the results to rebuild adsorption_energy_database.db
./UpdateResults.sh
