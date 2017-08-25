#!/bin/sh

# This function should be used to look at traj files that are in the Primary FWs DB.
for fwid in 515211 53987
do

# Find the directory of where the traj file is
dir=$(lpad -l /global/project/projectdirs/m2755/zu_vaspsurfaces_files/my_launchpad.yaml \
          get_fws -i $fwid -d more \
          | grep dir | sed -e 's/"launch_dir": "\(.*\)",/\1/')
# Fix the spacing
dir=$(echo ${dir})

# Define the server to pull from as per the structure of the directory that the
# launchpad returned
if [ "$dir" = *"home-research"* ]; then
    server='@gilgamesh.cheme.cmu.edu:'
elif [ "$dir" = *"home/z"* ]; then
    server='@cori.nersc.gov:'
else
    server='@arjuna.psc.edu:'
fi
# Pull the file and then view it
rsync -z $(whoami)${server}${dir}/all.traj ./$fwid.traj
#ase gui ./$fwid.traj

done
