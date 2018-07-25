#!/bin/sh
# This script will use Docker to start an interactive container to run GASpy.
# Note that you need to run this script from the directory that it is in.


# Establish out how to mount GASpy to the container. This is the part
# that assumes that you are running this script inside GASpy.
gaspy_path=$(pwd)
gaspy_mounting_config="$gaspy_path:/home/GASpy"

# Now open the container
shifter --image=ulissigroup/gaspy:dev --volume=$gaspy_mounting_config bash -i
