#!/bin/sh
# This script will use Docker to start an interactive container to run GASpy.
# Note that you need to run this script from the directory that it is in.


# Figure out how to mount GASpy to the container
gaspy_path=$(pwd)
mounting_config="$gaspy_path:/home/GASpy"

# Create a container from the image
#   -it     run interactively
#   --rm    close the container when we exit
#   -v      mount GASpy to the container
docker run -it --rm -v $mounting_config -w "/home" ulissigroup/gaspy:dev
