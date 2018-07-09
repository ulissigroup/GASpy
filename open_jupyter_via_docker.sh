#!/bin/sh
# This script will create a Docker container 
# This script will use Docker to start an interactive container to run GASpy
# and then automatically open and pipe Jupyter to the localhost:8888 port.
# Note that you need to run this script from the directory that it is in.

# Figure out how to mount GASpy to the container
gaspy_path=$(pwd)
mounting_config="$gaspy_path:/root/GASpy"

# Create a container from the image
#   -it     run interactively
#   --rm    close the container when we exit
#   -p      connect to the default port used by Jupyter
#   -v      mount GASpy to the container
docker run -it --rm -p 8888:8888 -v $mounting_config ulissigroup/gaspy_jupyter:dev
