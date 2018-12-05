#!/bin/sh
# This script will use Docker to start an interactive container to run GASpy.
# Note that you need to run this script from the directory that it is in.


# Establish out how to mount GASpy to the container. This is the part
# that assumes that you are running this script inside GASpy.
gaspy_path=$(pwd)
gaspy_mounting_config="$gaspy_path:/home/GASpy"

# Create a container from the image
#   -it     run interactively
#   --rm    close the container when we exit
#   -w      The container directory to start in
#   -v      mount various things to the container
docker run -it --rm -w "/home" \
    -v $gaspy_mounting_config \
    -v $HOME/.ssh:/home/.ssh \
    ulissigroup/gaspy:latest \
    /bin/bash
