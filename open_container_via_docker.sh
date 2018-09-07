#!/bin/sh
# This script will use Docker to start an interactive container to run GASpy.
# Note that you need to run this script from the directory that it is in.
# It also assumes that you have the ssh keys necessary to access the host
# of your Mongo server.

# Optional input argument. If "jupyter", then open a container to run Jupyter.
jupyter=${1:-0}


# Establish out how to mount your ssh keys to the container. This is the part
# that assumes that you have ssh keys at `~/.ssh/`
ssh_mounting_config="$HOME/.ssh:/home/.ssh"

# Establish out how to mount GASpy to the container. This is the part
# that assumes that you are running this script inside GASpy.
gaspy_path=$(pwd)
gaspy_mounting_config="$gaspy_path:/home/GASpy"

# Create a container from the image
#   -it     run interactively
#   --rm    close the container when we exit
#   -p      connect to the default port used by Jupyter
#   -w      The container directory to start in
#   -v      mount various things to the container
if [ $jupyter = "jupyter" ]; then
    docker run -it --rm -w "/home" \
        -p 8888:8888 \
        -v $ssh_mounting_config \
        -v $gaspy_mounting_config \
        ulissigroup/gaspy:v0.20 \
        jupyter
else
    docker run -it --rm -w "/home" \
        -v $ssh_mounting_config \
        -v $gaspy_mounting_config \
        ulissigroup/gaspy:v0.20 \
        /bin/bash
fi
