#!/bin/bash
# This is a script that is meant to be run automatically upon
# creation of a Docker container. It automatically connects to your Mongo server.
# It also opens Jupyter if you ask it to.

# Optional input argument. If `1`, then start Jupyter. Otherwise continue as normal.
jupyter="${1:-0}"


# Go home and load GASpy environment and variables
cd
source ~/GASpy/.load_env.sh

# Tunnel to the Mongo server
mkdir -p /home/logs
ssh -nNT -4 \
    -L $LOCAL_AUX_HOST:$LOCAL_AUX_PORT:$AUX_HOST:$AUX_PORT \
    -L $LOCAL_AUX_HOST_READONLY:$LOCAL_AUX_PORT_READONLY:$AUX_HOST_READONLY:$AUX_PORT_READONLY \
    $MONGO_TUNNEL_USER@$MONGO_TUNNEL_HOST >> /home/logs/tunnel.log 2>&1 &

# Open Jupyter
if [ $jupyter = "jupyter" ]; then
    source activate GASpy_conda
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=''
fi

# Docker will close the container when it's out of things to do.
# This line will tell the container to do whatever else we tell it to do.
# If this is combined with a `docker run ... /bin/bash`, then Docker
# will re-open a bash terminal for us.
exec "$@";
