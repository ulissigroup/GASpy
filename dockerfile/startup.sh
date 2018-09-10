#!/bin/bash
# This is a script that is meant to be run automatically upon
# creation of a Docker container. It automatically connects to your Mongo server.
# It also opens Jupyter if you ask it to.

cd

# Load the FireWorks information from our .gaspyrc.json file
MONGO_HOST="$(python -c 'from gaspy.utils import read_rc; print(read_rc("mongo_info.host_server.host"))')"
MONGO_PORT="$(python -c 'from gaspy.utils import read_rc; print(read_rc("mongo_info.host_server.port"))')"
LOCAL_HOST="$(python -c 'from gaspy.utils import read_rc; print(read_rc("mongo_info.adsorption.host"))')"
LOCAL_PORT="$(python -c 'from gaspy.utils import read_rc; print(read_rc("mongo_info.adsorption.port"))')"
MONGO_HOST_READONLY="$(python -c 'from gaspy.utils import read_rc; print(read_rc("mongo_info.host_server.host"))')"
MONGO_PORT_READONLY="$(python -c 'from gaspy.utils import read_rc; print(read_rc("mongo_info.host_server.port"))')"
LOCAL_HOST_READONLY="$(python -c 'from gaspy.utils import read_rc; print(read_rc("mongo_info.adsorption.host"))')"
LOCAL_PORT_READONLY="$(python -c 'from gaspy.utils import read_rc; print(read_rc("mongo_info.adsorption.port"))')"
USER="$(python -c 'from gaspy.utils import read_rc; print(read_rc("mongo_info.tunneling.username"))')"
HOST="$(python -c 'from gaspy.utils import read_rc; print(read_rc("mongo_info.tunneling.host"))')"

# Tunnel to the Mongo server
ssh -nNT -4 \
    -L $LOCA_HOST:$LOCAL_PORT:$MONGO_HOST:$MONGO_PORT \
    -L $LOCA_HOST_READONLY:$LOCAL_PORT_READONLY:$MONGO_HOST_READONLY:$MONGO_PORT_READONLY \
    $USER@$HOST

# Docker will close the container when it's out of things to do.
# This line will tell the container to do whatever else we tell it to do.
# If this is combined with a `docker run ... /bin/bash`, then Docker
# will re-open a bash terminal for us.
exec "$@";
