#!/bin/sh
# This script reads a .gaspyrc.json file and exports the values to particular keys.
# It is meant to be sourced inside other scripts so that those scripts can
# use the enviroment variables as defined by the .gaspyrc.json file.

# Get information from the .gaspyrc.json file
export LUIGI_PORT="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("luigi_port"))')"
export LPAD_PATH="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad_path"))')"

# GASdb website login info
export GASDB_WEB_USER="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("gasdb_server.username"))')"
export GASDB_WEB_PW="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("gasdb_server.password"))')"

# FireWorks info
export FW_HOST="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad.host"))')"
export FW_DB="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad.name"))')"
export FW_PORT="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad.port"))')"
export FW_USER="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad.username"))')"
export FW_PW="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad.password"))')"

# Auxiliary DB (i.e., MongoDB) info
export AUX_HOST="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.host_server.host"))')"
export AUX_PORT="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.host_server.port"))')"
export LOCAL_AUX_HOST="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.atoms.host"))')"
export LOCAL_AUX_PORT="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.atoms.port"))')"
# Auxiliary DB (i.e., MongoDB) info for read-only service
export AUX_HOST_READONLY="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.host_server_readonly.host"))')"
export AUX_PORT_READONLY="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.host_server_readonly.port"))')"
export LOCAL_AUX_HOST_READONLY="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.catalog_readonly.host"))')"
export LOCAL_AUX_PORT_READONLY="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.catalog_readonly.port"))')"
# Mongo login/access info
export AUX_DB="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.host_server.database"))')"
export AUX_USER="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.host_server.user"))')"
export AUX_PW="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.host_server.password"))')"
export MONGO_TUNNEL_USER="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.tunneling.username"))')"
export MONGO_TUNNEL_HOST="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("mongo_info.tunneling.host"))')"
