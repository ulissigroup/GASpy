#!/bin/sh

module load python

# Get information from the .gaspyrc.json file
export GASDB_PATH="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("gasdb_path"))')"
export CONDA_PATH="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("conda_path"))')"
export LUIGI_PORT="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("luigi_port"))')"
export LPAD_PATH="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad_path"))')"
# GASdb website login info
export GASDB_WEB_USER="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("gasdb_server.username"))')"
export GASDB_WEB_PW="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("gasdb_server.password"))')"
# Mongo info
export FW_HOST="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad.host"))')"
export FW_DB="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad.name"))')"
export FW_PORT="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad.port"))')"
export FW_USER="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad.username"))')"
export FW_PW="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad.password"))')"
export AUX_HOST="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("atoms_client.host"))')"
export AUX_DB="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("atoms_client.database"))')"
export AUX_PORT="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("atoms_client.port"))')"
export AUX_USER="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("atoms_client.user"))')"
export AUX_PW="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("atoms_client.password"))')"

# Load the conda
source activate $CONDA_PATH
