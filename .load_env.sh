#!/bin/bash -l

module load python

# Get the path information so we can load things. Thanks to StackExchange/Dave Dopson for this
source="${BASH_SOURCE[0]}"
while [ -h "$source" ]; do # resolve $source until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$source" )" && pwd )"
  source="$(readlink "$source")"
  [[ $source != /* ]] && source="$DIR/$source" # if $source was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
GASPY_PATH="$( cd -P "$( dirname "$source" )" && pwd )"
gaspy_fb_path="$GASPY_PATH/GASpy_feedback"
gaspy_reg_path="$GASPY_PATH/GASpy_regressions"

# Add GASpy to the Python Path, but only if it's not already there
echo $PYTHONPATH | grep -qF "$GASPY_PATH" || PYTHONPATH="$GASPY_PATH:$PYTHONPATH"
echo $PYTHONPATH | grep -qF "$gaspy_fb_path" || PYTHONPATH="$gaspy_fb_path:$PYTHONPATH"
echo $PYTHONPATH | grep -qF "$gaspy_reg_path" || PYTHONPATH="$gaspy_reg_path:$PYTHONPATH"
export PYTHONPATH=$PYTHONPATH

# Get information from the .gaspyrc.json file
export GASDB_PATH="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("gasdb_path"))')"
export CONDA_PATH="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("conda_path"))')"
export LUIGI_PORT="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("luigi_port"))')"
export LPAD_PATH="$(python -c 'import gaspy.readrc; print(gaspy.readrc.read_rc("lpad_path"))')"

# Load the conda
source activate $CONDA_PATH
