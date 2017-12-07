#!/bin/sh

# Go back to home directory, then go to GASpy. Note that this implies that you
# either have a clone of GASpy in your home directory, or at least a link to it
cd
cd GASpy/

# Get path information from the .gaspyrc.json file
export GASPY_PATH="$(python .read_rc.py gaspy_path)"
export GASDB_PATH="$(python .read_rc.py gasdb_path)"
export CONDA_PATH="$(python .read_rc.py conda_path)"
export LUIGI_PORT="$(python .read_rc.py luigi_port)"
export GASPY_FB_PATH="$GASPY_PATH/GASpy_feedback"
export GASPY_REG_PATH="$GASPY_PATH/GASpy_regressions"

# Add GASpy to the Python Path, but only if it's not already there
echo $PYTHONPATH | grep -qF "$gaspy_path" || PYTHONPATH="$gaspy_path:$PYTHONPATH"
echo $PYTHONPATH | grep -qF "$gaspy_fb_path" || PYTHONPATH="$gaspy_fb_path:$PYTHONPATH"
echo $PYTHONPATH | grep -qF "$gaspy_reg_path" || PYTHONPATH="$gaspy_reg_path:$PYTHONPATH"
export PYTHONPATH=$PYTHONPATH

# Load the conda
module load python
source activate $CONDA_PATH
