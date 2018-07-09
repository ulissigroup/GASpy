#!/bin/bash
# This is a script that is meant to be used to open Jupyter
# while inside of a Docker container.

source activate GASpy_conda
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=''
