#!/bin/sh
cd /global/homes/z/zulissi/m2755/GASpy/ExampleSubmodule
#rm /global/cscratch1/sd/zulissi/GASpy_DB/DumpToAuxDB.token
#PYTHONPATH='.' luigi --module ExampleTargets UpdateDBs --scheduler-host gilgamesh.cheme.cmu.edu  --workers=1 --log-level=WARNING --parallel-scheduling 
PYTHONPATH='.' luigi --module ExampleTargets PredictAndSubmit --Nsubmit=400 --scheduler-host gilgamesh.cheme.cmu.edu  --workers=16 --log-level=WARNING --parallel-scheduling --worker-timeout 300
