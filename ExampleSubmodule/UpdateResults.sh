#!/bin/sh

# This section populates the database with alloys
# source ~/fireworks/fireworks_virtualenv/bin/activate
# maxindex=2
# PYTHONPATH='.' luigi --module ExampleTargets  EnumerateAlloys --max-index $maxindex --writeDB --scheduler-host gilgamesh.cheme.cmu.edu  --log-level=WARNING --parallel-scheduling --worker-timeout 300  

for i in {0..0}
do
    rm /global/cscratch1/sd/zulissi/GASpy_DB/DumpToAuxDB.token
    PYTHONPATH='.' luigi --module ExampleTargets UpdateDBs --scheduler-host gilgamesh.cheme.cmu.edu  --workers=1 --log-level=WARNING --parallel-scheduling 
    rm /global/cscratch1/sd/zulissi/GASpy_DB/DumpToAuxDB.token
    PYTHONPATH='.' luigi --module ExampleTargets UpdateDBs --scheduler-host gilgamesh.cheme.cmu.edu  --workers=1 --log-level=WARNING --parallel-scheduling 
done
