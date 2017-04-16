#!/bin/sh

for i in {0..1000}
do
    rm updatedDB.token
    PYTHONPATH='.' luigi --module ExampleTargets UpdateDBs --scheduler-host gilgamesh.cheme.cmu.edu  --workers=1 --log-level=WARNING --parallel-scheduling 
    PYTHONPATH='.' luigi --module ExampleTargets PredictAndSubmit --scheduler-host gilgamesh.cheme.cmu.edu  --workers=16 --log-level=WARNING --parallel-scheduling --worker-timeout 300 
done
