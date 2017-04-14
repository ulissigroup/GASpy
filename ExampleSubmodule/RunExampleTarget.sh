#!/bin/sh

# This section populates the database with alloys
# source ~/fireworks/fireworks_virtualenv/bin/activate
# maxindex=2
# PYTHONPATH='.' luigi --module ExampleTargets  EnumerateAlloys --max-index $maxindex --writeDB --scheduler-host gilgamesh.cheme.cmu.edu  --log-level=WARNING --parallel-scheduling --worker-timeout 300  

for i in {0..1000}
do
    rm updatedDB.token
    PYTHONPATH='.' luigi --module ExampleTargets InitializeDB --scheduler-host gilgamesh.cheme.cmu.edu  --workers=1 --log-level=WARNING --parallel-scheduling 
    PYTHONPATH='.' luigi --module ExampleTargets UpdateDFTAdsorptionEnergies --scheduler-host gilgamesh.cheme.cmu.edu  --workers=1 --log-level=WARNING 	--parallel-scheduling 
    #PYTHONPATH='.' luigi --module ExampleTargets StudyCoordinationSites --scheduler-host gilgamesh.cheme.cmu.edu  --workers=16 --log-level=WARNING --parallel-scheduling --worker-timeout 300 
done
