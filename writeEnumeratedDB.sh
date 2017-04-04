#!/bin/sh

source ~/fireworks/fireworks_virtualenv/bin/activate
maxindex=2
PYTHONPATH='.' luigi --module adsorptionTargets  EnumerateAlloys --max-index $maxindex --writeDB --scheduler-host gilgamesh.cheme.cmu.edu  --log-level=WARNING --parallel-scheduling --worker-timeout 300  
