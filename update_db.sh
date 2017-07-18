#!/bin/sh

cd /global/project/projectdirs/m2755/GASpy

rm /global/cscratch1/sd/zulissi/GASpy_DB/DumpToAuxDB.token
PYTHONPATH='.' luigi \
    --module tasks UpdateAllDB \
    --scheduler-host cori09-bond0.224 \
    --workers=4 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
