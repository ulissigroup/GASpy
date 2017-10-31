#!/bin/sh

module load python
cd /global/project/projectdirs/m2755/GASpy/
source activate /project/projectdirs/m2755/GASpy_conda/

rm /global/cscratch1/sd/zulissi/GASpy_DB/DumpToAuxDB.token
PYTHONPATH='.' luigi \
    --module tasks UpdateAllDB \
    --max-processes 5000 \
    --scheduler-host 128.55.224.51 \
    --workers=10 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
