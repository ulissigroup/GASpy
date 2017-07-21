#!/bin/sh
module load python
cd /global/project/projectdirs/m2755/GASpy/
source activate /project/projectdirs/m2755/GASpy_conda/

rm /global/cscratch1/sd/zulissi/GASpy_DB/DumpToAuxDB.token
PYTHONPATH='.' luigi \
    --module tasks UpdateAllDB \
    --max-processes 1000 \
    --scheduler-host 128.55.144.139 \
    --workers=4 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
