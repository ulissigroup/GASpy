#!/bin/sh

# Load the .gaspyrc, which has the lpad_loc variable we need
lpad_path="$(python ../.read_rc.py lpad_path)"

echo 'RPBE:'
queued=`lpad -l $lpad_path get_fws -q '{"name.vasp_settings.gga":"RP","state":"READY"}' -d count`
running=`lpad -l $lpad_path get_fws -q '{"name.vasp_settings.gga":"RP","state":"RUNNING"}' -d count`
completed=`lpad -l $lpad_path get_fws -q '{"name.vasp_settings.gga":"RP","state":"COMPLETED"}' -d count`
fizzled=`lpad -l $lpad_path get_fws -q '{"name.vasp_settings.gga":"RP","state":"FIZZLED"}' -d count`
echo Running $running / Queued: $queued  / Completed: $completed / Fizzled: $fizzled

echo 'BEEF:'
queued=`lpad -l $lpad_path get_fws -q '{"name.vasp_settings.gga":"BF","state":"READY"}' -d count`
running=`lpad -l $lpad_path get_fws -q '{"name.vasp_settings.gga":"BF","state":"RUNNING"}' -d count`
completed=`lpad -l $lpad_path get_fws -q '{"name.vasp_settings.gga":"BF","state":"COMPLETED"}' -d count`
echo Running $running / Queued: $queued  / Completed: $completed

squeue -u zulissi,ktran -t R
