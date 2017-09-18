#!/bin/sh
echo 'RPBE:'
lploc=/project/projectdirs/m2755/zu_vaspsurfaces_files/my_launchpad.yaml
queued=`lpad -l $lploc get_fws -q '{"name.vasp_settings.gga":"RP","state":"READY"}' -d count`
running=`lpad -l $lploc get_fws -q '{"name.vasp_settings.gga":"RP","state":"RUNNING"}' -d count`
completed=`lpad -l $lploc get_fws -q '{"name.vasp_settings.gga":"RP","state":"COMPLETED"}' -d count`
fizzled=`lpad -l $lploc get_fws -q '{"name.vasp_settings.gga":"RP","state":"FIZZLED"}' -d count`
echo Running $running / Queued: $queued  / Completed: $completed / Fizzled: $fizzled
echo 'BEEF:'
queued=`lpad -l $lploc get_fws -q '{"name.vasp_settings.gga":"BF","state":"READY"}' -d count`
running=`lpad -l $lploc get_fws -q '{"name.vasp_settings.gga":"BF","state":"RUNNING"}' -d count`
completed=`lpad -l $lploc get_fws -q '{"name.vasp_settings.gga":"BF","state":"COMPLETED"}' -d count`
echo Running $running / Queued: $queued  / Completed: $completed

squeue -u zulissi,ktran -t R
