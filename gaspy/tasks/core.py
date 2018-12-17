'''
This module houses various utility functions that we can use when working with
Luigi tasks
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import os
from collections import Iterable
import pickle
import luigi
from .. import utils
from .. import fireworks_helper_scripts as fwhs

GASDB_PATH = utils.read_rc('gasdb_path')


def evaluate_luigi_task(task, force=False):
    '''
    This follows luigi logic to evaluate a task by recursively evaluating all
    requirements. This is useful for executing tasks that are typically
    independent of other tasks, e.g., populating a catalog of sites.

    Arg:
        task    Class instance of a luigi task
        force   A boolean indicating whether or not you want to forcibly
                evaluate the task and all the upstream requirements.
                Useful for re-doing tasks that you know have already been
                completed.
    '''
    # Don't do anything if it's already done and we're not redoing
    if task.complete() and not(force):
        return

    else:
        # Execute prerequisite task[s] recursively
        requirements = task.requires()
        if requirements:
            if isinstance(requirements, Iterable):
                for req in requirements:
                    if not(req.complete()) or force:
                        evaluate_luigi_task(req, force)
            else:
                if not(requirements.complete()) or force:
                    evaluate_luigi_task(requirements, force)

        # Luigi will yell at us if we try to overwrite output files.
        # So if we're foricbly redoing tasks, we need to delete the old outputs.
        if force:
            os.remove(task.output().fn)

        # After prerequisites are done, run the task
        task.run()


def save_luigi_task_run_results(task, output):
    '''
    This function is a light wrapper to save a luigi task's output. Instead of
    writing the output directly onto the output file, we write onto a temporary
    file and then atomically move the temporary file onto the output file.

    This defends against situations where we may have accidentally queued
    multiple instances of a task; if this happens and both tasks try to write
    to the same file, then the file gets corrupted. But if both of these tasks
    simply write to separate files and then each perform an atomic move, then
    the final output file remains uncorrupted.

    Doing this for more or less every single task in GASpy gots annoying, so
    we wrapped it.

    Args:
        task    Instance of a luigi task whose output you want to write to
        output  Whatever object that you want to save
    '''
    with task.output().temporary_path() as task.temp_output_path:
        with open(task.temp_output_path, 'wb') as file_handle:
            pickle.dump(output, file_handle)


class DumpFWToTraj(luigi.Task):
    '''
    Given a FWID, this task will dump a traj file into GASdb/FW_structures for viewing/debugging
    purposes
    '''
    fwid = luigi.IntParameter()

    def run(self):
        lpad = fwhs.get_launchpad()
        fw = lpad.get_fw_by_id(self.fwid)
        atoms_trajhex = fw.launches[-1].action.stored_data['opt_results'][1]

        # Write a blank token file to indicate this was done so that the entry is not written again
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(utils.decode_trajhex_to_atoms(atoms_trajhex))

    def output(self):
        return luigi.LocalTarget(GASDB_PATH+'/FW_structures/%s.traj' % (self.fwid))
