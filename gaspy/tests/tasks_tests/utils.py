'''
This submodule contains various utilities that can be used while making unit
tests for the tasks_test submodule
'''

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Imports necessary to make these utility functions
from ...tasks import make_task_output_location


def clean_up_task(task):
    '''
    As a general practice, we have decided to clear out our task output caches.
    This function does this.

    Arg:
        task    Instance of a luigi.Task whose output you want to delete/clean up
    '''
    output_file = make_task_output_location(task)
    try:
        os.remove(output_file)
    except OSError:
        pass
