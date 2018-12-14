'''
This is the submodule that contains the tests for `gaspy.tasks`. The functions
in this __init__ submodule are various utility functions to be used during this
testing.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Imports necessary to make these utility functions
import pickle
from ...utils import read_rc

TASKS_CACHE_LOCATION = read_rc('gasdb_path') + '/pickles/'


def get_task_output(task):
    '''
    We have a standard location where we store task outputs.
    This function will find that location and automatically open it for you.

    Arg:
        task    Instance of a luigi.Task that you want to find the output location for
    Output:
        output  Whatever was saved by the task
    '''
    file_name = get_task_output_location(task)
    with open(file_name, 'rb') as file_handle:
        output = pickle.load(file_handle)
    return output


def get_task_output_location(task):
    '''
    We have a standard location where we store task outputs. This function
    will find that location for you.

    Arg:
        task    Instance of a luigi.Task that you want to find the output location for
    Output:
        file_name   String indication the full path of where the output is
    '''
    task_name = type(task).__name__
    task_id = task.task_id
    file_name = TASKS_CACHE_LOCATION + '%s/%s.pkl' % (task_name, task_id)
    return file_name


def clean_up_task(task):
    '''
    As a general practice, we have decided to clear out our task output caches.
    This function does this.

    Arg:
        task    Instance of a luigi.Task whose output you want to delete/clean up
    '''
    output_file = get_task_output_location(task)
    try:
        os.remove(output_file)
    except OSError:
        pass
