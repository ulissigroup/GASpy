'''
This submodule contains various utilities that can be used while making unit
tests for the tasks_test submodule
'''

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

import shutil
import warnings
import luigi


def clean_up_tasks():
    '''
    As a general practice, we have decided to clear out our task output caches.
    This function does this. Credit to Nick Stinemates and Michael Scott
    Cuthbert on Stack Exchange.
    '''
    folder = '/home/GASpy/gaspy/tests/test_caches/pickles'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def run_task_locally(task):
    '''
    This is similar to the `gaspy.tasks.core.run_tasks` function, but it runs
    one task and it runs it on a local scheduler. You should really only be
    using this for debugging and/or testing purposes.

    Arg:
        task    Instance of a `luigi.Task` object that you want to run
    '''
    # Ignore this silly Luigi warning that they're too lazy to fix
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Parameter '
                                '"task_process_context" with value "None" is not '
                                'of type string.')

        luigi.build([task], local_scheduler=True)
