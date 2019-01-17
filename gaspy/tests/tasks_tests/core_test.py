''' Tests for the `gaspy.tasks.core` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.core import (make_task_output_object,
                           make_task_output_location,
                           save_task_output,
                           get_task_output,
                           evaluate_luigi_task)

# Things we need to do the tests
import pickle
import luigi
from .utils import clean_up_tasks
from ...utils import read_rc

# Get the path for the GASdb folder location from the gaspy config file
TASKS_OUTPUTS_LOCATION = read_rc('gasdb_path') + '/pickles/'


def test_make_task_output_object():
    task = RootTestTask()
    target = make_task_output_object(task)

    # Verify that the target is the correct object type
    assert isinstance(target, luigi.LocalTarget)

    # Verify that the path of the target is correct
    assert target.path == make_task_output_location(task)


class RootTestTask(luigi.Task):
    def requires(self):
        return [BranchTestTask(task_result=1),
                BranchTestTask(task_result=7, branch_again=True)]

    def run(self):
        save_task_output(self, 'We did it!')

    def output(self):
        return make_task_output_object(self)


class BranchTestTask(luigi.Task):
    task_result = luigi.IntParameter(42)
    branch_again = luigi.BoolParameter(False)

    def requires(self):
        if self.branch_again:
            return BranchTestTask()
        else:
            return

    def run(self):
        save_task_output(self, self.task_result)

    def output(self):
        return make_task_output_object(self)


def test_make_task_output_location():
    task = RootTestTask()
    file_name = make_task_output_location(task)

    task_name = type(task).__name__
    task_id = task.task_id
    expected_file_name = TASKS_OUTPUTS_LOCATION + '%s/%s.pkl' % (task_name, task_id)
    assert file_name == expected_file_name


def test_save_task_output():
    '''
    Instead of actually testing this function, we perform a rough
    learning test on Luigi.
    '''
    assert 'temporary_path' in dir(luigi.LocalTarget)


def test_get_task_output():
    task = RootTestTask()
    try:
        evaluate_luigi_task(task)
        output = get_task_output(task)

        expected_output = 'We did it!'
        assert output == expected_output

    finally:
        clean_up_tasks()


def test_evaluate_luigi_task():
    '''
    We made some test tasks and try to execute them here. Then we verify
    the output results of the tasks.
    '''
    # Define where/what the outputs should be
    output_file_names = ['BranchTestTask/BranchTestTask_False_1_ca4048d8e6.pkl',
                         'BranchTestTask/BranchTestTask_False_42_fedcdcbd62.pkl',
                         'BranchTestTask/BranchTestTask_True_7_498ea8eed2.pkl',
                         'RootTestTask/RootTestTask__99914b932b.pkl']
    output_file_names = [TASKS_OUTPUTS_LOCATION + file_name
                         for file_name in output_file_names]
    expected_outputs = [1, 42, 7, 'We did it!']

    # Run the tasks
    task = RootTestTask()
    try:
        evaluate_luigi_task(task)

        # Test that each task executed correctly
        for output_file_name, expected_output in zip(output_file_names, expected_outputs):
            with open(output_file_name, 'rb') as file_handle:
                output = pickle.load(file_handle)
            assert output == expected_output

        # Test that when the "force" argument is `False`, tasks ARE NOT rerun
        file_creation_times = [os.path.getmtime(output_file) for output_file in output_file_names]
        evaluate_luigi_task(RootTestTask(), force=False)
        for output_file, expected_ctime in zip(output_file_names, file_creation_times):
            ctime = os.path.getmtime(output_file)
            assert ctime == expected_ctime

        # Test that when the "force" argument is `True`, tasks ARE rerun
        evaluate_luigi_task(RootTestTask(), force=True)
        for output_file, old_ctime in zip(output_file_names, file_creation_times):
            ctime = os.path.getmtime(output_file)
            assert ctime > old_ctime

    # Clean up
    finally:
        clean_up_tasks()
