''' Tests for the `gaspy.tasks.core` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.core import (evaluate_luigi_task,
                           save_luigi_task_run_results,
                           GenerateBulk)

# Things we need to do the tests
import pytest
import pickle
import luigi
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from .__init__ import (get_task_output,
                       clean_up_task)
from ... import defaults
from ...utils import read_rc
from ...mongo import make_atoms_from_doc

# Get the path for the GASdb folder location from the gaspy config file
TASKS_OUTPUTS_LOCATION = read_rc('gasdb_path')


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
    output_file_names = [TASKS_OUTPUTS_LOCATION + '/pickles/' + file_name
                         for file_name in output_file_names]
    expected_outputs = [1, 42, 7, 'We did it!']

    # Run the tasks
    try:
        evaluate_luigi_task(RootTestTask())

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
        __delete_files(output_file_names)


def __delete_files(file_names):
    ''' Helper function to try and delete some files '''
    for file_name in file_names:
        try:
            os.remove(file_name)
        except OSError:
            pass


class RootTestTask(luigi.Task):
    def requires(self):
        return [BranchTestTask(task_result=1),
                BranchTestTask(task_result=7, branch_again=True)]

    def run(self):
        save_luigi_task_run_results(self, 'We did it!')

    def output(self):
        return luigi.LocalTarget(TASKS_OUTPUTS_LOCATION + '/pickles/%s/%s.pkl'
                                 % (type(self).__name__, self.task_id))


class BranchTestTask(luigi.Task):
    task_result = luigi.IntParameter(42)
    branch_again = luigi.BoolParameter(False)

    def requires(self):
        if self.branch_again:
            return BranchTestTask()
        else:
            return

    def run(self):
        save_luigi_task_run_results(self, self.task_result)

    def output(self):
        return luigi.LocalTarget(TASKS_OUTPUTS_LOCATION + '/pickles/%s/%s.pkl'
                                 % (type(self).__name__, self.task_id))


def test_save_luigi_task_run_results():
    '''
    Instead of actually testing this function, we perform a rough
    learning test on Luigi.
    '''
    assert 'temporary_path' in dir(luigi.LocalTarget)


@pytest.mark.parametrize('mpid', ['mp-30', 'mp-867306'])
def test_GenerateBulk(mpid):
    try:
        # Execute the task
        parameters = {'bulk': defaults.bulk_parameters(mpid)}
        task = GenerateBulk(parameters)
        evaluate_luigi_task(task)

        # Fetch and parse the output of the task
        docs = get_task_output(task)
        atoms = make_atoms_from_doc(docs[0])

        # Verify that the task worked by comparing it with Materials Project
        with MPRester(read_rc('matproj_api_key')) as rester:
            structure = rester.get_structure_by_material_id(mpid)
        expected_atoms = AseAtomsAdaptor.get_atoms(structure)
        assert atoms == expected_atoms

    # Clean up
    finally:
        clean_up_task(task)
