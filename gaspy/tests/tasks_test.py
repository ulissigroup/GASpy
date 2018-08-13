''' Tests for the `utils` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ..tasks import GenerateBulk

# Things we need to do the tests
import pytest
import pickle
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from gaspy import defaults
from gaspy.utils import evaluate_luigi_task, read_rc
from gaspy.mongo import make_atoms_from_doc

# Get the path for the GASdb folder location from the gaspy config file
TASKS_CACHE_LOCATION = read_rc()['gasdb_path'] + '/pickles/'


@pytest.mark.parametrize('mpid', ['mp-30', 'mp-867306'])
def test_GenerateBulk(mpid):
    try:
        # Execute the task
        parameters = {'bulk': defaults.bulk_parameters(mpid)}
        task = GenerateBulk(parameters)
        evaluate_luigi_task(task)

        # Fetch and parse the output of the task
        docs = _get_task_output(task)
        atoms = make_atoms_from_doc(docs[0])

        # Verify that the task worked by comparing it with Materials Project
        with MPRester(read_rc('matproj_api_key')) as rester:
            structure = rester.get_structure_by_material_id(mpid)
        expected_atoms = AseAtomsAdaptor.get_atoms(structure)
        assert atoms == expected_atoms

    # Clean up
    except:     # noqa: E722
        _clean_up_task(task)
        raise
    _clean_up_task(task)


def _get_task_output(task):
    '''
    We have a standard location where we store task outputs.
    This function will find that location and automatically open it for you.

    Arg:
        task    Instance of a luigi.Task that you want to find the output location for
    Output:
        output  Whatever was saved by the task
    '''
    file_name = _get_task_output_location(task)
    with open(file_name, 'rb') as file_handle:
        output = pickle.load(file_handle)
    return output


def _get_task_output_location(task):
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


def _clean_up_task(task):
    '''
    As a general practice, we have decided to clear out our task output caches.
    This function does this.

    Arg:
        task    Instance of a luigi.Task whose output you want to delete/clean up
    '''
    output_file = _get_task_output_location(task)
    try:
        os.remove(output_file)
    except OSError:
        pass
