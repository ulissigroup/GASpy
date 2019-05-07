''' Tests for the `utils` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we're testing
from ..mongo import (make_doc_from_atoms,
                     _make_atoms_dict,
                     make_spglib_cell_from_atoms,
                     _make_calculator_dict,
                     _make_results_dict,
                     make_atoms_from_doc)

# Things we need to do the tests
import pytest
import os
from collections import OrderedDict
import datetime
import pickle
import numpy as np
import numpy.testing as npt
import ase.io
from . import test_cases

REGRESSION_BASELINES_LOCATION = '/home/GASpy/gaspy/tests/regression_baselines/mongo/'


@pytest.mark.parametrize('bulk_atoms_name', ['Cu_FCC.traj'])
def test_make_doc_from_atoms_concatenation(bulk_atoms_name):
    '''
    We test `make_doc_from_atoms` in two separate tests.
    This test is an integration test to see if the function
    concatenates other subfunctions correctly.

    Note that this is currently hard-coded to test only bulks.
    Feel free to change it if you care enough.
    '''
    atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    doc = make_doc_from_atoms(atoms)
    del doc['user']
    del doc['ctime']
    del doc['mtime']

    atoms_dict = _make_atoms_dict(atoms)
    calculator_dict = _make_calculator_dict(atoms)
    results_dict = _make_results_dict(atoms)
    expected = OrderedDict(atoms=atoms_dict,
                           calc=calculator_dict,
                           results=results_dict)
    assert doc == expected


@pytest.mark.parametrize('bulk_atoms_name', ['Cu_FCC.traj'])
def test_make_doc_from_atoms_population(bulk_atoms_name):
    '''
    We test `make_doc_from_atoms` in two separate tests.
    This test is an integration test to see if the function
    populates subfields correctly, such as the creation time.

    Since the time will not be correct, we simply check that
    whatever is inside there is a datetime object.

    Note that this is currently hard-coded to test only bulks.
    Feel free to change it if you care enough.
    '''
    atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    doc = make_doc_from_atoms(atoms)

    expected_user = os.getenv('USER')
    assert doc['user'] == expected_user

    expected_datetime_type = type(datetime.datetime.utcnow())
    assert type(doc['ctime']) == expected_datetime_type
    assert type(doc['mtime']) == expected_datetime_type


@pytest.mark.baseline
def test_to_create_atoms_dict():
    '''
    Make sure we test at least one relaxed and one unrelaxed
    '''
    for filename in ['bulks/Cu_FCC.traj', 'relaxed/Pt_slab.traj']:
        atoms_name = filename.split('/')[-1]
        atoms = ase.io.read('/home/GASpy/gaspy/tests/test_cases/' + filename)
        atoms_dict = _make_atoms_dict(atoms)

        file_name = REGRESSION_BASELINES_LOCATION + 'atoms_dict_for_' + atoms_name.split('.')[0] + '.pkl'
        with open(file_name, 'wb') as file_handle:
            pickle.dump(atoms_dict, file_handle)
        assert True


def test__make_atoms_dict():
    '''
    Make sure we test at least one relaxed and one unrelaxed
    '''
    for filename in ['bulks/Cu_FCC.traj', 'relaxed/Pt_slab.traj']:
        atoms_name = filename.split('/')[-1]
        atoms = ase.io.read('/home/GASpy/gaspy/tests/test_cases/' + filename)
        atoms_dict = _make_atoms_dict(atoms)

        file_name = REGRESSION_BASELINES_LOCATION + 'atoms_dict_for_' + atoms_name.split('.')[0] + '.pkl'
        with open(file_name, 'rb') as file_handle:
            expected_atoms_dict = pickle.load(file_handle)
        assert atoms_dict == expected_atoms_dict


def test_make_spglib_cell_from_atoms():
    atoms = test_cases.get_bulk_atoms('Cu_FCC.traj')
    lattice, positions, numbers = make_spglib_cell_from_atoms(atoms)

    expected_lattice = np.array(atoms.get_cell().T, dtype='double', order='C')
    expected_positions = np.array(atoms.get_scaled_positions(),
                                  dtype='double', order='C')
    expected_numbers = np.array(atoms.get_atomic_numbers(), dtype='intc')

    npt.assert_allclose(lattice, expected_lattice)
    npt.assert_allclose(positions, expected_positions)
    npt.assert_allclose(numbers, expected_numbers)


@pytest.mark.baseline
@pytest.mark.parametrize('bulk_atoms_name', ['Cu_FCC.traj'])
def test_to_create_calculator_dict(bulk_atoms_name):
    atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    atoms = test_cases.relax_atoms(atoms)
    calculator_dict = _make_calculator_dict(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'calculator_dict_for_' + bulk_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(calculator_dict, file_handle)
    assert True


@pytest.mark.parametrize('bulk_atoms_name', ['Cu_FCC.traj'])
def test__make_calculator_dict(bulk_atoms_name):
    '''
    Note that this is currently hard-coded to test only bulks.
    Feel free to change it if you care enough.
    '''
    atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    atoms = test_cases.relax_atoms(atoms)
    calculator_dict = _make_calculator_dict(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'calculator_dict_for_' + bulk_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_calculator_dict = pickle.load(file_handle)
    assert calculator_dict == expected_calculator_dict


@pytest.mark.baseline
@pytest.mark.parametrize('bulk_atoms_name', ['Cu_FCC.traj'])
def test_to_create_results_dict(bulk_atoms_name):
    atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    atoms = test_cases.relax_atoms(atoms)
    results_dict = _make_results_dict(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'results_dict_for_' + bulk_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(results_dict, file_handle)
    assert True


@pytest.mark.parametrize('bulk_atoms_name', ['Cu_FCC.traj'])
def test__make_results_dict(bulk_atoms_name):
    '''
    Note that this is currently hard-coded to test only bulks.
    Feel free to change it if you care enough.
    '''
    atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    atoms = test_cases.relax_atoms(atoms)
    results_dict = _make_results_dict(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'results_dict_for_' + bulk_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_results_dict = pickle.load(file_handle)
    assert results_dict == expected_results_dict


@pytest.mark.parametrize('bulk_atoms_name', ['Cu_FCC.traj'])
def test_make_atoms_from_doc(bulk_atoms_name):
    '''
    Note that this is currently hard-coded to test only bulks.
    Feel free to change it if you care enough.
    '''
    expected_atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    expected_atoms = test_cases.relax_atoms(expected_atoms)
    doc = make_doc_from_atoms(expected_atoms)
    atoms = make_atoms_from_doc(doc)
    assert atoms == expected_atoms
