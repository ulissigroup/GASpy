''' Tests for the `utils` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we're testing
from ..mongo import make_doc_from_atoms, \
    _make_atoms_dict, \
    _make_calculator_dict, \
    _make_results_dict, \
    make_atoms_from_doc

# Things we need to do the tests
import pytest
import os
from collections import OrderedDict
import datetime
from . import test_cases


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


@pytest.mark.parametrize('bulk_atoms_name', ['Cu_FCC.traj'])
def test__make_atoms_dict(bulk_atoms_name):
    '''
    Note that this is currently hard-coded to test only bulks.
    Feel free to change it if you care enough.
    '''
    atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    atoms_dict = _make_atoms_dict(atoms)
    expected = {'atoms': [{'charge': 0.0,
                           'index': 0,
                           'magmom': 0.0,
                           'momentum': [0.0, 0.0, 0.0],
                           'position': [0.0, 0.0, 0.0],
                           'symbol': 'Cu',
                           'tag': 0}],
                'cell': [[0.0, 1.805, 1.805], [1.805, 0.0, 1.805], [1.805, 1.805, 0.0]],
                'chemical_symbols': ['Cu'],
                'constraints': [],
                'info': {},
                'mass': 63.546,
                'natoms': 1,
                'pbc': [True, True, True],
                'spacegroup': 'Fm-3m (225)',
                'symbol_counts': {'Cu': 1},
                'volume': 11.76147025}
    assert atoms_dict == expected


@pytest.mark.parametrize('bulk_atoms_name', ['Cu_FCC.traj'])
def test__make_calculator_dict(bulk_atoms_name):
    '''
    Note that this is currently hard-coded to test only bulks.
    Feel free to change it if you care enough.
    '''
    atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    atoms = test_cases.relax_atoms(atoms)
    calculator_dict = _make_calculator_dict(atoms)
    expected = OrderedDict(calculator={'module': 'ase.calculators.emt', 'class': 'EMT'})
    assert calculator_dict == expected


@pytest.mark.parametrize('bulk_atoms_name', ['Cu_FCC.traj'])
def test__make_results_dict(bulk_atoms_name):
    '''
    Note that this is currently hard-coded to test only bulks.
    Feel free to change it if you care enough.
    '''
    atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    atoms = test_cases.relax_atoms(atoms)
    results_dict = _make_results_dict(atoms)
    expected = OrderedDict(energy=-0.005681511358588409, forces=[[0.0, 0.0, 0.0]], fmax=0.0)
    assert results_dict == expected


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
