''' Tests for the `atoms_operators` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ..atoms_operators import (make_slabs_from_bulk_atoms,
                               orient_atoms_upwards,
                               constrain_slab,
                               is_structure_invertible,
                               flip_atoms,
                               tile_atoms)

# Things we need to do the tests
import os
import pytest
import warnings
import pickle
import numpy as np
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from .tasks_tests.utils import clean_up_task
from .. import defaults
from ..tasks import get_task_output, evaluate_luigi_task
from ..mongo import make_atoms_from_doc
from ..tasks.atoms_generators import GenerateBulk

REGRESSION_BASELINES_LOCATION = '/home/GASpy/gaspy/tests/regression_baselines/atoms_operators/'
TEST_CASE_LOCATION = '/home/GASpy/gaspy/tests/test_cases/'
SLAB_SETTINGS = defaults.SLAB_SETTINGS


@pytest.mark.baseline
@pytest.mark.parametrize('mpid, miller_indices',
                         [('mp-30', (1, 1, 1)),
                          ('mp-867306', (1, 1, 1)),
                          ('mp-30', (1, 0, 0)),
                          ('mp-867306', (2, 1, 1))])
def test_to_create_slabs_from_bulk_atoms(mpid, miller_indices):
    bulk_generator = GenerateBulk(mpid)
    try:
        evaluate_luigi_task(bulk_generator)
        doc = get_task_output(bulk_generator)
        atoms = make_atoms_from_doc(doc)
        slabs = make_slabs_from_bulk_atoms(atoms, miller_indices,
                                           SLAB_SETTINGS['slab_generator_settings'],
                                           SLAB_SETTINGS['get_slab_settings'])
    finally:
        clean_up_task(bulk_generator)

    file_name = (REGRESSION_BASELINES_LOCATION + 'slabs_of_' + mpid + '_' +
                 ''.join(str(index) for index in miller_indices) + '.pkl')
    with open(file_name, 'wb') as file_handle:
        pickle.dump(slabs, file_handle)

    assert True


@pytest.mark.parametrize('mpid, miller_indices',
                         [('mp-30', (1, 1, 1)),
                          ('mp-867306', (1, 1, 1)),
                          ('mp-30', (1, 0, 0)),
                          ('mp-867306', (2, 1, 1))])
def test_make_slabs_from_bulk_atoms(mpid, miller_indices):
    bulk_generator = GenerateBulk(mpid)
    try:
        evaluate_luigi_task(bulk_generator)
        doc = get_task_output(bulk_generator)
        atoms = make_atoms_from_doc(doc)
        slabs = make_slabs_from_bulk_atoms(atoms, miller_indices,
                                           SLAB_SETTINGS['slab_generator_settings'],
                                           SLAB_SETTINGS['get_slab_settings'])
    finally:
        clean_up_task(bulk_generator)

    file_name = (REGRESSION_BASELINES_LOCATION + 'slabs_of_' + mpid + '_' +
                 ''.join(str(index) for index in miller_indices) + '.pkl')
    with open(file_name, 'rb') as file_handle:
        expected_slabs = pickle.load(file_handle)

    for slab, expected_slab in zip(slabs, expected_slabs):
        assert slab == expected_slab


def test_make_slabs_from_bulk_atoms_warning():
    '''
    `miller_index` is an argument for the SlabGenerator class, whose arguments
    we pass implicitly through a dictionary. But we also pass the
    `miller_indices` argument explicitly. We intend for the user to only pass
    the argument in one location (the explicit, `miller_indices` location). If
    they try to use the other one, we should be warning them that we're
    ignoring it.
    '''
    # Prepare to call the function (incorrectly)
    slab_generator_settings = SLAB_SETTINGS['slab_generator_settings']
    slab_generator_settings['miller_index'] = (2, 1, 1)
    bulk_generator = GenerateBulk('mp-30')
    try:
        evaluate_luigi_task(bulk_generator)
        doc = get_task_output(bulk_generator)
        atoms = make_atoms_from_doc(doc)

        # Turn on and record all warnings
        with warnings.catch_warnings(record=True) as warning_manager:
            warnings.simplefilter('always')
            _ = make_slabs_from_bulk_atoms(atoms, (1, 1, 1),    # noqa: F841
                                      slab_generator_settings,
                                      SLAB_SETTINGS['get_slab_settings'])

            # Did we throw the warning, and only that warning?
            assert len(warning_manager) == 1
            assert issubclass(warning_manager[-1].category, SyntaxWarning)
            assert ('will instead use the explicit argument, `miller_indices`'
                    in str(warning_manager[-1].message))

    finally:
        clean_up_task(bulk_generator)


def test_orient_atoms_upwards():
    slabs_folder = TEST_CASE_LOCATION + 'slabs/'
    for file_name in os.listdir(slabs_folder):
        slab = ase.io.read(slabs_folder + file_name)
        oriented_slab = orient_atoms_upwards(slab)

        # Make sure that the z-direction of the cell is going straight upwards
        z_vector = oriented_slab.cell[2]
        assert z_vector[0] == 0
        assert z_vector[1] == 0
        assert z_vector[2] > 0


@pytest.mark.parametrize('z_cutoff', [3., 0.])
def test_constrain_slab(z_cutoff):
    '''
    Assume that the last constraint of the atoms object is the slab constraint,
    and the verify that we fixed the correct atoms
    '''
    # Iterate over both the slab and adslab examples
    for structure_type in ['slabs/', 'adslabs/']:
        test_case_folder = TEST_CASE_LOCATION + structure_type
        for file_name in os.listdir(test_case_folder):
            atoms = ase.io.read(test_case_folder + file_name)

            # Find which atoms that the function fixed
            atoms_constrained = constrain_slab(atoms, z_cutoff)
            fixed_atom_indices = atoms_constrained.constraints[-1].get_indices()

            # If the slab is pointing upwards...
            if atoms_constrained.cell[2, 2] > 0:
                max_height = max(atom.position[2] for atom in atoms_constrained if atom.tag == 0)
                threshold = max_height - z_cutoff
                for i, atom in enumerate(atoms_constrained):
                    if atom.tag == 0 and atom.position[2] < threshold:
                        assert i in fixed_atom_indices
                    else:
                        assert i not in fixed_atom_indices

            # If the slab is pointing downwards...
            if atoms_constrained.cell[2, 2] < 0:
                min_height = min(atom.position[2] for atom in atoms_constrained if atom.tag == 0)
                threshold = min_height + z_cutoff
                for i, atom in enumerate(atoms_constrained):
                    if atom.tag == 0 and atom.position[2] > threshold:
                        assert i in fixed_atom_indices
                    else:
                        assert i not in fixed_atom_indices


def test_is_structure_invertible():
    '''
    Currently, we test this function only on slabs, because that's what we use
    it on mainly. You can test other things if you want.
    '''
    # Get all of our test slabs and convert them to `pymatgen.Structure` ojbects
    slabs_folder = TEST_CASE_LOCATION + 'slabs/'
    for file_name in os.listdir(slabs_folder):
        atoms = ase.io.read(slabs_folder + file_name)
        structure = AseAtomsAdaptor.get_structure(atoms)

        # Check for invertibility manually
        expected_invertibility = False
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        for operation in sga.get_symmetry_operations():
            xform_matrix = operation.affine_matrix
            z_xform = xform_matrix[2, 2]
            if z_xform == -1:
                expected_invertibility = True

        invertibility = is_structure_invertible(structure)
        assert invertibility == expected_invertibility


@pytest.mark.baseline
def test_to_create_flipped_atoms():
    slabs_folder = TEST_CASE_LOCATION + 'slabs/'
    for file_name in os.listdir(slabs_folder):
        slab = ase.io.read(slabs_folder + file_name)
        flipped_atoms = flip_atoms(slab)
        flipped_atoms.write(REGRESSION_BASELINES_LOCATION + 'flipped_' + file_name)
    assert True


def test_flip_atoms():
    '''
    This should probably not be a regression test, but I sure as hell don't
    know how to test it otherwise.
    '''
    slabs_folder = TEST_CASE_LOCATION + 'slabs/'
    for file_name in os.listdir(slabs_folder):
        slab = ase.io.read(slabs_folder + file_name)
        flipped_atoms = flip_atoms(slab)

        expected_atoms = ase.io.read(REGRESSION_BASELINES_LOCATION + 'flipped_' + file_name)
        assert flipped_atoms == expected_atoms


@pytest.mark.parametrize('min_x, min_y', [(4.5, 4.5), (0, 20), (50, 50)])
def test_tile_atoms(min_x, min_y):
    slabs_folder = '/home/GASpy/gaspy/tests/test_cases/slabs/'
    for file_name in os.listdir(slabs_folder):
        atoms = ase.io.read(slabs_folder + file_name)
        atoms_tiled, _ = tile_atoms(atoms, min_x, min_y)
        x_length = np.linalg.norm(atoms_tiled.cell[0])
        y_length = np.linalg.norm(atoms_tiled.cell[1])
        assert x_length >= min_x
        assert y_length >= min_y


#def test_remove_adsorbate():
#    '''
#    This test verifies that anything with a non-zero tag was removed.
#    '''
#    test_case_location = TEST_CASE_LOCATION + 'adslabs/'
#    for file_name in os.listdir(test_case_location):
#        adslab = ase.io.read(test_case_location + file_name)
#
#        slab = remove_adsorbate(adslab)
#        tags = slab.get_tags()
#        assert np.count_nonzero(tags) == 0
