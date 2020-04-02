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
                               tile_atoms,
                               find_adsorption_sites,
                               find_bulk_cn_dict,
                               find_surface_atoms_indices,
                               find_adsorption_vector,
                               add_adsorbate_onto_slab,
                               fingerprint_adslab,
                               remove_adsorbate,
                               calculate_unit_slab_height,
                               find_max_movement,
                               get_stoich_from_mpid)

# Things we need to do the tests
import pytest
import inspect
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
import pickle
import numpy as np
import numpy.testing as npt
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from . import test_cases
from .tasks_tests.utils import clean_up_tasks
from .. import defaults
from ..utils import read_rc
from ..tasks import get_task_output, schedule_tasks
from ..gasdb import get_mongo_collection
from ..mongo import make_atoms_from_doc
from ..tasks.atoms_generators import GenerateBulk


REGRESSION_BASELINES_LOCATION = '/home/GASpy/gaspy/tests/regression_baselines/atoms_operators/'
TEST_CASE_LOCATION = '/home/GASpy/gaspy/tests/test_cases/'
SLAB_SETTINGS = defaults.slab_settings()


@pytest.mark.baseline
@pytest.mark.parametrize('mpid, miller_indices',
                         [('mp-30', (1, 1, 1)),
                          ('mp-867306', (1, 1, 1)),
                          ('mp-30', (1, 0, 0)),
                          ('mp-867306', (2, 1, 1))])
def test_to_create_slabs_from_bulk_atoms(mpid, miller_indices):
    bulk_generator = GenerateBulk(mpid)
    try:
        schedule_tasks([bulk_generator], local_scheduler=True)
        doc = get_task_output(bulk_generator)
        atoms = make_atoms_from_doc(doc)
        slabs = make_slabs_from_bulk_atoms(atoms, miller_indices,
                                           SLAB_SETTINGS['slab_generator_settings'],
                                           SLAB_SETTINGS['get_slab_settings'])
    finally:
        clean_up_tasks()

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
        schedule_tasks([bulk_generator], local_scheduler=True)
        doc = get_task_output(bulk_generator)
        atoms = make_atoms_from_doc(doc)
        slabs = make_slabs_from_bulk_atoms(atoms, miller_indices,
                                           SLAB_SETTINGS['slab_generator_settings'],
                                           SLAB_SETTINGS['get_slab_settings'])
    finally:
        clean_up_tasks()

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
    slab_generator_settings = SLAB_SETTINGS['slab_generator_settings'].copy()
    slab_generator_settings['miller_index'] = (2, 1, 1)
    bulk_generator = GenerateBulk('mp-30')
    try:
        schedule_tasks([bulk_generator], local_scheduler=True)
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
        clean_up_tasks()


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

            # Normalization factors to use when assesing distance from surface
            scaled_positions = atoms.get_scaled_positions()
            unit_cell_height = np.linalg.norm(atoms.cell[2])

            # If the slab is pointing upwards...
            if atoms_constrained.cell[2, 2] > 0:
                max_height = max(position[2]
                                 for position, atom in zip(scaled_positions, atoms_constrained)
                                 if atom.tag == 0)
                threshold = max_height - z_cutoff / unit_cell_height
                for i, (position, atom) in enumerate(zip(scaled_positions, atoms)):
                    if atom.tag == 0 and position[2] < threshold:
                        assert i in fixed_atom_indices
                    else:
                        assert i not in fixed_atom_indices

            # If the slab is pointing downwards...
            if atoms_constrained.cell[2, 2] < 0:
                min_height = min(position[2]
                                 for position, atom in zip(scaled_positions, atoms_constrained)
                                 if atom.tag == 0)
                threshold = min_height + z_cutoff / unit_cell_height
                for i, (position, atom) in enumerate(zip(scaled_positions, atoms)):
                    if atom.tag == 0 and position[2] > threshold:
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


@pytest.mark.baseline
def test_to_create_adsorption_sites():
    slab_folder = TEST_CASE_LOCATION + 'slabs/'
    for slab_atoms_name in os.listdir(slab_folder):
        atoms = test_cases.get_slab_atoms(slab_atoms_name)
        sites = find_adsorption_sites(atoms)

        file_name = (REGRESSION_BASELINES_LOCATION + 'sites_for_' +
                     slab_atoms_name.split('.')[0] + '.pkl')
        with open(file_name, 'wb') as file_handle:
            pickle.dump(sites, file_handle)
        assert True


def test_find_adsorption_sites():
    '''
    Check out `.learning_tests.pymatgen_test._get_sites_for_standard_structure`
    to see what pymatgen gives us. Our
    `gaspy.atoms_operators.find_adsorption_sites` simply gives us the value of
    that object when the key is 'all'.
    '''
    slab_folder = TEST_CASE_LOCATION + 'slabs/'
    for slab_atoms_name in os.listdir(slab_folder):
        atoms = test_cases.get_slab_atoms(slab_atoms_name)
        sites = find_adsorption_sites(atoms)

        file_name = (REGRESSION_BASELINES_LOCATION + 'sites_for_' +
                     slab_atoms_name.split('.')[0] + '.pkl')
        with open(file_name, 'rb') as file_handle:
            expected_sites = pickle.load(file_handle)

        npt.assert_allclose(np.array(sites), np.array(expected_sites),
                            rtol=1e-5, atol=1e-7)


def test_find_bulk_cn_dict():
    """
    For the slabs that we use to test adsorption vector
    (in the slab_folder: TEST_CASE_LOCATION + 'slabs/'),
    we use this test function to find out the coordinations
    number in these slabs' bulk form, so we can further
    identify the under-cooridnated atoms in slabs.
    testing for these bulk
    mpids_for_struct = {'Pt12Si5': 'mp-16317',
                        'FeNi': 'mp-2213',
                        'AlAu2Cu': 'mp-867306',
                        'Cu': 'mp-30',
                        'Ni4W': 'mp-30811',
                        'CoSb2': 'mp-755'}
    """
    bulk_atoms_folder = TEST_CASE_LOCATION + 'bulk_atoms_of_slabs/'
    bulk_cn_dicts = {}
    for bulk_atoms_file in os.listdir(bulk_atoms_folder):
        struct = bulk_atoms_file.split('.')[0]
        file_name = bulk_atoms_folder + bulk_atoms_file
        with open(file_name, 'rb') as file_handle:
            bulk_atoms = pickle.load(file_handle)
        bulk_cn = find_bulk_cn_dict(bulk_atoms)
        bulk_cn_dicts[struct] = bulk_cn

    cn_dict_file = REGRESSION_BASELINES_LOCATION + 'bulk_cn_dicts.pkl'
    with open(cn_dict_file, 'rb') as file_handle:
        expected_cn_dict = pickle.load(file_handle)
    assert bulk_cn_dicts == expected_cn_dict


@pytest.mark.baseline
def test_to_create_surface_atoms_indices():
    cn_dict_file = REGRESSION_BASELINES_LOCATION + 'bulk_cn_dicts.pkl'
    with open(cn_dict_file, 'rb') as file_handle:
        bulk_cn_dicts = pickle.load(file_handle)

    slab_folder = TEST_CASE_LOCATION + 'slabs/'
    for slab_atoms_name in os.listdir(slab_folder):
        slab_atoms = test_cases.get_slab_atoms(slab_atoms_name)
        bulk_composition = slab_atoms_name.split('.')[0].split('_')[0]
        cn_dict = bulk_cn_dicts[bulk_composition]
        surface_sites = find_surface_atoms_indices(cn_dict, slab_atoms)

        file_name = (REGRESSION_BASELINES_LOCATION + 'surface_sites_for_' +
                     slab_atoms_name.split('.')[0] + '.pkl')
        with open(file_name, 'wb') as file_handle:
            pickle.dump(surface_sites, file_handle)
        assert True


def test_find_surface_atoms_indices():
    '''
    `gaspy.atoms_operators.find_surface_atoms_indices` gives
    us a list of surface atoms indices.
    '''
    cn_dict_file = REGRESSION_BASELINES_LOCATION + 'bulk_cn_dicts.pkl'
    with open(cn_dict_file, 'rb') as file_handle:
        bulk_cn_dicts = pickle.load(file_handle)

    slab_folder = TEST_CASE_LOCATION + 'slabs/'
    for slab_atoms_name in os.listdir(slab_folder):
        slab_atoms = test_cases.get_slab_atoms(slab_atoms_name)
        bulk_composition = slab_atoms_name.split('.')[0].split('_')[0]
        cn_dict = bulk_cn_dicts[bulk_composition]
        surface_sites = find_surface_atoms_indices(cn_dict, slab_atoms)

        file_name = (REGRESSION_BASELINES_LOCATION + 'surface_sites_for_' +
                     slab_atoms_name.split('.')[0] + '.pkl')
        with open(file_name, 'rb') as file_handle:
            expected_surface_sites = pickle.load(file_handle)
        assert surface_sites == expected_surface_sites


def test_find_adsorption_vector():
    """
    `gaspy.atoms_operators.find_adsorption_vector` gives
    us a (1,3) numpy array.
    """
    # bulk coordination number of test slabs
    cn_dicts_file = REGRESSION_BASELINES_LOCATION + 'bulk_cn_dicts.pkl'
    with open(cn_dicts_file, 'rb') as file_handle:
        bulk_cn_dicts = pickle.load(file_handle)

    slab_folder = TEST_CASE_LOCATION + 'slabs/'
    for slab_atoms_name in os.listdir(slab_folder):
        bulk_composition = slab_atoms_name.split('.')[0].split('_')[0]
        cn_dict = bulk_cn_dicts[bulk_composition]
        atoms = test_cases.get_slab_atoms(slab_atoms_name)
        repeated_slab_atoms = atoms.repeat((2, 2, 1))
        surface_atoms_list = find_surface_atoms_indices(cn_dict, repeated_slab_atoms)

        sites_and_vectors_file = (REGRESSION_BASELINES_LOCATION +
                                  'adsorption_vectors_list_for_' +
                                  slab_atoms_name.split('.')[0] + '.pkl')
        with open(sites_and_vectors_file, 'rb') as file_handle:
            expected_sites_and_vectors = pickle.load(file_handle)

        for site_and_vector in expected_sites_and_vectors:
            adsorption_vector = find_adsorption_vector(cn_dict,
                                                       repeated_slab_atoms,
                                                       surface_atoms_list,
                                                       site_and_vector['site'])
            assert np.array_equal(adsorption_vector, site_and_vector['vector'])


def test_add_adsorbate_onto_slab():
    '''
    Yeah, I know I golfed the crap out of this. I'm sorry. I'm ill and cutting
    corners.
    '''
    slab_folder = TEST_CASE_LOCATION + 'slabs/'
    for slab_atoms_name in os.listdir(slab_folder):
        slab = test_cases.get_slab_atoms(slab_atoms_name)

        for ads_name in ['H', 'CO', 'OH', 'OOH']:
            adsorbate = defaults.adsorbates()[ads_name].copy()

            for site in [np.array([0., 0., 0.]), np.array([1., 1., 1.])]:
                adslab = add_adsorbate_onto_slab(adsorbate, slab, site)

                # Is the adsorbate in the correct position?
                npt.assert_allclose(adslab[0].position, adsorbate[0].position+site)

                # Is the adslab periodic and have the correct cell?
                assert adslab.pbc.tolist() == [True, True, True]
                npt.assert_allclose(adslab.cell, slab.cell)

                # Are the adsorbate and slab atoms even correct?
                npt.assert_array_equal(adslab[0:len(adsorbate)].symbols, adsorbate.symbols)
                assert adslab[-len(slab):] == slab

                # Did the adsorbate constraints carry over?
                assert all(ads_constraint.todict() == adslab_constraint.todict()
                           for ads_constraint, adslab_constraint
                           in zip(adsorbate.constraints, adslab.constraints[:-1]))

                # Is everything tagged correctly?
                tags = adslab.get_tags()
                assert all(tag == 1 for tag in tags[0:len(adsorbate)])
                assert all(tag == 0 for tag in tags[-len(slab):])

                # Are the correct slab atoms fixed?
                fixed_atom_indices = adslab.constraints[-1].get_indices()
                z_cutoff = inspect.signature(constrain_slab).parameters['z_cutoff'].default
                scaled_positions = adslab.get_scaled_positions()
                unit_cell_height = np.linalg.norm(adslab.cell[2])
                # If the slab is pointing upwards...
                if adslab.cell[2, 2] > 0:
                    max_height = max(position[2]
                                     for position, atom in zip(scaled_positions, adslab)
                                     if atom.tag == 0)
                    threshold = max_height - z_cutoff / unit_cell_height
                    for i, (position, atom) in enumerate(zip(scaled_positions, adslab)):
                        if atom.tag == 0 and position[2] < threshold:
                            assert i in fixed_atom_indices
                        else:
                            assert i not in fixed_atom_indices
                # If the slab is pointing downwards...
                if adslab.cell[2, 2] < 0:
                    min_height = min(position[2]
                                     for position, atom in zip(scaled_positions, adslab)
                                     if atom.tag == 0)
                    threshold = min_height + z_cutoff / unit_cell_height
                    for i, (position, atom) in enumerate(zip(scaled_positions, adslab)):
                        if atom.tag == 0 and position[2] > threshold:
                            assert i in fixed_atom_indices
                        else:
                            assert i not in fixed_atom_indices


@pytest.mark.baseline
def test_to_create_adslab_fingerprints():
    adslabs_folder = TEST_CASE_LOCATION + 'adslabs/'
    for file_name in os.listdir(adslabs_folder):
        atoms = ase.io.read(adslabs_folder + file_name)
        fingerprint = fingerprint_adslab(atoms)

        with open(REGRESSION_BASELINES_LOCATION + file_name.split('.')[0] +
                  '_fingerprint.pkl', 'wb') as file_handle:
            pickle.dump(fingerprint, file_handle)
    assert True


def test_fingerprint_adslab():
    adslabs_folder = TEST_CASE_LOCATION + 'adslabs/'
    for file_name in os.listdir(adslabs_folder):
        atoms = ase.io.read(adslabs_folder + file_name)
        fingerprint = fingerprint_adslab(atoms)

        with open(REGRESSION_BASELINES_LOCATION + file_name.split('.')[0] +
                  '_fingerprint.pkl', 'rb') as file_handle:
            expected_fingerprint = pickle.load(file_handle)
        assert fingerprint == expected_fingerprint


def test_remove_adsorbate():
    adslabs_folder = TEST_CASE_LOCATION + 'adslabs/'
    for file_name in os.listdir(adslabs_folder):
        adslab = ase.io.read(adslabs_folder + file_name)
        slab, positions = remove_adsorbate(adslab)

        # Make sure the positions are correct
        for tag, position in positions.items():
            binding_atom_index = np.where(adslab.get_tags() == tag)[0][0]
            expected_position = adslab[binding_atom_index].position
            npt.assert_allclose(position, expected_position)

        # Make sure adsorbates are gone
        for atom in slab:
            assert atom.tag == 0

        # Make sure the slab is still constrained
        assert slab == constrain_slab(slab)


def test_calculate_unit_slab_height():
    '''
    Test all the Miller indices for one bulk
    '''
    # Find all the Miller indices we can use
    atoms = ase.io.read(TEST_CASE_LOCATION + 'bulks/Cu_FCC.traj')
    structure = AseAtomsAdaptor.get_structure(atoms)
    distinct_millers = get_symmetrically_distinct_miller_indices(structure, 3)

    # These are the hard-coded answers
    expected_heights = [6.272210880031006, 6.176446311678471, 4.984647756561888,
                        5.121238738402999, 4.984647756561888, 6.176446311678471,
                        6.272210880031005, 6.147996384691849, 4.839115752287323,
                        4.839115752287323, 6.147996384691849, 6.176446311678471,
                        3.275555302966866, 4.839115752287323, 4.984647756561887,
                        3.4354313844223103, 5.021787742477727, 5.121238738402999,
                        3.275555302966866, 4.839115752287322, 6.14799638469185]

    # Test our function
    for miller_indices, expected_height in zip(distinct_millers, expected_heights):
        height = calculate_unit_slab_height(atoms, miller_indices)
        assert height == expected_height


def test_find_max_movement():
    with get_mongo_collection('atoms') as collection:
        doc = collection.find_one()
    atoms_initial = make_atoms_from_doc(doc['initial_configuration'])
    atoms_final = make_atoms_from_doc(doc)
    max_movement = find_max_movement(atoms_initial, atoms_final)

    # I can't really think of a way to test this function without doing the
    # same exact thing the function did. So let's just "test" that it ran and
    # returned a float.
    assert isinstance(max_movement, float)


def test_get_stoich_from_mpid():
    '''
    Test out three different MPIDs whose stoichiometries we looked up manually
    and hard-coded here.
    '''
    try:
        # Delete the cache first
        cache_dir = read_rc('gasdb_path') + 'mp_stoichs/'
        for file_name in os.listdir(cache_dir):
            if file_name != '.gitignore':
                os.remove(cache_dir + file_name)

        # Try getting the info with no cache
        stoich = get_stoich_from_mpid('mp-30')
        assert stoich == {'Cu': 1}
        stoich = get_stoich_from_mpid('mp-12802')
        assert stoich == {'Al': 1, 'Cu': 3}
        stoich = get_stoich_from_mpid('mp-867306')
        assert stoich == {'Al': 1, 'Cu': 1, 'Au': 2}

        # Make sure the cache exists
        assert len(os.listdir(cache_dir)) >= 4

        # Try getting the info using the cache
        stoich = get_stoich_from_mpid('mp-30')
        assert stoich == {'Cu': 1}
        stoich = get_stoich_from_mpid('mp-12802')
        assert stoich == {'Al': 1, 'Cu': 3}
        stoich = get_stoich_from_mpid('mp-867306')
        assert stoich == {'Al': 1, 'Cu': 1, 'Au': 2}

    # Clean up
    finally:
        for file_name in os.listdir(cache_dir):
            if file_name != '.gitignore':
                os.remove(cache_dir + file_name)
