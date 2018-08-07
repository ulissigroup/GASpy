''' Test the pymatgen functionalities that we use '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we'll be testing
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

# Things we need to do the testing
import pytest
import pickle
import numpy as np
import numpy.testing as npt
from .. import test_cases

REGRESSION_BASELINES_LOCATION = '/home/GASpy/gaspy/tests/regression_baselines/learning_tests/pymatgen/'


@pytest.mark.parametrize('bulk_atoms_name,bulk_structure_name',
                         [('Cu_FCC.traj', 'Cu_FCC_structure.pkl')])
def test_AseAtomsAdaptor_get_atoms(bulk_atoms_name, bulk_structure_name):
    '''
    This is currently hard-coded to test only bulks. If you care enough,
    you can update to check slabs and/or adslabs, too.
    '''
    with open(REGRESSION_BASELINES_LOCATION + bulk_structure_name, 'rb') as file_handle:
        structure = pickle.load(file_handle)
    atoms = AseAtomsAdaptor.get_atoms(structure)

    expected_atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    assert atoms == expected_atoms


@pytest.mark.parametrize('bulk_atoms_name,bulk_structure_name',
                         [('Cu_FCC.traj', 'Cu_FCC_structure.pkl')])
def test_AseAtomsAdaptor_get_structure(bulk_atoms_name, bulk_structure_name):
    '''
    This is currently hard-coded to test only bulks. If you care enough,
    you can update to check slabs and/or adslabs, too.
    '''
    atoms = test_cases.get_bulk_atoms(bulk_atoms_name)
    structure = AseAtomsAdaptor.get_structure(atoms)

    with open(REGRESSION_BASELINES_LOCATION + bulk_structure_name, 'rb') as file_handle:
        expected_structure = pickle.load(file_handle)
    assert structure == expected_structure


@pytest.mark.baseline
@pytest.mark.parametrize('slab_name',
                         ['AlAu2Cu_210.traj',
                          'CoSb2_110.traj',
                          'Cu_211.traj',
                          'FeNi_001.traj',
                          'Ni4W_001.traj',
                          'Pt12Si5_110.traj'])
def test_to_create_adsorption_sites(slab_name):
    structure = test_cases.get_slab_structure(slab_name)
    sites_dict = AdsorbateSiteFinder(structure).find_adsorption_sites(put_inside=True)

    slab_name_no_extension = slab_name.split('.')[0]
    file_name = REGRESSION_BASELINES_LOCATION + 'adsorption_sites_for_' + slab_name_no_extension + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(sites_dict, file_handle)

    assert True


@pytest.mark.parametrize('slab_name',
                         ['AlAu2Cu_210.traj',
                          'CoSb2_110.traj',
                          'Cu_211.traj',
                          'FeNi_001.traj',
                          'Ni4W_001.traj',
                          'Pt12Si5_110.traj'])
def test_AdsorbateSiteFinder_find_adsorption_site_types(slab_name):
    ''' Verify that ASF finds the same exact site types (e.g., ontop, bride, hollow) '''
    # Use pymatgen to find the sites
    structure = test_cases.get_slab_structure(slab_name)
    sites_dict = AdsorbateSiteFinder(structure).find_adsorption_sites(put_inside=True)
    site_types = sites_dict.keys()

    # Load the baseline sites
    slab_name_no_extension = slab_name.split('.')[0]
    file_name = REGRESSION_BASELINES_LOCATION + 'adsorption_sites_for_' + slab_name_no_extension + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_sites_dict = pickle.load(file_handle)
    expected_site_types = expected_sites_dict.keys()

    assert site_types == expected_site_types


@pytest.mark.parametrize('slab_name',
                         ['AlAu2Cu_210.traj',
                          'CoSb2_110.traj',
                          'Cu_211.traj',
                          'FeNi_001.traj',
                          'Ni4W_001.traj',
                          'Pt12Si5_110.traj'])
def test_AdsorbateSiteFinder_find_adsorption_site_locations(slab_name):
    ''' Verify that ASF finds the same cartesion site locations for each site type '''
    # Use pymatgen to find the sites
    structure = test_cases.get_slab_structure(slab_name)
    sites_dict = AdsorbateSiteFinder(structure).find_adsorption_sites(put_inside=True)

    # Load the baseline sites
    slab_name_no_extension = slab_name.split('.')[0]
    file_name = REGRESSION_BASELINES_LOCATION + 'adsorption_sites_for_' + slab_name_no_extension + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_sites_dict = pickle.load(file_handle)

    # The output we're checking is a dictionary of lists of numpy arrays.
    # Let's check convert each list to an array and then check each array one at a time.
    for site_type, expected_sites in expected_sites_dict.items():
        sites = np.array(sites_dict[site_type])
        expected_sites = np.array(expected_sites)
        npt.assert_allclose(sites, expected_sites, rtol=1e-5, atol=1e-7)
