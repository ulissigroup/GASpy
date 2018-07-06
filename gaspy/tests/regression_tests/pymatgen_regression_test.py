''' Test the pymatgen functionalities that we use '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we'll be doing regression tests on
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

# Things we need to do the tests
#import pytest
import numpy as np
import numpy.testing as npt
from ..learning_tests.pymatgen_test import _get_standard_structure


def _get_sites_for_standard_structure():
    ''' These are the sites that pymatgen v2018.6.11 found on our standard structure. '''
    sites = {'all': [np.array([1.15470054, 1.15470054, -1.15470054]),
                     np.array([2.95970054, 3.86220054, -0.25220054]),
                     np.array([1.75636721, 1.75636721, 0.04863279])],
             'bridge': [np.array([2.05720054, 2.05720054, 0.65029946])],
             'hollow': [np.array([2.35803387, 2.35803387, 1.25196613])],
             'ontop': [np.array([1.15470054, 1.15470054, -1.15470054])]}
    return sites


def test_AdsorbateSiteFinder_find_adsorption_site_for_standard_structure_site_types():
    ''' Verify that ASF finds the same exact site types (e.g., ontop, bride, hollow) '''
    struct = _get_standard_structure()
    site_types = AdsorbateSiteFinder(struct).find_adsorption_sites(put_inside=True).keys()
    standard_site_types = _get_sites_for_standard_structure().keys()
    assert site_types == standard_site_types


def test_AdsorbateSiteFinder_find_adsorption_site_for_standard_structure_sites():
    ''' Verify that ASF finds the same cartesion site locations for each site type '''
    struct = _get_standard_structure()
    sites_dict = AdsorbateSiteFinder(struct).find_adsorption_sites(put_inside=True)
    standard_sites_dict = _get_sites_for_standard_structure()
    # The output we're checking is a dictionary of lists of numpy arrays.
    # Let's check convert each list to an array and then check each array one at a time.
    for site_type in standard_sites_dict:
        sites = np.array(sites_dict[site_type])
        standard_sites = np.array(standard_sites_dict[site_type])
        npt.assert_allclose(sites, standard_sites, rtol=1e-5, atol=1e-7)
