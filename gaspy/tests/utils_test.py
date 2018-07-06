''' Tests for the `utils` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we're testing
from ..utils import find_adsorption_sites

# Things we need to do the tests
import numpy as np
import numpy.testing as npt
from .baselines import get_standard_structure
from .regression_tests.pymatgen_regression_test import _get_sites_for_standard_structure


def test_find_adsorption_sites():
    '''
    Check out `.regression_tests.pymatgen_regression_test._get_sites_for_standard_structure`
    to see what pymatgen gives us. Our `gaspy.utils.find_adsorption_sites` simply gives us
    the value of that object when the key is 'all'.
    '''
    standard_sites = _get_sites_for_standard_structure()['all']
    sites = find_adsorption_sites(get_standard_structure())
    npt.assert_allclose(np.array(sites), np.array(standard_sites), rtol=1e-5, atol=-1e-7)
