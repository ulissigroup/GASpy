''' Tests for the `utils` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we're testing
from ..utils import find_adsorption_sites
from ..utils import encode_atoms_to_hex, decode_hex_to_atoms

# Things we need to do the tests
import numpy as np
import numpy.testing as npt
from .baselines import get_standard_structure
from .regression_tests.pymatgen_regression_test import _get_sites_for_standard_structure
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import pickle

def test_find_adsorption_sites():
    '''
    Check out `.regression_tests.pymatgen_regression_test._get_sites_for_standard_structure`
    to see what pymatgen gives us. Our `gaspy.utils.find_adsorption_sites` simply gives us
    the value of that object when the key is 'all'.
    '''
    standard_sites = _get_sites_for_standard_structure()['all']
    atoms = AseAtomsAdaptor.get_atoms(get_standard_structure())
    sites = find_adsorption_sites(atoms)
    npt.assert_allclose(np.array(sites), np.array(standard_sites), rtol=1e-5, atol=-1e-7)



def test_encode_decode():
    '''
    Test the encode/decode from atoms to hex and back
    '''
    atoms = Atoms('CC')
    npt.assert_equal(atoms,decode_hex_to_atoms(encode_atoms_to_hex(atoms)))

