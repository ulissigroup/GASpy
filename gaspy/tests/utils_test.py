''' Tests for the `utils` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we're testing
from ..utils import find_adsorption_sites, \
    unfreeze_dict, \
    encode_atoms_to_hex, \
    decode_hex_to_atoms, \
    encode_atoms_to_trajhex, \
    decode_trajhex_to_atoms

# Things we need to do the tests
import pytest
import pickle
import numpy as np
import numpy.testing as npt
from luigi.parameter import _FrozenOrderedDict
from . import test_cases

REGRESSION_BASELINES_LOCATION = '/home/GASpy/gaspy/tests/regression_baselines/utils/'


@pytest.mark.baseline
@pytest.mark.parametrize('slab_atoms_name',
                         ['AlAu2Cu_210.traj',
                          'CoSb2_110.traj',
                          'Cu_211.traj',
                          'FeNi_001.traj',
                          'Ni4W_001.traj',
                          'Pt12Si5_110.traj'])
def test_to_create_adsorption_sites(slab_atoms_name):
    atoms = test_cases.get_slab_atoms(slab_atoms_name)
    sites = find_adsorption_sites(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'sites_for_' + slab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(sites, file_handle)
    assert True


@pytest.mark.parametrize('slab_atoms_name',
                         ['AlAu2Cu_210.traj',
                          'CoSb2_110.traj',
                          'Cu_211.traj',
                          'FeNi_001.traj',
                          'Ni4W_001.traj',
                          'Pt12Si5_110.traj'])
def test_find_adsorption_sites(slab_atoms_name):
    '''
    Check out `.learning_tests.pymatgen_test._get_sites_for_standard_structure`
    to see what pymatgen gives us. Our `gaspy.utils.find_adsorption_sites` simply gives us
    the value of that object when the key is 'all'.
    '''
    atoms = test_cases.get_slab_atoms(slab_atoms_name)
    sites = find_adsorption_sites(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'sites_for_' + slab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_sites = pickle.load(file_handle)

    npt.assert_allclose(np.array(sites), np.array(expected_sites), rtol=1e-5, atol=1e-7)


def test_unfreeze_dict():
    frozen_dict = _FrozenOrderedDict(foo='bar', alpha='omega',
                                     sub_dict0=_FrozenOrderedDict(),
                                     sub_dict1=_FrozenOrderedDict(great='googly moogly'))
    unfrozen_dict = unfreeze_dict(frozen_dict)
    _look_for_type_in_dict(type_=_FrozenOrderedDict, dict_=unfrozen_dict)


def _look_for_type_in_dict(type_, dict_):
    '''
    Recursive function that checks if there is any object type inside any branch
    of a dictionary. It does so by performing an `assert` check on every single
    value in the dictionary.

    Args:
        type_   An object type (e.g, int, float, str, etc) that you want to look for
        dict_   A dictionary that you want to parse. Can really be any object with
                the `items` method.
    '''
    # Check the current layer's values
    for key, value in dict_.items():
        assert type(value) != type_
        # Recur
        try:
            _look_for_type_in_dict(type_, value)
        except AttributeError:
            pass


@pytest.mark.baseline
@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'C_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_to_create_atoms_hex_encoding(adslab_atoms_name):
    '''
    This is hard-coded for adslabs. It should be able to work on bulks and slabs, too.
    Feel free to update it.
    '''
    atoms = test_cases.get_adslab_atoms(adslab_atoms_name)
    hex_ = encode_atoms_to_hex(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'hex_for_' + adslab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(hex_, file_handle)
    assert True


@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'C_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_encode_atoms_to_hex(adslab_atoms_name):
    '''
    This is hard-coded for adslabs. It should be able to work on bulks and slabs, too.
    Feel free to update it.
    '''
    atoms = test_cases.get_adslab_atoms(adslab_atoms_name)
    hex_ = encode_atoms_to_hex(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'hex_for_' + adslab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_hex = pickle.load(file_handle)
    assert hex_ == expected_hex


@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'C_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_decode_hex_to_atoms(adslab_atoms_name):
    '''
    This is hard-coded for adslabs. It should be able to work on bulks and slabs, too.
    Feel free to update it.
    '''
    file_name = REGRESSION_BASELINES_LOCATION + 'hex_for_' + adslab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'rb') as file_handle:
        hex_ = pickle.load(file_handle)
    atoms = decode_hex_to_atoms(hex_)

    expected_atoms = test_cases.get_adslab_atoms(adslab_atoms_name)
    assert atoms == expected_atoms


@pytest.mark.baseline
@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'C_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_to_create_atoms_trajhex_encoding(adslab_atoms_name):
    '''
    This is hard-coded for adslabs. It should be able to work on bulks and slabs, too.
    Feel free to update it.
    '''
    atoms = test_cases.get_adslab_atoms(adslab_atoms_name)
    hex_ = encode_atoms_to_trajhex(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'trajhex_for_' + adslab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(hex_, file_handle)
    assert True


@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'C_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_encode_atoms_to_trajhex(adslab_atoms_name):
    '''
    This is hard-coded for adslabs. It should be able to work on bulks and slabs, too.
    Feel free to update it.
    '''
    atoms = test_cases.get_adslab_atoms(adslab_atoms_name)
    trajhex = encode_atoms_to_trajhex(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'trajhex_for_' + adslab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_trajhex = pickle.load(file_handle)
    assert trajhex == expected_trajhex


@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'C_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_decode_trajhex_to_atoms(adslab_atoms_name):
    '''
    This is hard-coded for adslabs. It should be able to work on bulks and slabs, too.
    Feel free to update it.
    '''
    file_name = REGRESSION_BASELINES_LOCATION + 'trajhex_for_' + adslab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'rb') as file_handle:
        trajhex = pickle.load(file_handle)
    atoms = decode_trajhex_to_atoms(trajhex)

    expected_atoms = test_cases.get_adslab_atoms(adslab_atoms_name)
    assert atoms == expected_atoms
