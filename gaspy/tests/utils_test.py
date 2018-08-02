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
import numpy as np
import numpy.testing as npt
from luigi.parameter import _FrozenOrderedDict
from .baselines import get_standard_atoms
from .learning_tests.pymatgen_test import _get_sites_for_standard_structure


def test_find_adsorption_sites():
    '''
    Check out `.learning_tests.pymatgen_test._get_sites_for_standard_structure`
    to see what pymatgen gives us. Our `gaspy.utils.find_adsorption_sites` simply gives us
    the value of that object when the key is 'all'.
    '''
    standard_sites = _get_sites_for_standard_structure()['all']
    sites = find_adsorption_sites(get_standard_atoms())
    npt.assert_allclose(np.array(sites), np.array(standard_sites), rtol=1e-5, atol=-1e-7)


def test_unfreeze_dict():
    frozen_dict = _FrozenOrderedDict(foo='bar', alpha='omega',
                                     sub_dict0=_FrozenOrderedDict(),
                                     sub_dict1=_FrozenOrderedDict(great='googly moogly'))
    unfrozen_dict = unfreeze_dict(frozen_dict)
    _look_for_type_in_dict(_FrozenOrderedDict, unfrozen_dict)


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


def test_encode_atoms_to_hex():
    atoms = get_standard_atoms()
    atoms_hex = encode_atoms_to_hex(atoms)
    standard_atoms_hex = _get_standard_atoms_hex()
    assert atoms_hex == standard_atoms_hex


def test_decode_hex_to_atoms():
    atoms_hex = _get_standard_atoms_hex()
    atoms = decode_hex_to_atoms(atoms_hex)
    standard_atoms = get_standard_atoms()
    assert atoms == standard_atoms


def _get_standard_atoms_hex():
    '''
    This is the hex string that is supposed to be created when encoding our standard atoms object.
    '''
    standard_atoms_hex = '8003636173652e61746f6d730a41746f6d730a7100298171017d710228580600000061727261797371037d71042858070000006e756d6265727371056364696c6c2e5f64696c6c0a5f6765745f617474720a71066364696c6c2e5f64696c6c0a5f696d706f72745f6d6f64756c650a710758150000006e756d70792e636f72652e6d756c74696172726179710885710952710a580c0000005f7265636f6e737472756374710b86710c52710d636e756d70790a6e6461727261790a710e4b0085710f4301627110877111527112284b014b01857113636e756d70790a64747970650a71145802000000693871154b004b01877116527117284b0358010000003c71184e4e4e4affffffff4affffffff4b00747119628943081d00000000000000711a74711b625809000000706f736974696f6e73711c680d680e4b0085711d681087711e52711f284b014b014b0386712068145802000000663871214b004b01877122527123284b0368184e4e4e4affffffff4affffffff4b00747124628943180000000000000000000000000000000000000000000000007125747126627558050000005f63656c6c7127680d680e4b00857128681087712952712a284b014b034b0386712b68238943480000000000000000e17a14ae47e1fc3fe17a14ae47e1fc3fe17a14ae47e1fc3f0000000000000000e17a14ae47e1fc3fe17a14ae47e1fc3fe17a14ae47e1fc3f0000000000000000712c74712d6258090000005f63656c6c64697370712e680d680e4b0085712f6810877130527131284b014b034b018671326823894318000000000000000000000000000000000000000000000000713374713462580c0000005f636f6e73747261696e747371355d713658040000005f7062637137680d680e4b00857138681087713952713a284b014b0385713b681458020000006231713c4b004b0187713d52713e284b0358010000007c713f4e4e4e4affffffff4affffffff4b00747140628943030101017141747142625804000000696e666f71437d714458050000005f63616c6371454e75622e'
    return standard_atoms_hex


def test_encode_atoms_to_trajhex():
    atoms = get_standard_atoms()
    atoms_trajhex = encode_atoms_to_trajhex(atoms)
    expected_atoms_trajhex = _get_standard_trajhex()
    assert atoms_trajhex == expected_atoms_trajhex


def test_decode_trajhex_to_atoms():
    atoms_trajhex = _get_standard_trajhex()
    atoms = decode_trajhex_to_atoms(atoms_trajhex)
    expected_atoms = get_standard_atoms()
    assert atoms == expected_atoms


def _get_standard_trajhex():
    trajhex = '2d206f6620556c6d4153452d5472616a6563746f7279202003000000000000000100000000000000300000000000000058000000000000001d00000000000000000000000000000000000000000000000000000000000000ec000000000000007b2276657273696f6e223a20312c20226173655f76657273696f6e223a2022332e31362e32222c2022706263223a205b747275652c20747275652c20747275655d2c20226e756d626572732e223a207b226e646172726179223a205b5b315d2c2022696e743634222c2035365d7d2c2022706f736974696f6e732e223a207b226e646172726179223a205b5b312c20335d2c2022666c6f61743634222c2036345d7d2c202263656c6c223a205b5b302e302c20312e3830352c20312e3830355d2c205b312e3830352c20302e302c20312e3830355d2c205b312e3830352c20312e3830352c20302e305d5d7d'
    return trajhex
