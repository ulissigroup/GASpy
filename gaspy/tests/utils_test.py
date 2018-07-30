''' Tests for the `utils` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we're testing
from ..utils import find_adsorption_sites, encode_atoms_to_hex, decode_hex_to_atoms

# Things we need to do the tests
import numpy as np
import numpy.testing as npt
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
