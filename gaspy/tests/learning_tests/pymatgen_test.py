''' Test the pymatgen functionalities that we use '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we'll be doing learning tests on
from pymatgen.io.ase import AseAtomsAdaptor

# Things we need to do the tests. Yes these are 3rd party objects,
# but they're at least less likely to change.
import ase
from pymatgen import Structure


def _get_standard_cell():
    ''' Returns a nested list for the standard unit cell of FCC Cu '''
    cell = [[0.0, 1.805, 1.805],
            [1.805, 0.0, 1.805],
            [1.805, 1.805, 0.0]]
    return cell


def _get_standard_atoms():
    ''' Returns the ase Atoms object for a standard unit cell of FCC Cu '''
    atoms = ase.Atoms(symbols='Cu', pbc=True, cell=_get_standard_cell())
    return atoms


def _get_standard_structure():
    ''' Returns the standard pymatgen structure object for a standard unit cell of FCC Cu '''
    structure = Structure(lattice=_get_standard_cell(), species=['Cu'], coords=[[0.0, 0.0, 0.0]])
    return structure


def test_AseAtomsAdaptor_get_atoms():
    standard_atoms = _get_standard_atoms()
    standard_structure = _get_standard_structure()

    atoms = AseAtomsAdaptor.get_atoms(standard_structure)
    assert atoms == standard_atoms


def test_AseAtomsAdaptor_get_structure():
    standard_atoms = _get_standard_atoms()
    standard_structure = _get_standard_structure()

    structure = AseAtomsAdaptor.get_structure(standard_atoms)
    assert structure == standard_structure
