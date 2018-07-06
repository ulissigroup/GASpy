''' Test the pymatgen functionalities that we use '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we'll be testing
from pymatgen.io.ase import AseAtomsAdaptor

# Things we need to do the testing
from ..baselines import get_standard_atoms, get_standard_structure


def test_AseAtomsAdaptor_get_atoms():
    standard_atoms = get_standard_atoms()
    standard_structure = get_standard_structure()

    atoms = AseAtomsAdaptor.get_atoms(standard_structure)
    assert atoms == standard_atoms


def test_AseAtomsAdaptor_get_structure():
    standard_atoms = get_standard_atoms()
    standard_structure = get_standard_structure()

    structure = AseAtomsAdaptor.get_structure(standard_atoms)
    assert structure == standard_structure
