'''
This framework uses a lot of atypical class objects such as ase.Atoms objects
or pymatgen.Structure objects. Testing functions that use these objects
requires specific examples. Here is where we store the examples we use to test.
'''

import ase
from pymatgen import Structure


def _get_standard_cell():
    ''' Returns a nested list for the standard unit cell of FCC Cu '''
    cell = [[0.0, 1.805, 1.805],
            [1.805, 0.0, 1.805],
            [1.805, 1.805, 0.0]]
    return cell


def get_standard_atoms():
    ''' Returns the ase Atoms object for a standard unit cell of FCC Cu '''
    atoms = ase.Atoms(symbols='Cu', pbc=True, cell=_get_standard_cell())
    return atoms


def get_standard_structure():
    ''' Returns the standard pymatgen structure object for a standard unit cell of FCC Cu '''
    structure = Structure(lattice=_get_standard_cell(), species=['Cu'], coords=[[0.0, 0.0, 0.0]])
    return structure
