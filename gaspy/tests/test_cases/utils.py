'''
This framework uses a lot of ase.Atoms objects. Testing functions that use these
objects requires specific instances. Here is where we fetch the instances we use to test.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import ase.io
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from pymatgen.io.ase import AseAtomsAdaptor

BULKS_LOCATION = '/home/GASpy/gaspy/tests/test_cases/bulks/'
SLABS_LOCATION = '/home/GASpy/gaspy/tests/test_cases/slabs/'
ADSLABS_LOCATION = '/home/GASpy/gaspy/tests/test_cases/adslabs/'


def get_bulk_atoms(name):
    '''
    Gets the ase.Atoms object of a bulk structure that is stored
    in the cache of test cases.

    Arg:
        name    String indicating the name of the atoms object you want to read,
                e.g., `Cu_FCC.traj`. You need to look in BULKS_LOCATION to see
                what options there are.
    Output:
        atoms   ase.Atoms object of whichever object you chose to get.
    '''
    atoms = ase.io.read(BULKS_LOCATION + name)
    return atoms


def get_slab_atoms(name):
    '''
    Gets the ase.Atoms object of a slab structure that is stored
    in the cache of test cases.

    Arg:
        name    String indicating the name of the atoms object you want to read,
                e.g., `Cu_211.traj`. You need to look in SLABS_LOCATION to see
                what options there are.
    Output:
        atoms   ase.Atoms object of whichever object you chose to get.
    '''
    atoms = ase.io.read(SLABS_LOCATION + name)
    return atoms


def get_adslab_atoms(name):
    '''
    Gets the ase.Atoms object of a bulk structure that is stored
    in the cache of test cases.

    Arg:
        name    String indicating the name of the atoms object you want to read,
                e.g., `CO_top_Cu.traj`. You need to look in ADSLABS_LOCATION to see
                what options there are.
    Output:
        atoms   ase.Atoms object of whichever object you chose to get.
    '''
    atoms = ase.io.read(ADSLABS_LOCATION + name)
    return atoms


def get_bulk_structure(name):
    '''
    Gets the pymatgen.Structure object of a bulk structure that is stored
    in the cache of test cases.

    Arg:
        name    String indicating the name of the object you want to read,
                e.g., `Cu_FCC.traj`. You need to look in BULKS_LOCATION to see
                what options there are.
    Output:
        structure   pymatgen.Structure object of whichever object you chose to get.
    '''
    atoms = get_bulk_atoms(name)
    structure = AseAtomsAdaptor.get_structure(atoms)
    return structure


def get_slab_structure(name):
    '''
    Gets the pymatgen.Structure object of a bulk structure that is stored
    in the cache of test cases.

    Arg:
        name    String indicating the name of the object you want to read,
                e.g., `Cu_211.traj`. You need to look in SLABS_LOCATION to see
                what options there are.
    Output:
        structure   pymatgen.Structure object of whichever object you chose to get.
    '''
    atoms = get_slab_atoms(name)
    structure = AseAtomsAdaptor.get_structure(atoms)
    return structure


def get_adslab_structure(name):
    '''
    Gets the pymatgen.Structure object of a bulk structure that is stored
    in the cache of test cases.

    Arg:
        name    String indicating the name of the object you want to read,
                e.g., `CO_top_Cu.traj`. You need to look in ADSLABS_LOCATION to see
                what options there are.
    Output:
        structure   pymatgen.Structure object of whichever object you chose to get.
    '''
    atoms = get_adslab_atoms(name)
    structure = AseAtomsAdaptor.get_structure(atoms)
    return structure


def relax_atoms(atoms):
    '''
    Performs a quick, rough relaxation of an atoms object using EMT and BFGS.

    Arg:
        atoms   ase.Atoms object that you want to relax
    Output:
        atoms   Relaxed version of the input
    '''
    calculator = EMT()
    atoms.set_calculator(calculator)

    dynamics = BFGS(atoms)
    dynamics.run()

    return atoms
