'''
This framework uses a lot of ase.Atoms objects. Testing functions that use these
objects requires specific instances. Here is where we fetch the instances we use to test.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# flake8: noqa
from .utils import (get_bulk_atoms,
                    get_slab_atoms,
                    get_adslab_atoms,
                    get_bulk_structure,
                    get_slab_structure,
                    get_adslab_structure,
                    relax_atoms)
