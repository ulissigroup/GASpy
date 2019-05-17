'''
This submodule contains the various functions/Luigi tasks that manage our Mongo
databases
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# flake8: noqa

from .catalog import update_catalog_collection
from .atoms import update_atoms_collection
from .adsorption import update_adsorption_collection
from .surfaces import update_surface_energy_collection


def update_all_collections(n_processes=1):
    update_atoms_collection(n_processes=n_processes)
    update_adsorption_collection(n_processes=n_processes)
    udpate_surface_energy_collection(n_processes=n_processes)
