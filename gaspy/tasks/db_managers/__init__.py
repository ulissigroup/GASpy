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


def update_all_collections():
    update_atoms_collection()
    update_adsorption_collection()
