'''
This script will populate your `catalog` Mongo collection with adsorption sites
of alloys containing the given set of elements and with Miller indices no
higher than the specified `max_miller`.
'''

__authors__ = ['Kevin Tran']
__email__ = 'ktran@andrew.cmu.edu'

from gaspy.tasks.db_managers.catalog import update_catalog_collection


elements = ['Ag', 'Al', 'As', 'Au', 'Ca', 'Cd', 'Cl', 'Co', 'Cr', 'Cs', 'Cu',
            'Fe', 'Ga', 'Ge', 'H', 'Hf', 'Hg', 'In', 'Ir', 'K', 'Mn', 'Mo',
            'N', 'Na', 'Nb', 'Os', 'P', 'Pb', 'Pd', 'Pt', 'Rh', 'Ru', 'S',
            'Sb', 'Sc', 'Se', 'Si', 'Sn', 'Sr', 'Ta', 'Tc', 'Te', 'Ti', 'V',
            'W', 'Y', 'Zn', 'Zr']
max_miller = 2

update_catalog_collection(elements=elements, max_miller=max_miller, n_processes=32)
