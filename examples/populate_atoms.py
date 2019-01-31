'''
This script will populate your `atoms` Mongo collection with completed
calculations in your FireWorks database.
'''

__authors__ = ['Kevin Tran']
__email__ = 'ktran@andrew.cmu.edu'

from gaspy.tasks.db_managers.atoms import update_atoms_collection


update_atoms_collection(n_processes=32, progress_bar=True)
