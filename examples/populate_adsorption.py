'''
This script will populate your `adsorption` Mongo collection with completed
calculations in your FireWorks database.
'''

__authors__ = ['Kevin Tran']
__email__ = 'ktran@andrew.cmu.edu'

from gaspy.tasks.db_managers.adsorption import update_adsorption_collection


update_adsorption_collection(n_processes=32)
