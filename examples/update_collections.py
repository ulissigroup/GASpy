'''
This script will populate your `atoms` Mongo collection with completed
calculations in your FireWorks database.
'''

__authors__ = ['Kevin Tran']
__email__ = 'ktran@andrew.cmu.edu'

from gaspy.tasks.db_managers import update_all_collections


update_all_collections(n_processes=32)
