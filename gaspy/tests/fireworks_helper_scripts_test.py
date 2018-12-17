'''
Tests for the `fireworks_helper_scripts` submodule.
'''

__author__ = 'Aini Palizhati'
__email__ = 'apalizha@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we are testing
from ..fireworks_helper_scripts import check_jobs_status

# Things we need to do the tests
import pandas as pd

# Here the @parameterize defines varies different tuples so that the
# check_jobs_status will run multiple times using them in turns
import pytest


def test_check_jobs_output_type():
    ''' This function tests if the output is Pandas Dataframe '''
    dataframe = check_jobs_status('zulissi', 10)
    assert isinstance(dataframe, pd.DataFrame)


@pytest.mark.parametrize('user', ['zulissi', 'apalizha'])
def test_check_jobs_user(user):
    ''' This function test if the DataFrame contains only the user inquired '''
    dataframe = check_jobs_status(user, 10)
    user_from_results = dataframe['user'].unique()
    assert user == user_from_results


@pytest.mark.parametrize('n_jobs', [10, 20])
def test_check_jobs_num_jobs(n_jobs):
    ''' This function test if the DataFrame contains the requested number of rows '''
    dataframe = check_jobs_status('zulissi', n_jobs)
    n_jobs_from_results = len(dataframe.index)
    assert n_jobs == n_jobs_from_results
