"""Tests for the `fireworks_helper_scripts` submodule.
Most of the script formatting followed the formats in gaspy/test folder in UlissiGroup Github
That created by Kevin Tran"""

__author__ = 'Aini Palizhati'
__email__ = 'apalizha@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder

import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

#Things we are testing
from ..fireworks_helper_scripts import check_jobs_status

#Things we need to do the tests
import pandas as pd
from ..fireworks_helper_scripts import get_launchpad

#Here the @parameterize defines varies different tuples 
#so that the check_jobs_status will run multiple times using them in turns
import pytest

@pytest.mark.parametrize("test_input1, test_input2",[('zulissi',10)])
def test_output_type(test_input1, test_input2):
    """this function test if the output is Pandas Dataframe"""
    expected_output = pd.DataFrame
    output = check_jobs_status(test_input1, test_input2)
    assert isinstance(output, expected_output)

def check_user(test_input1, test_input2):
    """This function test if the DataFrame contains only the user inquired"""
    expected_output = test_input1
    docs = check_jobs_status(test_input1, test_input2)
    output = docs['user'].unique()
    assert output == expected_output

def check_num_jobs(test_input1, test_input2):
    """This function test if the DataFrame contains requested number of rows"""
    expected_output = test_input2
    docs = check_jobs_status(test_input1, test_input2)
    output = len(docs.index)
    assert output == expected_output
