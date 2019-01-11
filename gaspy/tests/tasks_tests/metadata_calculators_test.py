''' Tests for the `gaspy.tasks.metadata_calculators` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.metadata_calculators import (CalculateAdsorbateBasisEnergies)

# Things we need to do the tests
from .utils import clean_up_task
from ... import defaults
from ...utils import unfreeze_dict
from ...tasks import get_task_output, evaluate_luigi_task


def test_CalculateAdsorbateBasisEnergies():
    '''
    WARNING:  This test uses `evaluate_luigi_task`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the gas calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `evaluate_luigi_task` appropriately.
    '''
    vasp_settings = defaults.GAS_SETTINGS['vasp']
    task = CalculateAdsorbateBasisEnergies(vasp_settings)
    assert unfreeze_dict(task.vasp_settings) == vasp_settings

    try:
        evaluate_luigi_task(task)
        basis_energies = get_task_output(task)
        assert basis_energies == {'H': -3.480310465,
                                  'O': -7.19957549,
                                  'C': -7.29110228,
                                  'N': -8.08570028}

    finally:
        clean_up_task(task)
