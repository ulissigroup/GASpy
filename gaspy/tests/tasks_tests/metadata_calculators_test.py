''' Tests for the `gaspy.tasks.metadata_calculators` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.metadata_calculators import (CalculateAdsorbateEnergy,
                                           CalculateAdsorbateBasisEnergies)

# Things we need to do the tests
import pytest
from .utils import clean_up_task
from ... import defaults
from ...utils import unfreeze_dict
from ...tasks import get_task_output, evaluate_luigi_task


def test_CalculateAdsorbateEnergy():
    '''
    WARNING:  This test uses `evaluate_luigi_task`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the gas calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `evaluate_luigi_task` appropriately.
    '''
    adsorbate_name = 'OOH'
    vasp_settings = defaults.GAS_SETTINGS['vasp']
    task = CalculateAdsorbateEnergy(adsorbate_name=adsorbate_name,
                                    vasp_settings=vasp_settings)
    assert task.adsorbate_name == adsorbate_name
    assert unfreeze_dict(task.vasp_settings) == vasp_settings

    try:
        evaluate_luigi_task(task)
        energy = get_task_output(task)
        assert energy == 2*(-7.19957549) + (-3.480310465)

    finally:
        clean_up_task(task)


def test_CalculateAdsorbateEnergy_Error():
    '''
    WARNING:  This test uses `evaluate_luigi_task`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the gas calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `evaluate_luigi_task` appropriately.

    If we try to calculate the energy of an adsorbate that we have not yet
    defined, then this task should yell at us.
    '''
    adsorbate_name = 'U'
    vasp_settings = defaults.GAS_SETTINGS['vasp']
    task = CalculateAdsorbateEnergy(adsorbate_name=adsorbate_name,
                                    vasp_settings=vasp_settings)
    assert task.adsorbate_name == adsorbate_name
    assert unfreeze_dict(task.vasp_settings) == vasp_settings

    try:
        with pytest.raises(KeyError, message='Expected a KeyError') as exc_info:
            evaluate_luigi_task(task)
            assert ('You are trying to calculate the adsorbate energy of an '
                    'undefined adsorbate, U' in str(exc_info.value))

    finally:
        clean_up_task(task)


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
