''' Tests for the `gaspy.tasks.metadata_calculators` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.metadata_calculators import (CalculateAdsorptionEnergy,
                                           CalculateAdsorbateEnergy,
                                           CalculateAdsorbateBasisEnergies)

# Things we need to do the tests
import pytest
import math
from .utils import clean_up_tasks, run_task_locally
from ... import defaults
from ...utils import unfreeze_dict
from ...tasks import get_task_output, run_task

GAS_SETTINGS = defaults.gas_settings()


def test_CalculateAdsorptionEnergy():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the gas calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `run_task_locally` appropriately.
    '''
    adsorption_site = (0., 1.41, 20.52)
    shift = 0.25
    top = True
    adsorbate_name = 'CO'
    mpid = 'mp-2'
    miller_indices = (1, 0, 0)
    task = CalculateAdsorptionEnergy(adsorption_site=adsorption_site,
                                     shift=shift,
                                     top=top,
                                     adsorbate_name=adsorbate_name,
                                     mpid=mpid,
                                     miller_indices=miller_indices,)

    try:
        run_task_locally(task)
        doc = get_task_output(task)

        # I just checked this one calculation by hand and found some key
        # information about it.
        assert math.isclose(doc['adsorption_energy'], -1.5959449799999899)
        assert doc['fwids']['slab'] == 124894
        assert doc['fwids']['adslab'] == 124897

    finally:
        clean_up_tasks()


def test_CalculateAdsorbateEnergy():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the gas calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `run_task_locally` appropriately.
    '''
    adsorbate_name = 'OOH'
    vasp_settings = GAS_SETTINGS['vasp']
    task = CalculateAdsorbateEnergy(adsorbate_name=adsorbate_name,
                                    vasp_settings=vasp_settings)
    assert task.adsorbate_name == adsorbate_name
    assert unfreeze_dict(task.vasp_settings) == vasp_settings

    try:
        run_task_locally(task)
        energy = get_task_output(task)
        assert energy == 2*(-7.19957549) + (-3.480310465)

    finally:
        clean_up_tasks()


def test_CalculateAdsorbateEnergy_Error():
    '''
    WARNING:  This test uses `run_task`, which has a chance of actually
    submitting a FireWork to production. To avoid this, you must try to make
    sure that you have all of the gas calculations in the unit testing atoms
    collection.  If you copy/paste this test into somewhere else, make sure
    that you use `run_task` appropriately.

    If we try to calculate the energy of an adsorbate that we have not yet
    defined, then this task should yell at us.
    '''
    adsorbate_name = 'U'
    vasp_settings = GAS_SETTINGS['vasp']
    task = CalculateAdsorbateEnergy(adsorbate_name=adsorbate_name,
                                    vasp_settings=vasp_settings)
    assert task.adsorbate_name == adsorbate_name
    assert unfreeze_dict(task.vasp_settings) == vasp_settings

    try:
        with pytest.raises(KeyError, message='Expected a KeyError') as exc_info:
            run_task(task)
            assert ('You are trying to calculate the adsorbate energy of an '
                    'undefined adsorbate, U' in str(exc_info.value))

    finally:
        clean_up_tasks()


def test_CalculateAdsorbateBasisEnergies():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the gas calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `run_task_locally` appropriately.
    '''
    vasp_settings = GAS_SETTINGS['vasp']
    task = CalculateAdsorbateBasisEnergies(vasp_settings)
    assert unfreeze_dict(task.vasp_settings) == vasp_settings

    try:
        run_task_locally(task)
        basis_energies = get_task_output(task)
        assert basis_energies == {'H': -3.480310465,
                                  'O': -7.19957549,
                                  'C': -7.29110228,
                                  'N': -8.08570028}

    finally:
        clean_up_tasks()
