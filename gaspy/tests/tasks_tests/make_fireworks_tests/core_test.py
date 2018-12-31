''' Tests for the `gaspy.tasks.make_fireworks.gases` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

import warnings
warnings.filterwarnings('ignore', category=ImportWarning)

# Things we're testing
from ....tasks.make_fireworks.core import (MakeGasFW,
                                           MakeBulkFW)

# Things we need to do the tests
from ..utils import clean_up_task
from .... import defaults
from ....tasks.atoms_generators import (GenerateGas,
                                        GenerateBulk)


def test_MakeGasFW():
    task = MakeGasFW('CO', defaults.GAS_SETTINGS['vasp'])

    # Check that our requirment is correct
    req = task.requires()
    assert isinstance(req, GenerateGas)
    assert req.gas_name == task.gas_name

    try:
        # Need to make sure our requirement is run before testing our task
        req.run()

        # Manually call the `run` method with the unit testing flag to get the
        # firework instead of actually submitting it
        fwork = task.run(_testing=True)
        assert fwork.name['gasname'] == 'CO'
        assert fwork.name['vasp_settings'] == defaults.GAS_SETTINGS['vasp']

    finally:
        clean_up_task(req)


def test_MakeBulkFW():
    task = MakeBulkFW('mp-30', defaults.BULK_SETTINGS['vasp'])

    # Check that our requirment is correct
    req = task.requires()
    assert isinstance(req, GenerateBulk)
    assert req.mpid == task.mpid

    try:
        # Need to make sure our requirement is run before testing our task
        req.run()

        # Manually call the `run` method with the unit testing flag to get the
        # firework instead of actually submitting it
        fwork = task.run(_testing=True)
        assert fwork.name['mpid'] == 'mp-30'
        assert fwork.name['vasp_settings'] == defaults.BULK_SETTINGS['vasp']

    finally:
        clean_up_task(req)
