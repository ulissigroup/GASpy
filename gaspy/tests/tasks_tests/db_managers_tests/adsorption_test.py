''' Tests for the `gaspy.tasks.db_managers.adsorption` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ....tasks.db_managers.adsorption import (update_adsorption_collection,
                                              _find_atoms_docs_not_in_adsorption_collection,
                                              __get_luigi_adsorption_energies,
                                              __create_adsorption_doc)

# Things we need to do the tests
from ....tasks.core import run_tasks, get_task_output
from ....tasks.metadata_calculators import CalculateAdsorptionEnergy
from ....utils import turn_string_site_into_tuple
from ....mongo import make_atoms_from_doc, make_doc_from_atoms
from ....gasdb import get_mongo_collection
from ....atoms_operators import fingerprint_adslab, find_max_movement


def test_update_adsorption_collection():
    assert False


def test__find_atoms_docs_not_in_adsorption_collection():
    assert False


def test___get_luigi_adsorption_energies():
    assert False


def test___create_adsorption_doc():
    assert False
