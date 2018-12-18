''' Tests for the `gaspy.tasks.generators` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.atoms_generators import (GenerateGas,
                                       GenerateBulk)

# Things we need to do the tests
import pytest
from ase.collections import g2
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from .utils import get_task_output, clean_up_task
from ...tasks import evaluate_luigi_task
from ...utils import read_rc
from ...mongo import make_atoms_from_doc


@pytest.mark.parametrize('gas_name', ['CO', 'H'])
def test_GenerateGas(gas_name):
    task = GenerateGas(gas_name)

    try:
        # Create, fetch, and parse the output of the task
        evaluate_luigi_task(task)
        doc = get_task_output(task)
        atoms = make_atoms_from_doc(doc)

        # Verify that the task worked by comparing it with what should be made
        expected_atoms = g2[gas_name]
        expected_atoms.positions += 10.
        expected_atoms.cell = [20, 20, 20]
        expected_atoms.pbc = [True, True, True]
        assert atoms == expected_atoms

    # Clean up
    finally:
        clean_up_task(task)


@pytest.mark.parametrize('mpid', ['mp-30', 'mp-867306'])
def test_GenerateBulk(mpid):
    task = GenerateBulk(mpid)

    try:
        # Create, fetch, and parse the output of the task
        evaluate_luigi_task(task)
        doc = get_task_output(task)
        atoms = make_atoms_from_doc(doc)

        # Verify that the task worked by comparing it with Materials Project
        with MPRester(read_rc('matproj_api_key')) as rester:
            structure = rester.get_structure_by_material_id(mpid)
        expected_atoms = AseAtomsAdaptor.get_atoms(structure)
        assert atoms == expected_atoms

    # Clean up
    finally:
        clean_up_task(task)
