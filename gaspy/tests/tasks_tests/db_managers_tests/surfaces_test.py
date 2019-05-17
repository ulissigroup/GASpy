''' Tests for the `gaspy.tasks.db_managers.adsorption` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ....tasks.db_managers.surfaces import (update_surface_energy_collection,
                                            _find_atoms_docs_not_in_surface_energy_collection,
                                            __run_calculate_surface_energy_task,
                                            __create_surface_energy_doc)

# Things we need to do the testing
from gaspy.tasks.core import get_task_output, schedule_tasks
from gaspy.tasks.calculation_finders import FindBulk
from gaspy.tasks.metadata_calculators import CalculateSurfaceEnergy
from ..utils import clean_up_tasks


def test_update_surface_energy_collection():
    assert False


def test__find_atoms_docs_not_in_surface_energy_collection():
    assert False


def test___run_calculate_surface_energy_task():
    try:
        # It turns out that our `run_task` function works terribly with dynamic
        # dependencies. It works less terribly when the dependencies are
        # already done. For this test, let's run that dependency first. In
        # production, we will effectively rely on our periodically running
        # scripts to take care of this pre-run part.
        # Note that we use `run_task` because `schedule_tasks` hangs up on
        # unfinished tasks, and we don't want it to hang up during database
        # updates.
        bulk_task = FindBulk(mpid='mp-1018129')
        schedule_tasks([bulk_task], local_scheduler=True)

        # Make sure the task can run from scratch
        task = CalculateSurfaceEnergy(mpid='mp-1018129', miller_indices=(0, 0, 1), shift=0.081)
        __run_calculate_surface_energy_task(task)
        surface_energy_doc = get_task_output(task)
        assert isinstance(surface_energy_doc, dict)

        # Make sure the task can run multiple timse without error
        task = CalculateSurfaceEnergy(mpid='mp-1018129', miller_indices=(0, 0, 1), shift=0.081)
        __run_calculate_surface_energy_task(task)
        surface_energy_doc = get_task_output(task)
        assert isinstance(surface_energy_doc, dict)

        # Make sure the function won't throw an error when the task isn't done
        task = CalculateSurfaceEnergy(mpid='mp-1018129', miller_indices=(0, 0, 1), shift=9001)
        __run_calculate_surface_energy_task(task)

    finally:
        clean_up_tasks()


def test___create_surface_energy_doc():
    assert False
