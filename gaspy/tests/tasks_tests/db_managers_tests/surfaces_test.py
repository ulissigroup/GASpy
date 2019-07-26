''' Tests for the `gaspy.tasks.db_managers.adsorption` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ....tasks.db_managers.surfaces import (update_surface_energy_collection,
                                            _find_surfaces_from_docs,
                                            _find_atoms_docs_not_in_surface_energy_collection,
                                            __run_calculate_surface_energy_task,
                                            __create_surface_energy_doc)

# Things we need to do the testing
import warnings
import datetime
import ase
from ..utils import clean_up_tasks
from ...test_cases.mongo_test_collections.mongo_utils import populate_unit_testing_collection
from ....gasdb import get_mongo_collection
from ....defaults import DFT_CALCULATOR
from ....mongo import make_atoms_from_doc
from ....tasks.core import get_task_output, schedule_tasks
from ....tasks.calculation_finders import FindBulk
from ....tasks.metadata_calculators import CalculateSurfaceEnergy


def test_update_surface_energy_collection():
    '''
    This may seem like a crappy test, but we really rely on unit testing of the
    sub-functions for integrity testing.
    '''
    try:
        # Clear out the surface energy collection before testing
        with get_mongo_collection('surface_energy_%s' % DFT_CALCULATOR) as collection:
            collection.delete_many({})

        # Try to add documents to the collection
        with warnings.catch_warnings():  # Ignore warnings for now
            warnings.simplefilter('ignore')
            update_surface_energy_collection()

        # Test if we did add anything
        with get_mongo_collection('surface_energy_%s' % DFT_CALCULATOR) as collection:
            assert collection.count_documents({}) > 0

    # Reset the surface energy collection behind us
    finally:
        with get_mongo_collection('surface_energy_%s' % DFT_CALCULATOR) as collection:
            collection.delete_many({})
        populate_unit_testing_collection('surface_energy_%s' % DFT_CALCULATOR)


def test__find_surfaces_from_docs():
    docs = _find_atoms_docs_not_in_surface_energy_collection(DFT_CALCULATOR)
    surfaces = _find_surfaces_from_docs(docs)

    # Find all of the surfaces manually
    for doc in docs:
        mpid = doc['fwname']['mpid']
        miller_indices = tuple(doc['fwname']['miller'])
        shift = round(doc['fwname']['shift'], 3)
        dft_settings = doc['fwname']['dft_settings']
        dft_settings['kpts'] = tuple(dft_settings['kpts'])  # make hashable
        dft_settings = tuple((key, value) for key, value in dft_settings.items())
        surface = (mpid, miller_indices, shift, dft_settings)

        # Make sure they inside the set of surfaces our function found
        assert surface in surfaces


def test__find_atoms_docs_not_in_surface_energy_collection():
    docs = _find_atoms_docs_not_in_surface_energy_collection(DFT_CALCULATOR)

    # Make sure these "documents" are actually `atoms` docs that can be turned
    # into `ase.Atoms` objects
    for doc in docs:
        atoms = make_atoms_from_doc(doc)
        assert isinstance(atoms, ase.Atoms)

    # Make sure that everything we found actually isn't in our `surface_energy`
    # collection
    fwids_in_atoms = {doc['fwid'] for doc in docs}
    with get_mongo_collection('surface_energy_%s' % DFT_CALCULATOR) as collection:
        surf_docs = list(collection.find({}, {'fwids': 1, '_id': 0}))
    fwids_in_surf = {fwid for doc in surf_docs for fwid in doc['fwids']}
    assert fwids_in_atoms.isdisjoint(fwids_in_surf)


def test___run_calculate_surface_energy_task():
    '''
    It turns out that our `run_task` function works terribly with dynamic
    dependencies. It works less terribly when the dependencies are already
    done. For this test, let's run that dependency first. In production, we
    will effectively rely on our periodically running scripts to take care of
    this pre-run part.

    Note that we use `run_task` because `schedule_tasks` hangs up on unfinished
    tasks, and we don't want it to hang up during database updates.
    '''
    try:
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

        # TODO:  Figure out a way to test this feature without accidentally
        # submitting a job to run on FireWorks
        # Make sure the function won't throw an error when the task isn't done
        #task = CalculateSurfaceEnergy(mpid='mp-1018129', miller_indices=(0, 0, 1), shift=9001)
        #__run_calculate_surface_energy_task(task)

    finally:
        clean_up_tasks()


def test___create_surface_energy_doc():
    try:
        # We need to run the task before making a document for it
        bulk_task = FindBulk(mpid='mp-1018129')
        schedule_tasks([bulk_task], local_scheduler=True)
        task = CalculateSurfaceEnergy(mpid='mp-1018129', miller_indices=(0, 0, 1), shift=0.081)
        __run_calculate_surface_energy_task(task)

        doc = __create_surface_energy_doc(task)
        for structure_doc in doc['surface_structures']:
            assert isinstance(make_atoms_from_doc(structure_doc), ase.Atoms)
        assert isinstance(doc['surface_energy'], float)
        assert isinstance(doc['surface_energy_standard_error'], float)
        for movement in doc['max_atom_movement']:
            assert isinstance(movement, float)
        for fwid in doc['fwids']:
            assert isinstance(fwid, int)
        for date in doc['calculation_dates']:
            assert isinstance(date, datetime.datetime)
        for directory in doc['fw_directories']:
            assert isinstance(directory, str)

    finally:
        clean_up_tasks()
