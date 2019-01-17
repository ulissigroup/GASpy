''' Tests for the `gaspy.tasks.db_managers` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.db_managers import (UpdateCatalogCollection,
                                  _GetMpids,
                                  _InsertSitesToCatalog)

# Things we need to do the tests
import numpy.testing as npt
from pymatgen.ext.matproj import MPRester
from .utils import clean_up_tasks, run_task_locally
from ... import defaults
from ...utils import unfreeze_dict, read_rc
from ...mongo import make_atoms_from_doc
from ...tasks import get_task_output

SLAB_SETTINGS = defaults.SLAB_SETTINGS


def test_UpdateCatalogCollection():
    '''
    We don't really test much of anything here. Rather, we rely on the unit
    testing of the helper tasks that this task relies on. We should probably
    test better than this, but I'm too lazy right now.
    '''
    elements = ['Cu', 'Al']
    max_miller = 2
    task = UpdateCatalogCollection(elements=elements, max_miller=max_miller)

    req = task.requires()
    assert isinstance(req, _GetMpids)
    assert list(req.elements) == elements


def test__GetMpids():
    elements = set(['Cu', 'Al'])
    task = _GetMpids(elements=list(elements))

    # Run the task
    try:
        task.run()
        mpids = get_task_output(task)

        # For each MPID it enumerated, make sure the formation energy and
        # composition are correct
        with MPRester(read_rc('matproj_api_key')) as rester:
            for mpid in mpids:
                docs = rester.query({'task_id': mpid},
                                    ['elements', 'formation_energy_per_atom'])
                assert docs[0]['formation_energy_per_atom'] <= 0.
                for element in docs[0]['elements']:
                    assert element in elements

    finally:
        clean_up_tasks()


def test__InsertAllSitesFromBulkToCatalog():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the bulk calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `run_task_locally` appropriately.
    '''
    mpid = 'mp-2'
    max_miller = 2
    catalog_inserter = _InsertSitesToCatalog(mpid=mpid, max_miller=max_miller)
    site_generator = catalog_inserter.requires()

    try:
        run_task_locally(site_generator)
        site_docs = get_task_output(site_generator)

        catalog_inserter.run(_testing=True)
        catalog_docs = get_task_output(catalog_inserter)

        for site_doc, catalog_doc in zip(site_docs, catalog_docs):
            assert catalog_doc['mpid'] == mpid
            assert max(catalog_doc['miller']) <= max_miller
            assert catalog_doc['min_xy'] == site_generator.min_xy
            assert catalog_doc['slab_generator_settings'] == unfreeze_dict(site_generator.slab_generator_settings)
            assert catalog_doc['get_slab_settings'] == unfreeze_dict(site_generator.get_slab_settings)
            assert catalog_doc['bulk_vasp_settings'] == unfreeze_dict(site_generator.bulk_vasp_settings)
            assert catalog_doc['shift'] == site_doc['shift']
            assert catalog_doc['top'] == site_doc['top']
            assert make_atoms_from_doc(catalog_doc) == make_atoms_from_doc(site_doc)
            npt.assert_allclose(catalog_doc['slab_repeat'], site_doc['slab_repeat'])
            npt.assert_allclose(catalog_doc['adsorption_site'], site_doc['adsorption_site'])

    finally:
        clean_up_tasks()
