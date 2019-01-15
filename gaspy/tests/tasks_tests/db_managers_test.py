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
                                  _EnumerateDistinctFacets,
                                  _InsertFacetIntoCatalog)

# Things we need to do the tests
from itertools import combinations
import pickle
import numpy.testing as npt
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.structure_matcher import StructureMatcher
from .utils import clean_up_task
from ... import defaults
from ...atoms_operators import make_slabs_from_bulk_atoms
from ...utils import unfreeze_dict, read_rc
from ...mongo import make_atoms_from_doc
from ...tasks import get_task_output, evaluate_luigi_task

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
        clean_up_task(task)


def test__EnumerateDistinctFacets():
    '''
    We take all the facets that the task are distinct/unique, then actually
    make slabs out of them and compare all the slabs to see if they are
    identical. Note that this tests only if we get repeats. It does not
    test if we missed anything.

    WARNING:  This test uses `evaluate_luigi_task`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the gas calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `evaluate_luigi_task` appropriately.
    '''
    mpid = 'mp-2'
    max_miller = 2
    task = _EnumerateDistinctFacets(mpid=mpid, max_miller=max_miller)

    # Run the task to get the facets, and also get the bulk structure so we can
    # actually make slabs to check
    try:
        evaluate_luigi_task(task)
        distinct_millers = get_task_output(task)
        with open(task.input().path, 'rb') as file_handle:
            bulk_doc = pickle.load(file_handle)
        bulk_atoms = make_atoms_from_doc(bulk_doc)

        # Make all the slabs that the task said are distinct
        all_slabs = []
        for miller in distinct_millers:
            slabs = make_slabs_from_bulk_atoms(bulk_atoms,
                                               miller,
                                               SLAB_SETTINGS['slab_generator_settings'],
                                               SLAB_SETTINGS['get_slab_settings'],)
            all_slabs.extend(slabs)

        # Check that the slabs are actually different
        matcher = StructureMatcher()
        for slabs_to_compare in combinations(all_slabs, 2):
            assert not matcher.fit(*slabs_to_compare)

    finally:
        clean_up_task(task)


def test__InsertFacetIntoCatalog():
    '''
    WARNING:  This test uses `evaluate_luigi_task`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the bulk calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `evaluate_luigi_task` appropriately.
    '''
    mpid = 'mp-2'
    miller_indices = (1, 0, 0)
    catalog_inserter = _InsertFacetIntoCatalog(mpid=mpid, miller_indices=miller_indices)
    site_generator = catalog_inserter.requires()

    try:
        evaluate_luigi_task(site_generator)
        site_docs = get_task_output(site_generator)

        catalog_inserter.run(_testing=True)
        catalog_docs = get_task_output(catalog_inserter)

        for site_doc, catalog_doc in zip(site_docs, catalog_docs):
            assert catalog_doc['mpid'] == mpid
            assert catalog_doc['miller'] == miller_indices
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
        clean_up_task(site_generator)
        clean_up_task(catalog_inserter)
