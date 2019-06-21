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
                                              __run_calculate_adsorption_energy_task,
                                              __clean_calc_energy_docs,
                                              __create_adsorption_doc)

# Things we need to do the tests
import ase
from ..utils import clean_up_tasks
from ...test_cases.mongo_test_collections.mongo_utils import populate_unit_testing_collection
from ....mongo import make_atoms_from_doc
from ....gasdb import get_mongo_collection


def test_update_adsorption_collection():
    '''
    This may seem like a crappy test, but we really rely on unit testing of the
    sub-functions for integrity testing.
    '''
    try:
        # Clear out the adsorption collection before testing
        with get_mongo_collection('adsorption') as collection:
            collection.delete_many({})

        # See if we can actually add anything
        update_adsorption_collection()
        with get_mongo_collection('adsorption') as collection:
            assert collection.count_documents({}) > 0

    # Reset the adsorption collection behind us
    finally:
        with get_mongo_collection('adsorption') as collection:
            collection.delete_many({})
        populate_unit_testing_collection('adsorption')


def test__find_atoms_docs_not_in_adsorption_collection():
    docs = _find_atoms_docs_not_in_adsorption_collection()

    # Make sure these "documents" are actually `atoms` docs that can be turned
    # into `ase.Atoms` objects
    for doc in docs:
        atoms = make_atoms_from_doc(doc)
        assert isinstance(atoms, ase.Atoms)

    # Make sure that everything we found actually isn't in our `adsorption`
    # collection
    fwids = [doc['fwid'] for doc in docs]
    with get_mongo_collection('adsorption') as collection:
        ads_docs = list(collection.find({'fwids.adsorption': {'$in': fwids}}, {'_id': 1}))
    assert len(ads_docs) == 0


def test___run_calculate_adsorption_energy_task():
    with get_mongo_collection('atoms') as collection:
        query = {'fwname.calculation_type': 'slab+adsorbate optimization',
                 'fwname.adsorbate': {'$ne': ''}}
        adslab_docs = list(collection.find(query))
        try:
            energy_docs = [__run_calculate_adsorption_energy_task(doc)
                           for doc in adslab_docs]
        finally:
            clean_up_tasks()

        # Verify that the documents have adsorption energies
        for doc in energy_docs:
            assert isinstance(doc['adsorption_energy'], float)

            # Verify that the slabs and adslabs within our documents are from
            # our `atoms` collection
            slab_fwid = doc['fwids']['slab']
            adslab_fwid = doc['fwids']['adslab']
            assert len(list(collection.find({'fwid': slab_fwid}))) == 1
            assert len(list(collection.find({'fwid': adslab_fwid}))) == 1


def test___clean_calc_energy_docs():
    docs = [None,
            {'fwids': {'adslab': 2}},
            {'fwids': {'adslab': 3}},
            {'fwids': {'adslab': 2}}]
    missing_docs = [{'fwid': 2}, {'fwid': 3}]
    clean_docs = __clean_calc_energy_docs(docs, missing_docs)
    assert clean_docs == docs[1:3]


def test___create_adsorption_doc():
    with get_mongo_collection('atoms') as collection:
        query = {'fwname.calculation_type': 'slab+adsorbate optimization',
                 'fwname.adsorbate': {'$ne': ''}}
        adslab_doc = list(collection.find(query))[0]
    try:
        energy_doc = __run_calculate_adsorption_energy_task(adslab_doc)
    finally:
        clean_up_tasks()
    doc = __create_adsorption_doc(energy_doc)

    assert isinstance(make_atoms_from_doc(doc), ase.Atoms)
    assert isinstance(make_atoms_from_doc(doc['initial_configuration']), ase.Atoms)
    assert isinstance(doc['adsorbate'], str)
    assert set(doc['adsorbate_rotation'].keys()) == set(['phi', 'theta', 'psi'])
    assert all(isinstance(angle, float) for angle in doc['adsorbate_rotation'].values())
    assert all(isinstance(coord, float) for coord in doc['initial_adsorption_site'])
    assert isinstance(doc['mpid'], str)
    assert len(doc['miller']) == 3
    assert isinstance(doc['shift'], float) or isinstance(doc['shift'], int)
    assert isinstance(doc['top'], bool)
    # `slabrepeat` should be a tuple of ints, but historically it was made as a string
    assert all(isinstance(repeat, int) for repeat in doc['slab_repeat'])
    assert isinstance(doc['dft_settings'], dict)
    assert all(isinstance(fwid, int) for fwid in doc['fwids'].values())
    assert 'fp_init' in doc
    assert 'fp_final' in doc
    assert all(isinstance(datum, float) for datum in doc['movement_data'].values())
    assert all(isinstance(date, str) for date in doc['calculation_dates'].values())
