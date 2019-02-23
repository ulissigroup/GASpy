'''
Tests for the `gasdb` submodule. Since this submodule deals with Mongo
databases so much, we created and work with various collections that are tagged
like `unit_testing_adsorption` or `unit_testing_atoms`. We assume that these
collections have very specific contents. We also assume that documents within
each collection should have the same exact JSON architecture.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ..gasdb import (get_mongo_collection,
                     ConnectableCollection,
                     get_adsorption_docs,
                     _clean_up_aggregated_docs,
                     get_catalog_docs,
                     _pull_catalog_from_mongo,
                     get_catalog_docs_with_predictions,
                     _add_adsorption_energy_predictions_to_projection,
                     _add_orr_predictions_to_projection,
                     #get_surface_docs,
                     get_unsimulated_catalog_docs,
                     _get_attempted_adsorption_docs,
                     _duplicate_docs_per_rotations,
                     _hash_doc,
                     get_low_coverage_docs,
                     get_low_coverage_dft_docs,
                     get_surface_from_doc,
                     get_low_coverage_ml_docs)

# Things we need to do the tests
import pytest
import warnings
import copy
import pickle
import random
import hashlib
from datetime import datetime
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from ..utils import read_rc
from ..defaults import catalog_projection, adslab_settings

REGRESSION_BASELINES_LOCATION = '/home/GASpy/gaspy/tests/regression_baselines/gasdb/'


@pytest.mark.parametrize('collection_tag', ['adsorption'])
def test_get_mongo_collection(collection_tag):
    collection = get_mongo_collection(collection_tag=collection_tag)

    # Make sure it's the correct class type
    assert isinstance(collection, ConnectableCollection)

    # Make sure it's connected and authenticated by counting the documents
    try:
        _ = collection.count_documents({})  # noqa: F841
    except OperationFailure:
        assert False


@pytest.mark.parametrize('collection_tag', ['adsorption'])
def test_ConnectableCollection(collection_tag):
    '''
    Verify that the extended `ConnectableCollection` class
    is still a Collection class and has the appropriate methods
    '''
    # Login info
    mongo_info = read_rc('mongo_info.%s' % collection_tag)
    host = mongo_info['host']
    port = int(mongo_info['port'])
    database_name = mongo_info['database']
    user = mongo_info['user']
    password = mongo_info['password']
    collection_name = mongo_info['collection_name']

    # Connect to the database/collection
    client = MongoClient(host=host, port=port)
    database = getattr(client, database_name)
    database.authenticate(user, password)
    collection = ConnectableCollection(database=database, name=collection_name)

    # Verify that the extended class is still a Connection class
    assert isinstance(collection, Collection)

    # Verify that the methods necessary for connection are present
    assert '__enter__' in dir(collection)
    assert '__exit__' in dir(collection)


@pytest.mark.filterwarnings('ignore:  You are using adsorption document filters '
                            'for a set of adsorbate that we have not yet '
                            'established valid energy bounds for, yet.')
@pytest.mark.parametrize('adsorbate, extra_projections',
                         [(None, None),
                          ('CO', None),
                          (None, {'adslab FWID': 'fwids.adslab'}),
                          ('H', {'adslab FWID': 'fwids.adslab'})])
def test_get_adsorption_docs(adsorbate, extra_projections):
    '''
    Currently not testing the `filters` argument because, well, I am being lazy.
    Feel free to change that yourself.
    '''
    # If there is no adsorbate, then we should be alerting the user
    if adsorbate is None:
        with pytest.warns(UserWarning, match='You are using adsorption document '
                          'filters for an adsorbate'):
            docs = get_adsorption_docs(adsorbate=adsorbate,
                                       extra_projections=extra_projections)
    # If they specify an adsorbate, then proceed as normal
    else:
        docs = get_adsorption_docs(adsorbate=adsorbate,
                                   extra_projections=extra_projections)

    assert len(docs) > 0
    for doc in docs:
        assert isinstance(doc['mongo_id'], ObjectId)
        assert isinstance(doc['adsorbate'], str)
        assert isinstance(doc['mpid'], str)
        assert len(doc['miller']) == 3
        assert all(isinstance(miller, int) for miller in doc['miller'])
        assert isinstance(doc['shift'], (float, int))
        assert isinstance(doc['top'], bool)
        assert isinstance(doc['coordination'], str)
        for neighbor in doc['neighborcoord']:
            assert isinstance(neighbor, str)
        assert isinstance(doc['neighborcoord'], list)
        assert isinstance(doc['energy'], float)
        if extra_projections is not None:
            for projection in extra_projections:
                assert projection in doc


def test__clean_up_aggregated_docs():
    docs = get_adsorption_docs('CO')
    dirty_docs = __make_documents_dirty(docs)
    clean_docs = _clean_up_aggregated_docs(dirty_docs, expected_keys=docs[0].keys())
    assert docs == clean_docs


def __make_documents_dirty(docs):
    ''' Helper function for `test__clean_up_aggregated_docs`  '''
    # Make a document with a missing key/value item
    doc_partial = copy.deepcopy(random.choice(docs))
    key_to_delete = random.choice(list(doc_partial.keys()))
    _ = doc_partial.pop(key_to_delete)   # noqa: F841

    # Make a document with a `None` value
    doc_empty0 = copy.deepcopy(random.choice(docs))
    key_to_modify = random.choice(list(doc_empty0.keys()))
    doc_empty0[key_to_modify] = None

    # Make a document with a '' value
    doc_empty1 = copy.deepcopy(random.choice(docs))
    key_to_modify = random.choice(list(doc_empty1.keys()))
    doc_empty1[key_to_modify] = ''

    # Make a document with no second shell atoms
    doc_empty2 = copy.deepcopy(random.choice(docs))
    doc_empty2['neighborcoord'] = ['Cu:', 'Al:']

    # Make the dirty documents
    dirty_docs = docs.copy() + [doc_partial, doc_empty0, doc_empty1, doc_empty2]
    return dirty_docs


def test_get_catalog_docs():
    docs = get_catalog_docs()
    for doc in docs:
        assert isinstance(doc['mongo_id'], ObjectId)
        assert isinstance(doc['mpid'], str)
        assert len(doc['miller']) == 3
        assert all(isinstance(miller, int) for miller in doc['miller'])
        assert isinstance(doc['shift'], (float, int))
        assert isinstance(doc['top'], bool)
        assert isinstance(doc['natoms'], int)
        assert isinstance(doc['coordination'], str)
        for neighbor in doc['neighborcoord']:
            assert isinstance(neighbor, str)
        assert len(doc['adsorption_site']) == 3
        assert all(isinstance(coordinate, float) for coordinate in doc['adsorption_site'])


def test__pull_catalog_from_mongo():
    projection = catalog_projection()
    project = {'$project': projection}
    pipeline = [project]
    docs = _pull_catalog_from_mongo(pipeline)

    assert len(docs) > 0
    for doc in docs:
        for key in projection:
            try:
                assert key in doc

            # Ignore this projection that tells us to omit _id
            except AssertionError:
                if key == '_id':
                    pass
                else:
                    raise


@pytest.mark.parametrize('latest_predictions', [True, False])
def test_get_catalog_docs_with_predictions(latest_predictions):
    docs = get_catalog_docs_with_predictions(latest_predictions=latest_predictions)

    for doc in docs:
        assert 'adsorption_energy' in doc['predictions']
        assert 'orr_onset_potential_4e' in doc['predictions']

        if latest_predictions is True:
            energy_prediction = doc['predictions']['adsorption_energy']['CO']['model0']
            assert isinstance(energy_prediction[0], datetime)
            assert isinstance(energy_prediction[1], float)
            orr_prediction = doc['predictions']['orr_onset_potential_4e']['model0']
            assert isinstance(orr_prediction[0], datetime)
            assert isinstance(orr_prediction[1], float)


@pytest.mark.parametrize('latest_predictions', [True, False])
def test__add_adsorption_energy_predictions_to_projections(latest_predictions):
    default_projections = catalog_projection()
    projections = _add_adsorption_energy_predictions_to_projection(default_projections, latest_predictions)

    # Get ALL of the adsorbates and models in the unit testing collection
    with get_mongo_collection('catalog') as collection:
        cursor = collection.aggregate([{"$sample": {"size": 1}}])
        docs = list(cursor)
    adsorbates = set()
    models = set()
    for doc in docs:
        predictions = doc['predictions']['adsorption_energy']
        new_adsorbates = set(predictions.keys())
        new_models = set(model for adsorbate in adsorbates for model in predictions[adsorbate])
        adsorbates.update(new_adsorbates)
        models.update(new_models)

    # Make sure that every single query is there
    for adsorbate in adsorbates:
        for model in models:
            data_location = 'predictions.adsorption_energy.%s.%s' % (adsorbate, model)
            if latest_predictions:
                assert projections[data_location] == {'$arrayElemAt': ['$'+data_location, -1]}
            else:
                assert projections[data_location] == '$'+data_location


@pytest.mark.parametrize('latest_predictions', [True, False])
def test__add_orr_predictions_to_projections(latest_predictions):
    default_projections = catalog_projection()
    projections = _add_orr_predictions_to_projection(default_projections, latest_predictions)

    # Get ALL of the models in the unit testing collection
    with get_mongo_collection('catalog') as collection:
        cursor = collection.aggregate([{"$sample": {"size": 1}}])
        docs = list(cursor)
    models = set()
    for doc in docs:
        predictions = doc['predictions']['orr_onset_potential_4e']
        new_models = set(predictions.keys())
        models.update(new_models)

    # Make sure that every single query is there
    for model in models:
        data_location = 'predictions.orr_onset_potential_4e.%s' % model
        if latest_predictions:
            assert projections[data_location] == {'$arrayElemAt': ['$'+data_location, -1]}
        else:
            assert projections[data_location] == '$'+data_location


@pytest.mark.baseline
@pytest.mark.parametrize('adsorbate, adsorbate_rotation_list',
                         [('H', [{'phi': 0., 'theta': 0., 'psi': 0.}]),
                          ('CO', [{'phi': 0., 'theta': 0., 'psi': 0.},
                                  {'phi': 0., 'theta': 30., 'psi': 0.}])])
def test_to_create_unsimulated_catalog_docs(adsorbate, adsorbate_rotation_list):
    docs = get_unsimulated_catalog_docs(adsorbate=adsorbate,
                                        adsorbate_rotation_list=adsorbate_rotation_list)

    arg_hash = hashlib.sha224((str(adsorbate) + str(adsorbate_rotation_list)).encode()).hexdigest()
    file_name = REGRESSION_BASELINES_LOCATION + 'unsimulated_catalog_docs_%s' % arg_hash + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(docs, file_handle)
    assert True


@pytest.mark.parametrize('adsorbate, adsorbate_rotation_list',
                         [('H', [{'phi': 0., 'theta': 0., 'psi': 0.}]),
                          ('CO', [{'phi': 0., 'theta': 0., 'psi': 0.},
                                  {'phi': 0., 'theta': 30., 'psi': 0.}])])
def test_get_unsimulated_catalog_docs(adsorbate, adsorbate_rotation_list):
    arg_hash = hashlib.sha224((str(adsorbate) + str(adsorbate_rotation_list)).encode()).hexdigest()
    file_name = REGRESSION_BASELINES_LOCATION + 'unsimulated_catalog_docs_%s' % arg_hash + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_docs = pickle.load(file_handle)

    docs = get_unsimulated_catalog_docs(adsorbate=adsorbate,
                                        adsorbate_rotation_list=adsorbate_rotation_list)
    assert docs == expected_docs


def test__duplicate_docs_per_rotation():
    docs = [dict.fromkeys(range(i)) for i in range(10)]
    rotation_list = [{'phi': 0., 'theta': 0., 'psi': 0.},
                     {'phi': 180., 'theta': 180., 'psi': 180.}]
    duplicated_docs = _duplicate_docs_per_rotations(docs, rotation_list)

    # Check that we have enough documents
    assert len(rotation_list)*len(docs) == len(duplicated_docs)

    # Check that the rotations have been added correctly
    for i, expected_rotation in enumerate(rotation_list):
        start_slice = len(docs) * i
        end_slice = len(docs) * (i+1)
        for doc in duplicated_docs[start_slice:end_slice]:
            assert doc['adsorbate_rotation'] == expected_rotation

    # Check that we can handle a single rotation correctly, too
    rotation_list = [{'phi': 0., 'theta': 0., 'psi': 0.}]
    duplicated_docs = _duplicate_docs_per_rotations(docs, rotation_list)
    assert len(rotation_list)*len(docs) == len(duplicated_docs)
    for doc in duplicated_docs:
        assert doc['adsorbate_rotation'] == rotation_list[0]


@pytest.mark.parametrize('adsorbate', ['H', 'CO'])
def test__get_attempted_adsorption_docs(adsorbate):
    attempted_docs = _get_attempted_adsorption_docs(adsorbate=adsorbate)

    filters = {'vasp_settings.%s' % setting: value
               for setting, value in adslab_settings()['vasp'].items()}
    all_docs = get_adsorption_docs(adsorbate=adsorbate, filters=filters)
    assert len(attempted_docs) == len(all_docs)


@pytest.mark.baseline
@pytest.mark.parametrize('ignore_keys', [None, ['mpid'], ['mpid', 'top']])
def test_to_create_hashed_doc(ignore_keys):
    # GASpy is going to yell at us about finding documents with no adsorbates.
    # Ignore it.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)

        doc = get_adsorption_docs()[0]
    string = _hash_doc(doc=doc, ignore_keys=ignore_keys, _return_hash=False)

    # EAFP to find the pickle name of the cache, since it'll be odd if adsorbate == None
    try:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_doc_ignoring_' + '_'.join(ignore_keys) + '.pkl'
    except TypeError:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_doc_ignoring_nothing.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(string, file_handle)


@pytest.mark.parametrize('ignore_keys', [None, ['mpid'], ['mpid', 'top']])
def test__hash_doc(ignore_keys):
    '''
    Note that since Python 3's `hash` returns a different hash for
    each instance of Python, we actually perform regression testing
    on the pre-hashed string, not the hash itself.
    '''
    # EAFP to find the pickle name of the cache, since it'll be odd if adsorbate == None
    try:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_doc_ignoring_' + '_'.join(ignore_keys) + '.pkl'
    except TypeError:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_doc_ignoring_nothing.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_string = pickle.load(file_handle)

    # GASpy is going to yell at us about finding documents with no adsorbates.
    # Ignore it.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        doc = get_adsorption_docs()[0]
    string = _hash_doc(doc=doc, ignore_keys=ignore_keys, _return_hash=False)
    assert string == expected_string


@pytest.mark.parametrize('adsorbate, model_tag',
                         [('H', 'model0'),
                          ('CO', 'model0')])
def test_get_low_coverage_docs(adsorbate, model_tag):
    '''
    We test `get_low_coverage_docs_by_surface' in multiple unit tests.
    This test verifies that the documents provided by this function are
    actually have the minimum energy within each surface.
    '''
    low_coverage_docs = get_low_coverage_docs(adsorbate, model_tag)
    docs_dft = get_low_coverage_dft_docs(adsorbate)
    docs_ml = get_low_coverage_ml_docs(adsorbate, model_tag)
    docs_dft_by_surface = {get_surface_from_doc: doc for doc in docs_dft}
    docs_ml_by_surface = {get_surface_from_doc: doc for doc in docs_ml}

    for doc in low_coverage_docs:
        surface = get_surface_from_doc(doc)

        if doc['DFT_calculated'] is True:
            try:
                doc_ml = docs_ml_by_surface[surface]
                assert doc['energy'] <= doc_ml['energy']
            except KeyError:
                continue

        elif doc['DFT_calculated'] is False:
            try:
                doc_dft = docs_dft_by_surface[surface]
                assert doc['energy'] <= doc_dft['energy']
            except KeyError:
                continue


@pytest.mark.parametrize('adsorbate', ['H', 'CO'])
def test_get_low_coverage_dft_docs(adsorbate):
    '''
    For each surface, verify that every single document in our adsorption
    collection has a higher (or equal) adsorption energy than the one reported
    by `get_low_coverage_dft_docs`
    '''
    low_coverage_docs = get_low_coverage_dft_docs(adsorbate)
    low_coverage_docs_by_surface = {get_surface_from_doc(doc): doc
                                    for doc in low_coverage_docs}
    all_docs = get_adsorption_docs(adsorbate)

    for doc in all_docs:
        surface = get_surface_from_doc(doc)
        energy = doc['energy']
        low_cov_energy = low_coverage_docs_by_surface[surface]['energy']
        assert low_cov_energy <= energy


def test_get_surface_from_doc():
    doc = {'mpid': 'mp-23',
           'miller': [1, 0, 0],
           'shift': 0.001,
           'top': True}
    surface = get_surface_from_doc(doc)

    expected_surface = ('mp-23', '[1, 0, 0]', 0., True)
    assert surface == expected_surface


@pytest.mark.parametrize('adsorbate, model_tag',
                         [('H', 'model0'),
                          ('CO', 'model0')])
def test_get_low_coverage_ml_docs(adsorbate, model_tag):
    '''
    For each surface, verify that every single document in
    our adsorption collection has a higher (or equal) adsorption
    energy than the one reported by `get_low_coverage_ml_docs`
    '''
    low_coverage_docs = get_low_coverage_ml_docs(adsorbate)
    low_coverage_docs_by_surface = {get_surface_from_doc(doc): doc
                                    for doc in low_coverage_docs}
    all_docs = get_catalog_docs_with_predictions()

    for doc in all_docs:
        energy = doc['predictions']['adsorption_energy'][adsorbate][model_tag][1]
        surface = get_surface_from_doc(doc)
        low_cov_energy = low_coverage_docs_by_surface[surface]['energy']
        assert low_cov_energy <= energy


#@pytest.mark.baseline
#@pytest.mark.parametrize('extra_fingerprints', [None, {'user': 'user'}])
#def test_to_create_aggregated_surface_documents(extra_fingerprints):
#    try:
#        file_name = REGRESSION_BASELINES_LOCATION + 'aggregated_surface_documents_' + \
#            '_'.join(list(extra_fingerprints.keys())) + '.pkl'
#    except AttributeError:
#        file_name = REGRESSION_BASELINES_LOCATION + 'aggregated_surface_documents_' + '.pkl'
#
#    docs = get_surface_docs(extra_fingerprints)
#    with open(file_name, 'wb') as file_handle:
#        pickle.dump(docs, file_handle)
#    assert True
#
#
#@pytest.mark.parametrize('extra_fingerprints', [None, {'user': 'user'}])
#def test_get_surface_docs(extra_fingerprints):
#    '''
#    Currently not testing the "filters" argument because, well, I am being lazy.
#    Feel free to change that yourself.
#    '''
#    # EAFP to set the file name; depends on whether or not there are extra fingerprints
#    try:
#        file_name = (REGRESSION_BASELINES_LOCATION + 'aggregated_surface_documents_' +
#                     '_'.join(list(extra_fingerprints.keys())) + '.pkl')
#    except AttributeError:
#        file_name = REGRESSION_BASELINES_LOCATION + 'aggregated_surface_documents_' + '.pkl'
#
#    with open(file_name, 'rb') as file_handle:
#        expected_docs = pickle.load(file_handle)
#    docs = get_surface_docs(extra_fingerprints)
#    assert docs == expected_docs
