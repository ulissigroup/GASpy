'''
Tests for the `gasdb` submodule. Since this submodule deals with Mongo
databases so much, we created and work with various collections that are tagged
with `unit_testing_adsorption`. We assume that these collections have very
specific contents. We also assume that documents within each collection should
have the same exact JSON architecture.
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
                     get_catalog_docs_with_predictions,
                     get_surface_docs,
                     get_unsimulated_catalog_docs,
                     _get_attempted_adsorption_docs,
                     _duplicate_docs_per_rotations,
                     _hash_docs,
                     _hash_doc,
                     get_low_coverage_docs,
                     get_low_coverage_dft_docs,
                     _get_surface_from_doc,
                     get_low_coverage_ml_docs,
                     remove_duplicates_in_adsorption_collection)

# Things we need to do the tests
import pytest
import copy
import pickle
import random
import binascii
import hashlib
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from ..utils import read_rc

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


@pytest.mark.baseline
@pytest.mark.parametrize('adsorbates, extra_fingerprints',
                         [(None, None),
                          (['CO'], None),
                          (None, {'adslab FWID': 'processed.data.FW_info.slab+adsorbate'}),
                          (['H'], {'adslab FWID': 'processed.data.FW_info.slab+adsorbate'})])
def test_to_create_expected_aggregated_adsorption_documents(adsorbates, extra_fingerprints):
    docs = get_adsorption_docs(adsorbates=adsorbates, extra_fingerprints=extra_fingerprints)
    file_name = __get_file_name_from_adsorbates_and_extra_fingerprints(adsorbates, extra_fingerprints)
    with open(file_name, 'wb') as file_handle:
        pickle.dump(docs, file_handle)
    assert True


def get_expected_aggregated_adsorption_documents(adsorbates=None, extra_fingerprints=None):
    file_name = __get_file_name_from_adsorbates_and_extra_fingerprints(adsorbates, extra_fingerprints)
    with open(file_name, 'rb') as file_handle:
        docs = pickle.load(file_handle)
    return docs


def __get_file_name_from_adsorbates_and_extra_fingerprints(adsorbates, extra_fingerprints):
    '''
    We need to save pickle caches for some test results with different parameters.
    This is a helper function to establish what these files will be called.
    '''
    # Turn `extra_fingerprints` into a hex string. EAFP to deal with nested vs. flat fingerprints
    try:
        fps_as_str = str(list(extra_fingerprints.keys())) + str(list(extra_fingerprints.values()))
    except AttributeError:
        fps_as_str = ''
    fps_as_hex = binascii.hexlify(fps_as_str.encode()).hex()
    fps_as_hex = fps_as_hex[:5] + fps_as_hex[-5:]   # Hackly get around an issue around too-long-of-a-file-name

    # Turn `adsorbates` into a string. EAFP to deal with `None`
    try:
        ads_as_str = '_'.join(adsorbates)
    except TypeError:
        ads_as_str = ''

    # Concatenate inputs
    inputs_as_str = ads_as_str + '_' + fps_as_hex
    file_name = REGRESSION_BASELINES_LOCATION + 'aggregated_adsorption_documents_' + inputs_as_str + '.pkl'
    return file_name


@pytest.mark.filterwarnings('ignore:  You are using adsorption document filters '
                            'for a set of adsorbates that we have not yet '
                            'established valid energy bounds for, yet.')
@pytest.mark.parametrize('adsorbates, extra_fingerprints',
                         [(None, None),
                          (['CO'], None),
                          (None, {'adslab FWID': 'processed.data.FW_info.slab+adsorbate'}),
                          (['H'], {'adslab FWID': 'processed.data.FW_info.slab+adsorbate'})])
def test_get_adsorption_docs(adsorbates, extra_fingerprints):
    '''
    Currently not testing the "filters" argument because, well, I am being lazy.
    Feel free to change that yourself.
    '''
    expected_docs = get_expected_aggregated_adsorption_documents(adsorbates=adsorbates,
                                                                 extra_fingerprints=extra_fingerprints)
    docs = get_adsorption_docs(adsorbates=adsorbates, extra_fingerprints=extra_fingerprints)
    assert _compare_unordered_sequences(docs, expected_docs)


def _compare_unordered_sequences(seq0, seq1):
    '''
    If (1) we want to see if two sequences are identical, (2) do not care about ordering,
    (3) the items are not hashable, and (4) items are not orderable, then use this
    function to compare them. Credit goes to CrowbarKZ on StackOverflow.
    '''
    # A hack to ignore mongo IDs, which we don't _really_ care about...
    seq0 = __remove_mongo_ids_from_docs(seq0)
    seq1 = __remove_mongo_ids_from_docs(seq1)

    try:
        for element in seq1:
            seq0.remove(element)
    except ValueError:
        return False
    return not seq0


def __remove_mongo_ids_from_docs(docs):
    ''' Helper function to remove the 'mongo_id' key:value pairing from a list of dictionaries '''
    new_docs = []
    for doc in docs:
        try:
            del doc['mongo_id']
        except KeyError:
            pass
        new_docs.append(doc)
    return new_docs


def test__clean_up_aggregated_docs():
    expected_docs = get_expected_aggregated_adsorption_documents()
    dirty_docs = __make_documents_dirty(expected_docs)
    clean_docs = _clean_up_aggregated_docs(dirty_docs, expected_keys=expected_docs[0].keys())
    assert _compare_unordered_sequences(clean_docs, expected_docs)


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


@pytest.mark.baseline
def test_to_create_aggregated_catalog_documents():
    docs = get_catalog_docs()
    with open(REGRESSION_BASELINES_LOCATION + 'aggregated_catalog_documents' + '.pkl', 'wb') as file_handle:
        pickle.dump(docs, file_handle)
    assert True


def get_expected_aggregated_catalog_documents():
    with open(REGRESSION_BASELINES_LOCATION + 'aggregated_catalog_documents' + '.pkl', 'rb') as file_handle:
        docs = pickle.load(file_handle)
    return docs


def test_get_catalog_docs():
    expected_docs = get_expected_aggregated_catalog_documents()
    docs = get_catalog_docs()
    assert docs == expected_docs


@pytest.mark.baseline
@pytest.mark.parametrize('adsorbates, models, latest_predictions',
                         [(['CO'], ['model0'], True),
                          (['CO', 'H'], ['model0'], True),
                          (['CO', 'H'], ['model0'], False)])
def test_to_create_catalog_docs_with_predictions(adsorbates, models, latest_predictions):
    docs = get_catalog_docs_with_predictions(adsorbates, models, latest_predictions)

    arg_hash = hashlib.sha224((str(adsorbates) + str(models) + str(latest_predictions)).encode()).hexdigest()
    file_name = REGRESSION_BASELINES_LOCATION + 'catalog_predictions_%s' % arg_hash + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(docs, file_handle)
    assert True


@pytest.mark.parametrize('adsorbates, models, latest_predictions',
                         [(['CO'], ['model0'], True),
                          (['CO', 'H'], ['model0'], True),
                          (['CO', 'H'], ['model0'], False)])
def test_get_catalog_docs_with_predictions(adsorbates, models, latest_predictions):
    '''
    This could be a "real" test, but I am really busy and don't have time to design one.
    So I'm turning this into a regression test to let someone else (probably me)
    deal with this later.

    If you do fix this, you should probably add more than one day's worth of predictions
    to the unit testing catalog.
    '''
    docs = get_catalog_docs_with_predictions(adsorbates, models, latest_predictions)

    arg_hash = hashlib.sha224((str(adsorbates) + str(models) + str(latest_predictions)).encode()).hexdigest()
    file_name = REGRESSION_BASELINES_LOCATION + 'catalog_predictions_%s' % arg_hash + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_docs = pickle.load(file_handle)
    assert docs == expected_docs


@pytest.mark.baseline
@pytest.mark.parametrize('extra_fingerprints', [None, {'user': 'user'}])
def test_to_create_aggregated_surface_documents(extra_fingerprints):
    try:
        file_name = REGRESSION_BASELINES_LOCATION + 'aggregated_surface_documents_' + \
            '_'.join(list(extra_fingerprints.keys())) + '.pkl'
    except AttributeError:
        file_name = REGRESSION_BASELINES_LOCATION + 'aggregated_surface_documents_' + '.pkl'

    docs = get_surface_docs(extra_fingerprints)
    with open(file_name, 'wb') as file_handle:
        pickle.dump(docs, file_handle)
    assert True


@pytest.mark.parametrize('extra_fingerprints', [None, {'user': 'user'}])
def test_get_surface_docs(extra_fingerprints):
    '''
    Currently not testing the "filters" argument because, well, I am being lazy.
    Feel free to change that yourself.
    '''
    # EAFP to set the file name; depends on whether or not there are extra fingerprints
    try:
        file_name = (REGRESSION_BASELINES_LOCATION + 'aggregated_surface_documents_' +
                     '_'.join(list(extra_fingerprints.keys())) + '.pkl')
    except AttributeError:
        file_name = REGRESSION_BASELINES_LOCATION + 'aggregated_surface_documents_' + '.pkl'

    with open(file_name, 'rb') as file_handle:
        expected_docs = pickle.load(file_handle)
    docs = get_surface_docs(extra_fingerprints)
    assert docs == expected_docs


@pytest.mark.baseline
@pytest.mark.parametrize('adsorbates, adsorbate_rotation_list',
                         [(['H'], [{'phi': 0., 'theta': 0., 'psi': 0.}]),
                          (['CO'], [{'phi': 0., 'theta': 0., 'psi': 0.},
                                    {'phi': 0., 'theta': 30., 'psi': 0.}])])
def test_to_create_unsimulated_catalog_docs(adsorbates, adsorbate_rotation_list):
    docs = get_unsimulated_catalog_docs(adsorbates=adsorbates,
                                        adsorbate_rotation_list=adsorbate_rotation_list)

    arg_hash = hashlib.sha224((str(adsorbates) + str(adsorbate_rotation_list)).encode()).hexdigest()
    file_name = REGRESSION_BASELINES_LOCATION + 'unsimulated_catalog_docs_%s' % arg_hash + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(docs, file_handle)
    assert True


@pytest.mark.parametrize('adsorbates, adsorbate_rotation_list',
                         [(['H'], [{'phi': 0., 'theta': 0., 'psi': 0.}]),
                          (['CO'], [{'phi': 0., 'theta': 0., 'psi': 0.},
                                    {'phi': 0., 'theta': 30., 'psi': 0.}])])
def test_get_unsimulated_catalog_docs(adsorbates, adsorbate_rotation_list):
    arg_hash = hashlib.sha224((str(adsorbates) + str(adsorbate_rotation_list)).encode()).hexdigest()
    file_name = REGRESSION_BASELINES_LOCATION + 'unsimulated_catalog_docs_%s' % arg_hash + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_docs = pickle.load(file_handle)

    docs = get_unsimulated_catalog_docs(adsorbates=adsorbates,
                                        adsorbate_rotation_list=adsorbate_rotation_list)
    assert docs == expected_docs


@pytest.mark.baseline
@pytest.mark.parametrize('adsorbates', [None, ['H'], ['CO']])
def test_to_create_attempted_adsorption_docs(adsorbates):
    docs = _get_attempted_adsorption_docs(adsorbates=adsorbates)

    # EAFP to tag the pickle, since it'll be odd if adsorbates == None
    try:
        file_name = REGRESSION_BASELINES_LOCATION + 'attempted_' + '_'.join(adsorbates) + '_adsorption_docs' + '.pkl'
    except TypeError:
        file_name = REGRESSION_BASELINES_LOCATION + 'attempted_' + 'all' + '_adsorption_docs' + '.pkl'

    with open(file_name, 'wb') as file_handle:
        pickle.dump(docs, file_handle)


@pytest.mark.parametrize('adsorbates', [None, ['H'], ['CO']])
def test__get_attempted_adsorption_docs(adsorbates):
    '''
    The expected documents in here should differ from the ones in
    `get_expected_aggregated_adsorption_documents` because these ones
    use initial fingerprints, not final fingerprints.
    '''
    # EAFP to find the pickle name, since it'll be odd if adsorbates == None
    try:
        file_name = REGRESSION_BASELINES_LOCATION + 'attempted_' + '_'.join(adsorbates) + '_adsorption_docs' + '.pkl'
    except TypeError:
        file_name = REGRESSION_BASELINES_LOCATION + 'attempted_' + 'all' + '_adsorption_docs' + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_docs = pickle.load(file_handle)

    docs = _get_attempted_adsorption_docs(adsorbates=adsorbates)
    #assert docs == expected_docs
    assert _compare_unordered_sequences(docs, expected_docs)


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


@pytest.mark.baseline
@pytest.mark.parametrize('ignore_keys', [None, ['mpid'], ['mpid', 'top']])
def test_to_create_hashed_docs(ignore_keys):
    docs = get_expected_aggregated_adsorption_documents()
    strings = _hash_docs(docs=docs, ignore_keys=ignore_keys, _return_hashes=False)

    # EAFP to find the pickle name of the cache, since it'll be odd if adsorbates == None
    try:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_docs_ignoring_' + '_'.join(ignore_keys) + '.pkl'
    except TypeError:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_docs_ignoring_nothing.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(strings, file_handle)


@pytest.mark.parametrize('ignore_keys', [None, ['mpid'], ['mpid', 'top']])
def test__hash_docs(ignore_keys):
    '''
    Note that since Python 3's `hash` returns a different hash for
    each instance of Python, we actually perform regression testing
    on the serialized documents, not the hash itself.
    '''
    # EAFP to find the pickle name of the cache, since it'll be odd if adsorbates == None
    try:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_docs_ignoring_' + '_'.join(ignore_keys) + '.pkl'
    except TypeError:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_docs_ignoring_nothing.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_serialized_doc = pickle.load(file_handle)

    docs = get_expected_aggregated_adsorption_documents()
    serialized_doc = _hash_docs(docs=docs, ignore_keys=ignore_keys, _return_hashes=False)
    assert serialized_doc == expected_serialized_doc


@pytest.mark.baseline
@pytest.mark.parametrize('ignore_keys', [None, ['mpid'], ['mpid', 'top']])
def test_to_create_hashed_doc(ignore_keys):
    doc = get_expected_aggregated_adsorption_documents()[0]
    string = _hash_doc(doc=doc, ignore_keys=ignore_keys, _return_hash=False)

    # EAFP to find the pickle name of the cache, since it'll be odd if adsorbates == None
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
    ond the pre-hashed string, not the hash itself.
    '''
    # EAFP to find the pickle name of the cache, since it'll be odd if adsorbates == None
    try:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_doc_ignoring_' + '_'.join(ignore_keys) + '.pkl'
    except TypeError:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_doc_ignoring_nothing.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_string = pickle.load(file_handle)

    doc = get_expected_aggregated_adsorption_documents()[0]
    string = _hash_doc(doc=doc, ignore_keys=ignore_keys, _return_hash=False)
    assert string == expected_string


@pytest.mark.parametrize('adsorbates, model_tag',
                         [(['H'], 'model0'),
                          (['CO'], 'model0')])
def test_surfaces_from_get_low_coverage_docs(adsorbates, model_tag):
    '''
    We test `get_low_coverage_docs_by_surface' in multiple unit tests.
    This test checks that the surfaces we find are correct.
    '''
    low_coverage_docs = get_low_coverage_docs(adsorbates, model_tag)
    surfaces = set(low_coverage_docs.keys())

    adsorption_docs = get_adsorption_docs(adsorbates)
    catalog_docs = get_unsimulated_catalog_docs(adsorbates)
    expected_surfaces = set((doc['mpid'], str(doc['miller']), round(doc['shift'], 2), doc['top'])
                            for doc in adsorption_docs + catalog_docs)
    assert surfaces == expected_surfaces


@pytest.mark.parametrize('adsorbates, model_tag',
                         [(['H'], 'model0'),
                          (['CO'], 'model0')])
def test_energies_from_get_low_coverage_docs(adsorbates, model_tag):
    '''
    We test `get_low_coverage_docs_by_surface' in multiple unit tests.
    This test verifies that the documents provided by this function are
    actually have the minimum energy within each surface.
    '''
    low_coverage_docs = get_low_coverage_docs(adsorbates, model_tag)
    docs_dft = get_low_coverage_dft_docs(adsorbates)
    docs_ml = get_low_coverage_ml_docs(adsorbates, model_tag)

    for surface, doc in low_coverage_docs.items():

        if doc['DFT_calculated']:
            try:
                doc_ml = docs_ml[surface]
                assert doc['energy'] <= doc_ml['energy']
            except KeyError:
                continue

        else:
            try:
                doc_dft = docs_dft[surface]
                assert doc['energy'] <= doc_dft['energy']
            except KeyError:
                continue


@pytest.mark.parametrize('adsorbates', [['H'], ['CO']])
def test_get_low_coverage_dft_docs(adsorbates):
    '''
    For each surface, verify that every single document in
    our adsorption collection has a higher (or equal) adsorption
    energy than the one reported by `get_low_coverage_dft_docs`
    '''
    low_coverage_docs = get_low_coverage_dft_docs(adsorbates)
    all_docs = get_adsorption_docs(adsorbates)

    for doc in all_docs:
        energy = doc['energy']

        surface = _get_surface_from_doc(doc)
        low_cov_energy = low_coverage_docs[surface]['energy']
        assert low_cov_energy <= energy


def test__get_surface_from_doc():
    doc = {'mpid': 'mp-23',
           'miller': [1, 0, 0],
           'shift': 0.001,
           'top': True}
    surface = _get_surface_from_doc(doc)

    expected_surface = ('mp-23', '[1, 0, 0]', 0., True)
    assert surface == expected_surface


@pytest.mark.parametrize('adsorbates, model_tag',
                         [(['H'], 'model0'),
                          (['CO'], 'model0')])
def test_get_low_coverage_ml_docs(adsorbates, model_tag):
    '''
    For each surface, verify that every single document in
    our adsorption collection has a higher (or equal) adsorption
    energy than the one reported by `get_low_coverage_ml_docs`
    '''
    models = ['model0']
    low_coverage_docs = get_low_coverage_ml_docs(adsorbates)
    all_docs = get_catalog_docs_with_predictions(adsorbates, models)

    for doc in all_docs:
        energy = doc['predictions']['adsorption_energy'][adsorbates[0]][model_tag][1]
        surface = _get_surface_from_doc(doc)
        low_cov_energy = low_coverage_docs[surface]['energy']
        assert low_cov_energy <= energy


def test_remove_duplicates_in_adsorption_collection():
    collection_tag = 'adsorption'
    starting_doc_count, id_of_duplicate_doc = add_duplicate_document_to_collection(collection_tag)

    try:
        remove_duplicates_in_adsorption_collection()

        # Verify that removal worked
        with get_mongo_collection(collection_tag=collection_tag) as collection:
            current_doc_count = collection.count_documents({})
        assert current_doc_count == starting_doc_count

    # Clean up the extra document if the function failed to delete it
    except:     # noqa: E722
        with get_mongo_collection(collection_tag=collection_tag) as collection:
            collection.delete_one({'_id': {'$eq': id_of_duplicate_doc}})
        raise


def add_duplicate_document_to_collection(collection_tag):
    '''
    Pick a random document in the collection, then add it to the collection

    Arg:
        collection_tag  String indicating which collection you'll be adding a duplicate to
    Output:
        starting_count  How many documents the collection starts with before the duplicate is added
        id_             Mongo's ID tag of the duplicate document that you just added
    '''
    with get_mongo_collection(collection_tag=collection_tag) as collection:
        docs = list(collection.find({}))
        starting_count = len(docs)

        doc_duplicate = random.choice(docs)
        del doc_duplicate['_id']

        insertion_result = collection.insert_one(doc_duplicate)
        id_ = insertion_result.inserted_id
    return starting_count, id_
