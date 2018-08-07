'''
Tests for the `gasdb` submodule. Since this submodule deals with Mongo
databases so much, we created and work with various collections that are tagged
with `unit_testing_adsorption`. We assume that these collections have very
specific contents. We also assume that documents within each collection should
have the same exact JSON architecture.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we're testing
from ..gasdb import get_mongo_collection, \
    ConnectableCollection, \
    get_adsorption_docs, \
    _clean_up_aggregated_docs, \
    get_catalog_docs, \
    get_surface_docs, \
    get_unsimulated_catalog_docs, \
    _get_attempted_adsorption_docs, \
    _hash_docs, \
    _hash_doc, \
    remove_duplicates_in_adsorption_collection

# Things we need to do the tests
import pytest
import copy
import pickle
import random
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from ..utils import read_rc

REGRESSION_BASELINES_LOCATION = '/home/GASpy/gaspy/tests/regression_baselines/gasdb/'


def test_get_mongo_collection():
    collection = get_mongo_collection(collection_tag='unit_testing_adsorption')

    # Make sure it's the correct class type
    assert isinstance(collection, ConnectableCollection)

    # Make sure it's connected and authenticated by counting the documents
    try:
        _ = collection.count()  # noqa: F841
    except OperationFailure:
        assert False


def test_ConnectableCollection():
    '''
    Verify that the extended `ConnectableCollection` class
    is still a Collection class and has the appropriate methods
    '''
    # Login info
    mongo_info = read_rc()['mongo_info']['unit_testing_adsorption']
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
def test_to_create_expected_aggregated_adsorption_documents():
    docs = get_adsorption_docs(_collection_tag='unit_testing_adsorption')
    with open(REGRESSION_BASELINES_LOCATION + 'aggregated_adsorption_documents' + '.pkl', 'wb') as file_handle:
        pickle.dump(docs, file_handle)
    assert True


def get_expected_aggregated_adsorption_documents():
    with open(REGRESSION_BASELINES_LOCATION + 'aggregated_adsorption_documents' + '.pkl', 'rb') as file_handle:
        docs = pickle.load(file_handle)
    return docs


def test_get_adsorption_docs():
    expected_docs = get_expected_aggregated_adsorption_documents()
    docs = get_adsorption_docs(_collection_tag='unit_testing_adsorption')
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
    doc_empty = copy.deepcopy(random.choice(docs))
    key_to_modify = random.choice(list(doc_empty.keys()))
    doc_empty[key_to_modify] = None

    # Make the dirty documents
    dirty_docs = copy.deepcopy(docs) + [doc_partial] + [doc_empty]
    dirty_docs = [{'_id': doc} for doc in dirty_docs]   # because mongo is stupid
    return dirty_docs


@pytest.mark.baseline
def test_to_create_aggregated_catalog_documents():
    docs = get_catalog_docs(_collection_tag='unit_testing_catalog')
    with open(REGRESSION_BASELINES_LOCATION + 'aggregated_catalog_documents' + '.pkl', 'wb') as file_handle:
        pickle.dump(docs, file_handle)
    assert True


def get_expected_aggregated_catalog_documents():
    with open(REGRESSION_BASELINES_LOCATION + 'aggregated_catalog_documents' + '.pkl', 'rb') as file_handle:
        docs = pickle.load(file_handle)
    return docs


def test_get_catalog_docs():
    expected_docs = get_expected_aggregated_catalog_documents()
    docs = get_catalog_docs(_collection_tag='unit_testing_catalog')
    assert docs == expected_docs


@pytest.mark.baseline
def test_to_create_aggregated_surface_documents():
    docs = get_surface_docs(_collection_tag='unit_testing_surface_energy')
    with open(REGRESSION_BASELINES_LOCATION + 'aggregated_surface_documents' + '.pkl', 'wb') as file_handle:
        pickle.dump(docs, file_handle)
    assert True


def test_get_surface_docs():
    with open(REGRESSION_BASELINES_LOCATION + 'aggregated_surface_documents' + '.pkl', 'rb') as file_handle:
        expected_docs = pickle.load(file_handle)
    docs = get_surface_docs(_collection_tag='unit_testing_surface_energy')
    assert docs == expected_docs


@pytest.mark.baseline
@pytest.mark.parametrize('adsorbates', [['H'], ['CO']])
def test_to_create_unsimulated_catalog_docs(adsorbates):
    docs = get_unsimulated_catalog_docs(adsorbates=adsorbates,
                                        _catalog_collection_tag='unit_testing_catalog',
                                        _adsorption_collection_tag='unit_testing_adsorption')

    file_name = REGRESSION_BASELINES_LOCATION + 'unsimulated_' + '_'.join(adsorbates) + '_catalog_docs' + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(docs, file_handle)
    assert True


@pytest.mark.parametrize('adsorbates', [['H'], ['CO']])
def test_get_unsimulated_catalog_docs(adsorbates):
    file_name = REGRESSION_BASELINES_LOCATION + 'unsimulated_' + '_'.join(adsorbates) + '_catalog_docs' + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_docs = pickle.load(file_handle)

    docs = get_unsimulated_catalog_docs(adsorbates=adsorbates,
                                        _catalog_collection_tag='unit_testing_catalog',
                                        _adsorption_collection_tag='unit_testing_adsorption')
    assert docs == expected_docs


@pytest.mark.baseline
@pytest.mark.parametrize('adsorbates', [None, ['H'], ['CO']])
def test_to_create_attempted_adsorption_docs(adsorbates):
    docs = _get_attempted_adsorption_docs(adsorbates=adsorbates,
                                          _collection_tag='unit_testing_adsorption')

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

    docs = _get_attempted_adsorption_docs(adsorbates=adsorbates,
                                          _collection_tag='unit_testing_adsorption')
    #assert docs == expected_docs
    assert _compare_unordered_sequences(docs, expected_docs)


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
    ond the pre-hashed strings, not the hash itself.
    '''
    # EAFP to find the pickle name of the cache, since it'll be odd if adsorbates == None
    try:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_docs_ignoring_' + '_'.join(ignore_keys) + '.pkl'
    except TypeError:
        file_name = REGRESSION_BASELINES_LOCATION + 'hashed_docs_ignoring_nothing.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_strings = pickle.load(file_handle)

    docs = get_expected_aggregated_adsorption_documents()
    strings = _hash_docs(docs=docs, ignore_keys=ignore_keys, _return_hashes=False)
    assert strings == expected_strings


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


def test_remove_duplicates_in_adsorption_collection():
    collection_tag = 'unit_testing_adsorption'
    starting_doc_count, id_of_duplicate_doc = add_duplicate_document_to_collection(collection_tag)

    try:
        remove_duplicates_in_adsorption_collection(_collection_tag=collection_tag)

        # Verify that removal worked
        with get_mongo_collection(collection_tag=collection_tag) as collection:
            current_doc_count = collection.count()
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
