'''
This submodule contains various helper functions that you may or may not
find useful for managing the unit testing collections.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import warnings
import pickle
import json
from bson.objectid import ObjectId
from pymongo import MongoClient
from ....gasdb import get_mongo_collection, ConnectableCollection

LOCATION_OF_DOCS = '/home/GASpy/gaspy/tests/test_cases/mongo_test_collections/'
ALL_COLLECTION_TAGS = ['atoms', 'catalog', 'adsorption', 'surface_energy']


def read_testing_rc(query=None):
    '''
    This function is nearly identical to the one in `gaspy.utils.read_rc`,
    except this one specifically reads from the .gaspyrc.json file
    within the testing folder, which should contain Mongo collection
    information about your testing database.

    Input:
        query   [Optional] The string indicating the configuration you want.
                If you're looking for nested information, use the syntax
                "foo.bar.key"
    Output:
        rc_contents  A dictionary whose keys are the input keys and whose values
                     are the values that we found in the .gaspyrc file
    '''
    try:
        with open('/home/GASpy/gaspy/tests/.gaspyrc.json', 'r') as file_handle:
            rc_contents = json.load(file_handle)
    except FileNotFoundError as error:
        raise FileNotFoundError('You have not yet created your .gaspyrc.json file in the gaspy/tests/ folder').with_traceback(error.__traceback__)

    # Return out the keys you asked for. If the user did not specify the key, then return it all
    if query:
        keys = query.split('.')
        for key in keys:
            try:
                rc_contents = rc_contents[key]
            except KeyError as error:
                raise KeyError('Check the spelling/capitalization of the key/values you are looking for').with_traceback(error.__traceback__)

    return rc_contents


def get_testing_mongo_collection(collection_tag):
    '''
    Virtually identical to `gaspy.gasdb.get_mongo_collection`, except
    it pulls the login information from .gaspyrc.json in the testing
    folder, instead.

    Args:
        collection_tag  All of the information needed to access a specific
                        Mongo collection is stored in the .gaspyrc.json file.
                        This argument specifices which branch within that json
                        to parse for the Mongo information needed to access
                        the data. Examples may include (but may not be limited to):
                            'catalog'
                            'atoms'
                            'adsorption'
                            'surface_energy'
    Returns:
        collection  A mongo collection object corresponding to the collection
                    tag you specified, but with `__enter__` and `__exit__` methods.
    '''
    # Login info
    mongo_info = read_testing_rc('mongo_info.%s' % collection_tag)
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

    return collection


def create_and_populate_all_unit_testing_collections():
    '''
    This function will create and populate all unit testing collections.
    '''
    for collection_tag in ALL_COLLECTION_TAGS:
        create_and_populate_unit_testing_collection(collection_tag)


def create_and_populate_unit_testing_collection(collection_tag):
    '''
    This function will create and populate a mock collection for unit testing.

    Arg:
        collection_tag  String indicating which collection you want
                        to create. Should probably be either:
                            'adsorption'
                            'catalog'
                            'surface_energy'
                            'atoms'
                        Valid tags can be seen in the
                        `.gaspyrc.json.template` file under the
                        `mongo_info` branch and should have a
                        'unit_testing_*' prefix.
    '''
    create_unit_testing_collection(collection_tag)
    populate_unit_testing_collection(collection_tag)


def create_unit_testing_collection(collection_tag):
    '''
    This function will create a mock collection for unit testing.

    Arg:
        collection_tag  String indicating which collection you want
                        to create. Should probably be either:
                            'adsorption'
                            'catalog'
                            'surface_energy'
                            'atoms'
                        Valid tags can be seen in the
                        `.gaspyrc.json.template` file under the
                        `mongo_info` branch and should have a
                        'unit_testing_*' prefix.
    '''
    # Get the information needed to [re]create the collection
    mongo_info = read_testing_rc('mongo_info.%s' % collection_tag)
    host = mongo_info['host']
    port = int(mongo_info['port'])
    database_name = mongo_info['database']
    user = mongo_info['user']
    password = mongo_info['password']
    collection_name = mongo_info['collection_name']

    # Create the collection
    with MongoClient(host=host, port=port) as client:
        database = getattr(client, database_name)
        database.authenticate(user, password)
        database.create_collection(collection_name)


def populate_unit_testing_collection(collection_tag):
    '''
    This function will populate a unit testing collection with
    the contents that it's "supposed" to have.

    Arg:
        collection_tag  String indicating which collection you want
                        to create. Should probably be either:
                            'adsorption'
                            'catalog'
                            'surface_energy'
                            'atoms'
                        Valid tags can be seen in the
                        `.gaspyrc.json.template` file under the
                        `mongo_info` branch and should have a
                        'unit_testing_*' prefix.
    '''
    # Get the documents that are supposed to be in the collection
    with open(LOCATION_OF_DOCS + 'unit_testing_' + collection_tag + '_docs.pkl', 'rb') as file_handle:
        docs = pickle.load(file_handle)

    # Put the documents in
    with get_testing_mongo_collection(collection_tag) as collection:
        collection.insert_many(docs)


def get_and_push_doc(mongo_id, collection_tag):
    '''
    This function will get one document from a collection for you and then
    push/write it to its corresponding unit testing collection.

    Args:
        mongo_id        Mongo _id for the document you want to get
        collection_tag  A string indicating the collection you want to pull from
    '''
    doc = get_doc(mongo_id, collection_tag)
    push_doc(doc, collection_tag)


def get_doc(mongo_id, collection_tag):
    '''
    This function will get one full document from the "live" Mongo database.

    Args:
        mongo_id        Mongo _id for the document you want to get. Can
                        be a string or a bjson.objectid.ObjectId object.
        collection_tag  A string indicating the collection you want to pull from
    Returns:
        doc     A dictionary of the document with the corresponding mongo ID
    '''
    # Convert the `mongo_id` into an ObjectId, if needed
    if isinstance(mongo_id, str):
        mongo_id = ObjectId(mongo_id)

    with get_mongo_collection(collection_tag) as collection:
        docs = list(collection.find({'_id': mongo_id}))

    if not docs:
        warnings.warn('Did not find any document with the Mongo id %s' % mongo_id, RuntimeWarning)
        doc = [{}]
    else:
        doc = docs[0]

    return doc


def push_doc(doc, collection_tag):
    '''
    This function will push/write one document to the testing database.

    Args:
        doc             Dictionary/document that you want to write
        collection_tag  A string indicating the collection you want to pull from
    '''
    with get_testing_mongo_collection(collection_tag) as collection:
        collection.insert_one(doc)


def populate_unit_testing_atoms_collection():
    '''
    If you're crafting your ouwn unit testing collections,
    it'd makes sense if your unit testing atoms collection contains
    all of the atoms objects that were prerequisite to the documents
    you have in the `adsorption` collections.

    This function will read your unit testing `adsorption` collection
    and populate your unit testing `atoms` collection accordingly.
    '''
    # Get the docs of all the atoms in the unit testing 'adsorption' collection
    with get_testing_mongo_collection('adsorption') as collection:
        adsorption_docs = list(collection.find({}, {'processed_data.FW_info': 1}))

    # Find the FWIDs of all of the prerequisite calculations
    fwids = []
    for doc in adsorption_docs:
        fw_info = doc['processed_data']['FW_info']
        for calc_type in ['bulk', 'slab', 'slab+adsorbate']:
            fwid = fw_info[calc_type]
            fwids.append(fwid)

    # Get and save all of the atoms documents needed to make the adsorption documents
    atoms_docs = []
    with get_mongo_collection('atoms') as collection:
        for fwid in set(fwids):
            doc = list(collection.find({'fwid': fwid}))[0]
            atoms_docs.append(doc)
    with get_testing_mongo_collection('atoms') as collection:
        collection.insert_many(atoms_docs)

    # Let's just throw in all of the adsorbate calculations for good measure
    with get_mongo_collection('atoms') as collection:
        gas_docs = list(collection.find({'type': 'gas'}))
    with get_testing_mongo_collection('atoms') as collection:
        collection.insert_many(gas_docs)


def dump_all_unit_testing_collections_to_pickles():
    '''
    This function will create and populate all unit testing collections.
    '''
    for collection_tag in ALL_COLLECTION_TAGS:
        dump_collection_to_pickle(collection_tag)


def dump_collection_to_pickle(collection_tag):
    '''
    If you've updated a unit testing collection in Mongo but have not yet
    updated the corresponding pickle cache, then you can use this function
    to do so.

    Args:
        collection_tag  String indicating which collection you want to pickle.
                        Note that you can write both normal and unit testing
                        collections with this function.
    '''
    with get_testing_mongo_collection(collection_tag) as collection:
        docs = list(collection.find())

    with open(LOCATION_OF_DOCS + 'unit_testing_' + collection_tag + '_docs.pkl', 'wb') as file_handle:
        pickle.dump(docs, file_handle)
