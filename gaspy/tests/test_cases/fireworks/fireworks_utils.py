'''
This submodule contains various helper functions that you may or may not
find useful for managing the unit testing collection for fireworks.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import warnings
import pickle
from bson.objectid import ObjectId
from pymongo import MongoClient
from ..utils import read_testing_rc
from ....gasdb import ConnectableCollection

LOCATION_OF_DOCS = {'unit_testing_fireworks': ('/home/GASpy/gaspy/tests'
                                               '/test_cases/fireworks'
                                               '/unit_testing_fireworks.pkl'),
                    'unit_testing_launches': ('/home/GASpy/gaspy/tests'
                                              '/test_cases/fireworks'
                                              '/unit_testing_launches.pkl')}


def get_testing_collection(collection_name):
    '''
    Gets you an Pymongo collection instance for our unit testing fireworks
    collection.

    Returns:
        collection  A mongo collection object corresponding to the collection
                    tag you specified, but with `__enter__` and `__exit__` methods.
    '''
    # Login info
    fireworks_info = read_testing_rc('fireworks_info.lpad')
    host = fireworks_info['host']
    port = int(fireworks_info['port'])
    database_name = fireworks_info['name']
    user = fireworks_info['username']
    password = fireworks_info['password']

    # Connect to the database/collection
    client = MongoClient(host=host, port=port)
    database = getattr(client, database_name)
    database.authenticate(user, password)
    collection = ConnectableCollection(database=database, name=collection_name)

    return collection


def create_and_populate_unit_testing_collection(collection_name):
    '''
    This function will create and populate a mock collection for unit testing.
    '''
    create_unit_testing_collection(collection_name)
    populate_unit_testing_collection(collection_name)


def create_unit_testing_collection(collection_name):
    '''
    This function will create a mock collection for unit testing.

    Arg:
        collection_name The name of the unit testing collection you want to use
    '''
    # Get the information needed to [re]create the collection
    fireworks_info = read_testing_rc('fireworks_info.lpad')
    host = fireworks_info['host']
    port = int(fireworks_info['port'])
    database_name = fireworks_info['name']
    user = fireworks_info['username']
    password = fireworks_info['password']

    # Create the collection
    with MongoClient(host=host, port=port) as client:
        database = getattr(client, database_name)
        database.authenticate(user, password)
        database.create_collection(collection_name)


def populate_unit_testing_collection(collection_name):
    '''
    This function will populate a unit testing collection with the contents
    that it's "supposed" to have.

    Arg:
        collection_name The name of the unit testing collection you want to use
    '''
    # Get the documents that are supposed to be in the collection
    with open(LOCATION_OF_DOCS[collection_name], 'rb') as file_handle:
        docs = pickle.load(file_handle)

    # Put the documents in
    with get_testing_collection(collection_name) as collection:
        collection.insert_many(docs)


def get_and_push_doc(mongo_id, collection_name):
    '''
    This function will get one document from a collection for you and then
    push/write it to its corresponding unit testing collection.

    Args:
        mongo_id        Mongo _id for the document you want to get
        collection_name The name of the unit testing collection you want to use
    '''
    doc = get_doc(mongo_id, collection_name)
    push_doc(doc, collection_name)


def get_doc(mongo_id, collection_name):
    '''
    This function will get one full document from the "live" Mongo database.

    Args:
        mongo_id        Mongo _id for the document you want to get. Can
                        be a string or a bjson.objectid.ObjectId object.
        collection_name The name of the unit testing collection you want to use
    Returns:
        doc     A dictionary of the document with the corresponding mongo ID
    '''
    # Convert the `mongo_id` into an ObjectId, if needed
    if isinstance(mongo_id, str):
        mongo_id = ObjectId(mongo_id)

    with get_testing_collection(collection_name) as collection:
        live_collection = collection.database.fireworks
        docs = list(live_collection.find({'_id': mongo_id}))

    if not docs:
        warnings.warn('Did not find any document with the Mongo id %s'
                      % mongo_id, RuntimeWarning)
        doc = [{}]
    else:
        doc = docs[0]

    return doc


def push_doc(doc, collection_name):
    '''
    This function will push/write one document to the testing database.

    Args:
        doc     Dictionary/document that you want to write
        collection_name The name of the unit testing collection you want to use
    '''
    with get_testing_collection(collection_name) as collection:
        collection.insert_one(doc)


def dump_collection_to_pickle(collection_name):
    '''
    If you've updated a unit testing collection in Mongo but have not yet
    updated the corresponding pickle cache, then you can use this function
    to do so.

    Arg:
        collection_name     The name of the unit testing collection you want to use
    '''
    with get_testing_collection(collection_name) as collection:
        docs = list(collection.find())

    with open(LOCATION_OF_DOCS[collection_name], 'wb') as file_handle:
        pickle.dump(docs, file_handle)
