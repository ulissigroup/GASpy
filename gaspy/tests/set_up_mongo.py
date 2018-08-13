'''
This module contains certain scripts that were constructed to set up your
Mongo testing environment for proper testing, because many of our unit
tests expect your Mongo server to have certain contents.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

import pickle
from pymongo import MongoClient
from ..utils import read_rc
from ..gasdb import get_mongo_collection

LOCATION_OF_DOCS = '/home/GASpy/gaspy/tests/mongo_test_collections/'


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
    create_unit_testing_adsorption_collection(collection_tag)
    populate_unit_testing_adsorption_collection(collection_tag)


def create_unit_testing_adsorption_collection(collection_tag):
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
    mongo_info = read_rc()['mongo_info'][collection_tag]
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


def populate_unit_testing_adsorption_collection(collection_tag):
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
    with get_mongo_collection(collection_tag) as collection:
        collection.insert_many(docs)
