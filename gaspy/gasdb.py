''' These tools form gaspy's API to its databases '''

import warnings
import copy
from collections import defaultdict
import numpy as np
import tqdm
from pymongo import MongoClient
from pymongo.collection import Collection
from . import defaults, utils


def get_mongo_collection(collection_tag):
    '''
    Get a mongo collection, but with `__enter__` and `__exit__` methods
    that will allow you to establish and close connections with `with` statements.

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
    mongo_info = utils.read_rc('mongo_info')[collection_tag]
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


# An extendeded version of the pymongo.collection.Collection class
# that can be open and closed via a `with` statement
class ConnectableCollection(Collection):
    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, exception_traceback):   # noqa: E301
        self.database.client.close()


def get_adsorption_docs(adsorbates=None, extra_fingerprints=None, filters=None):
    '''
    A wrapper for `collection.aggregate` that is tailored specifically for the
    collection that's tagged `adsorption`.

    Args:
        adsorbates          [optional] A list of the adsorbates that you need to
                            be present in each document's corresponding atomic
                            structure. Note that if you pass a list with two adsorbates,
                            then you will only get matches for structures with *both*
                            of those adsorbates; you will *not* get structures
                            with only one of the adsorbates. If you pass nothing, then we
                            get all documents regardless of adsorbates.
        extra_fingerprints  A dictionary with key/value pairings that correspond
                            to a new fingerprint you want to fetch and its location in
                            the Mongo docs, respectively. Refer to
                            `gaspy.defaults.adsorption_fingerprints` for examples.
        filters             A dictionary whose keys are the locations of elements
                            in the Mongo collection and whose values are Mongo
                            matching commands. For examples, look up Mongo `match`
                            commands. If this argument is `None`, then it will
                            fetch the default filters from
                            `gaspy.defaults.adsorption_filters`. If you want to modify
                            them, we suggest simply fetching that object, modifying it,
                            and then passing it here.
    Returns:
        docs    A list of dictionaries whose key/value pairings are the
                ones given by `gaspy.defaults.adsorption_fingerprints`
                and who meet the filtering criteria of
                `gaspy.defaults.adsorption_filters`
    '''
    # Establish the information that'll be contained in the documents we'll be getting.
    # Also add anything the user asked for.
    fingerprints = defaults.adsorption_fingerprints(adsorbates)
    if extra_fingerprints:
        for key, value in extra_fingerprints.items():
            fingerprints[key] = value
    group = {'$group': {'_id': fingerprints}}

    # Set the filtering criteria of the documents we'll be getting
    if not filters:
        filters = defaults.adsorption_filters(adsorbates)
    if adsorbates:
        filters['processed_data.calculation_info.adsorbate_names'] = adsorbates
    match = {'$match': filters}

    # Get the documents and clean them up
    pipeline = [match, group]
    with get_mongo_collection(collection_tag='adsorption') as collection:
        print('Now pulling adsorption documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm.tqdm(cursor)]
    docs = _clean_up_aggregated_docs(docs, expected_keys=fingerprints.keys())

    return docs


def _clean_up_aggregated_docs(docs, expected_keys):
    '''
    This function takes a list of dictionaries and returns a new instance
    of the list without dictionaries that have missing keys or `None` as values.
    It assumes that dictionaries are flat, thus the `aggregated` part in the name.

    Arg:
        docs            A list of mongo documents, AKA a list of dicts, AKA a list of JSONs
        expected_keys   The dict keys that that you expect to be in every document.
                        If a document doesn't have the right keys or has `None` for one of them,
                        then it is deleted.
    Returns:
        clean_docs  A subset of the `docs` argument with
    '''
    # Get rid of one of the useless JSON branch-points that `aggregate` forces on us
    docs = [doc['_id'] for doc in docs]

    cleaned_docs = []
    for doc in docs:
        is_clean = True

        # Clean up documents that don't have the right keys
        if doc.keys() != expected_keys:
            is_clean = False
        # Clean up documents that have `None` or '' as values
        for key, value in doc.items():
            if (value is None) or (value is ''):
                is_clean = False
                break
            # Clean up documents that have no second-shell atoms
            if key == 'neighborcoord':
                for neighborcoord in value:  # neighborcoord looks like ['Cu:Cu-Cu-Cu-Cu', 'Cu:Cu-Cu-Cu-Cu']
                    neighbor, coord = neighborcoord.split(':')
                    if not coord:
                        is_clean = False
                        break

        if is_clean:
            cleaned_docs.append(doc)

    # Warn the user if we did not actually get any documents out the end.
    if not cleaned_docs:
        warnings.warn('We did not find any matching documents', RuntimeWarning)

    return cleaned_docs


def get_catalog_docs():
    '''
    A wrapper for `collection.aggregate` that is tailored specifically for the
    collection that's tagged `relaxed_bulk_catalog`.

    Returns:
        docs    A list of dictionaries whose key/value pairings are the
                ones given by `gaspy.defaults.catalog_fingerprints`
    '''
    # Establish the information that'll be contained in the documents we'll be getting
    fingerprints = defaults.catalog_fingerprints()
    group = {'$group': {'_id': fingerprints}}

    # Get the documents and clean them up
    pipeline = [group]
    with get_mongo_collection(collection_tag='relaxed_bulk_catalog_readonly') as collection:
        print('Now pulling catalog documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm.tqdm(cursor)]
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=fingerprints.keys())

    return cleaned_docs


def get_surface_docs(extra_fingerprints=None, filters=None):
    '''
    A wrapper for `collection.aggregate` that is tailored specifically for the
    collection that's tagged `surface_energy`.

    Args:
        extra_fingerprints  A dictionary with key/value pairings that correspond
                            to a new fingerprint you want to fetch and its location in
                            the Mongo docs, respectively. Refer to
                            `gaspy.defaults.surface_fingerprints` for examples.
        filters             A dictionary whose keys are the locations of elements
                            in the Mongo collection and whose values are Mongo
                            matching commands. For examples, look up Mongo `match`
                            commands. If this argument is `None`, then it will
                            fetch the default filters from
                            `gaspy.defaults.surface_filters`. If you want to modify
                            them, we suggest simply fetching that object, modifying it,
                            and then passing it here.
    Returns:
        docs    A list of dictionaries whose key/value pairings are the
                ones given by `gaspy.defaults.adsorption_fingerprints`
                and who meet the filtering criteria of
                `gaspy.defaults.adsorption_filters`
    '''
    # Establish the information that'll be contained in the documents we'll be getting
    # Also add anything the user asked for.
    fingerprints = defaults.surface_fingerprints()
    if extra_fingerprints:
        for key, value in extra_fingerprints.items():
            fingerprints[key] = value
    group = {'$group': {'_id': fingerprints}}

    # Set the filtering criteria of the documents we'll be getting
    if not filters:
        filters = defaults.surface_filters()
    match = {'$match': filters}

    # Get the documents and clean them up
    pipeline = [match, group]
    with get_mongo_collection(collection_tag='surface_energy') as collection:
        print('Now pulling surface documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm.tqdm(cursor)]
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=fingerprints.keys())

    return cleaned_docs


def get_unsimulated_catalog_docs(adsorbates):
    '''
    Gets the same documents from `get_catalog_docs`, but then filters out
    all items that also show up in `get_adsorption_docs`, i.e., gets the
    catalog items that have not yet been simulated using our default
    settings.

    Args:
        adsorbates  Every site in the catalog can be simulated with sets of
                    different adsorbates. Here, you specify a sequence of
                    strings indicating which set of adsorbates you are
                    checking for simulation. For example:  using
                    ['OH', 'H'] will look for simulations where OH and H
                    are co-adsorbed. It will *not* look for simulations
                    with either OH or H.
    Returns:
        docs    A list of dictionaries for various fingerprints.
    '''
    docs_simulated = _get_attempted_adsorption_docs(adsorbates=adsorbates)
    docs_catalog = get_catalog_docs()

    # Identify unsimulated documents by comparing hashes
    # of catalog docs vs. simulated adsorption docs
    hashes_simulated = _hash_docs(docs_simulated, ignore_keys=['mongo_id',
                                                               'formula',
                                                               'energy',
                                                               'adsorbates',
                                                               'adslab_calculation_date'])
    hashes_catalog = _hash_docs(docs_catalog, ignore_keys=['mongo_id',
                                                           'formula',
                                                           'adsorption_site',
                                                           'predictions'])
    hashes_unsimulated = set(hashes_catalog) - set(hashes_simulated)

    docs = []
    for doc, hash_ in zip(docs_catalog, hashes_catalog):
        if hash_ in hashes_unsimulated:
            docs.append(doc)
    return docs


def _get_attempted_adsorption_docs(adsorbates=None):
    '''
    A wrapper for `collection.aggregate` that is tailored specifically for the
    collection that's tagged `adsorption`. This differs from `get_adsorption_docs`
    in two ways:  1) it does not filter out "bad adsorptions" and 2) it takes
    fingerprints based on initial configurations, not final, post-relaxation
    cofigurations.

    Args:
        adsorbates      [optional] A list of the adsorbates that you need to
                        be present in each document's corresponding atomic
                        structure. Note that if you pass a list with two adsorbates,
                        then you will only get matches for structures with *both*
                        of those adsorbates; you will *not* get structures
                        with only one of the adsorbates. If you pass nothing, then we
                        get all documents regardless of adsorbates.
    Returns:
        docs    A list of dictionaries whose key/value pairings are the
                ones given by `gaspy.defaults.adsorption_fingerprints`
                and who meet the filtering criteria of
                `gaspy.defaults.adsorption_filters`
    '''
    # Establish the information that'll be contained in the documents we'll be getting
    fingerprints = defaults.adsorption_fingerprints()
    fingerprints['coordination'] = '$processed_data.fp_init.coordination'
    fingerprints['neighborcoord'] = '$processed_data.fp_init.neighborcoord'
    fingerprints['nextnearestcoordination'] = '$processed_data.fp_init.nextnearestcoordination'
    group = {'$group': {'_id': fingerprints}}

    # Get only the documents that have the specified adsorbates
    filters = {}
    if adsorbates:
        filters['processed_data.calculation_info.adsorbate_names'] = adsorbates
    match = {'$match': filters}

    # Get the documents and clean them up
    pipeline = [match, group]
    with get_mongo_collection(collection_tag='adsorption') as collection:
        print('Now pulling adsorption documents for sites we have attempted...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm.tqdm(cursor)]
    docs = _clean_up_aggregated_docs(docs, expected_keys=fingerprints.keys())

    return docs


def _hash_docs(docs, ignore_keys=None, _return_hashes=True):
    '''
    This function helps convert the important characteristics of our systems into hashes
    so that we may sort through them more quickly. This is useful when trying to
    quickly compare entries in two databases.

    Arg:
        docs            Sequence of single-layered dictionary/json/Mongo
                        document/whatever you call it.
        ignore_keys     A list of strings indicating the keys that you want to ignore
                        when hashing each document.
        _return_hashes  *For unit testing only!* If `False`, returns the pre-hash objects
    Returns:
        hashes     A set containing the hashes of the each doc in `docs`.
    '''
    # Python doesn't do well with mutable default arguments
    if not ignore_keys:
        ignore_keys = []

    # Hash with a progress bar
    hashes = [_hash_doc(doc=doc, ignore_keys=ignore_keys, _return_hash=_return_hashes)
              for doc in docs]
    return hashes


def _hash_doc(doc, ignore_keys=None, _return_hash=True):
    '''
    Hash a single Mongo document (AKA dictionary). This function currently assumes
    that the document is flat---i.e., not nested.

    Args:
        doc             A single-layered dictionary/json/Mongo document/whatever you call it.
        ignore_keys     A sequence of strings indicating the keys that you want to ignore
                        when hashing the document.
        _return_hashe   *For unit testing only!* If `False`, returns the pre-hash object
    Returns:
        hash_   A hashed version of the document
    '''
    # Python doesn't do well with mutable default arguments
    if not ignore_keys:
        ignore_keys = []
    else:
        ignore_keys = copy.deepcopy(ignore_keys)

    # Add the mongo ID to the list of ignored keys, because that'll always yield a different hash.
    ignore_keys.append('mongo_id')

    # `system` will be one long string of the fingerprints.
    # After we populate it with non-ignored key/value pairs, we'll hash it and return it.
    system = ''
    for key in sorted(doc.keys()):
        if key not in set(ignore_keys):
            value = doc[key]
            # Clean up the values so they hash consistently
            if isinstance(value, float):
                value = round(value, 2)
            system += str(key + '=' + str(value) + '; ')

    if _return_hash:
        hash_ = hash(system)
        return hash_
    # For unit testing, because hashes change between instances of Python
    else:
        return system


def get_low_coverage_docs_by_surface(adsorbates, model_tag='model0'):
    '''
    Each surface has many possible adsorption sites. The site with the most
    negative adsorption energy (i.e., the strongest-binding site) will tend to
    be the dominating site at low adsorbate coverages. This function will find
    and return the low-coverage binding site for each surface. The adsorption
    energies used to find these sites are taken from DFT calculations whenever
    possible; when not possible, the energies are taken from model predictions.

    If a document came from the adsorption collection, then it will inherit
    the structure it got from `get_adsorption_docs`. If a document came from
    the catalog collection, then it will inherit the structure it got from
    `get_catalog_docs`. For ease-of-use, we also copied the predicted energies
    within each catalog document into the 'energy' key of the document (just
    like the adsorption documents have) so that all documents have energies
    in one consistent location.

    Args:
        adsorbates  A list of strings that represent the adsorbates you want
                    to get the low-coverage sites for. Example: ['CO'] or ['CO', 'C']
        model_tag   A string indicating which model you want to
                    use when using non-DFT, predicted energies. Check out
                    the `predictions.adsorption_energy` key in the catalog
                    documents for valid inputs. Note that these keys
                    are created by the `GASpy_regressions` submodule.
    Returns:
        low_coverage_docs_by_surface    A dictionary whose keys are 4-tuples
                                        of the MPID, miller index, shift,
                                        and top/bottom of a surface and whose
                                        values are the aggregated documents we
                                        get from either `get_adsorption_docs`
                                        or `get_catalog_docs`
    '''
    # Get the documents, and then copy the latest predicted energies into the appropriate key
    adsorption_docs = get_adsorption_docs(adsorbates)
    catalog_docs = get_unsimulated_catalog_docs(adsorbates)
    for doc in catalog_docs:
        energy_data_by_date = doc['predictions']['adsorption_energy'][adsorbates[0]][model_tag]
        dates = energy_data_by_date.keys()
        energy_data = energy_data_by_date.values()
        dates_and_energy_data = list(zip(dates, energy_data))
        latest_energy_data = sorted(dates_and_energy_data)[-1][1]
        doc['energy'] = latest_energy_data['energy']

    # Fetch each document and organize/bucket them by surface
    docs_by_surface = defaultdict(list)
    for doc in adsorption_docs + catalog_docs:
        surface = (doc['mpid'], str(doc['miller']), doc['shift'], doc['top'])
        docs_by_surface[surface].append(doc)

    # Initialize output
    surfaces = docs_by_surface.keys()
    low_coverage_docs_by_surface = dict.fromkeys(surfaces)

    # For each surface, find the document that corresponds to the low-coverage site
    for surface, docs in docs_by_surface.items():
        energies = [doc['energy'] for doc in docs]
        low_coverage_doc = docs[np.argmin(np.array(energies))]
        # Tag each document explicitly so that users will know which is which
        if 'predictions' not in low_coverage_doc:
            low_coverage_doc['DFT_calculated'] = True
        else:
            low_coverage_doc['DFT_calculated'] = False

        low_coverage_docs_by_surface[surface] = low_coverage_doc
    return low_coverage_docs_by_surface


def remove_duplicates_in_adsorption_collection():
    ''' Things that share FireWorks IDs for slab+adsorbate structures are duplicates '''
    identifying_query = {'adslab_fwid': '$processed_data.FW_info.slab+adsorbate'}
    _remove_duplicates_in_a_collection(collection_tag='adsorption',
                                       identifying_query=identifying_query)


def remove_duplicates_in_atoms_collection():
    ''' Things that share FireWorks IDs are duplicates '''
    identifying_query = {'fwid': '$fwid'}
    _remove_duplicates_in_a_collection(collection_tag='atoms',
                                       identifying_query=identifying_query)


def _remove_duplicates_in_a_collection(collection_tag, identifying_query):
    '''
    This function removes duplicate entries in a collection. "What constitutes
    a 'duplicate'", you ask? You do, of course, via the `identifying_query`
    argument!

    Args:
        collection_tag      A string indicating which collection you want to parse
        identifying_query   A mongo-style query (i.e., a dictionary) with a
                            virtually arbitrary key and a value corresponding
                            to the location of the Mongo document information
                            that should be unique among all documents.
    '''
    group = {'_id': identifying_query}

    # Reference <https://www.compose.com/articles/finding-duplicate-documents-in-mongodb/>
    # for details on how this works
    group['mongo_ids'] = {'$addToSet': '$_id'}
    group['count'] = {'$sum': 1}
    match = {'count': {'$gt': 1}}

    # `docs` will have one entry per FWID that happens to have multiple documents
    pipeline = [{'$group': group}, {'$match': match}]
    with get_mongo_collection(collection_tag=collection_tag) as collection:
        docs = list(collection.aggregate(pipeline=pipeline, allowDiskUse=True))

        # For each FWID that has duplicates, keep only the last document.
        # Delete the rest.
        for doc in docs:
            extra_mongo_ids = doc['mongo_ids'][:-1]
            for id_ in extra_mongo_ids:
                collection.delete_one({'_id': id_})
