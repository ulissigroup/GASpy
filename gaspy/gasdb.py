''' These tools form gaspy's API to its databases '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import warnings
import math
from copy import deepcopy
import json
from tqdm import tqdm
from pymongo import MongoClient
from pymongo.collection import Collection
from . import defaults
from .utils import read_rc
from .fireworks_helper_scripts import get_launchpad


def get_mongo_collection(collection_tag):
    '''
    Get a mongo collection, but with `__enter__` and `__exit__` methods that
    will allow you to establish and close connections with `with` statements.

    Args:
        collection_tag  All of the information needed to access a specific
                        Mongo collection is stored in the .gaspyrc.json file.
                        This argument specifices which branch within that json
                        to parse for the Mongo information needed to access the
                        data. Examples may include (but may not be limited to):
                            'catalog'
                            'atoms'
                            'adsorption'
                            'surface_energy'
    Returns:
        collection  A mongo collection object corresponding to the collection
                    tag you specified, but with `__enter__` and `__exit__`
                    methods.
    '''
    # Login info
    mongo_info = read_rc('mongo_info')[collection_tag]
    host = mongo_info['host']
    port = int(mongo_info['port'])
    database_name = mongo_info['database']
    user = mongo_info['user']
    password = mongo_info['password']
    collection_name = mongo_info['collection_name']

    # Connect to the database/collection
    client = MongoClient(host=host, port=port, maxPoolSize=None)
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


def get_adsorption_docs(adsorbate=None, extra_projections=None, filters=None):
    '''
    A wrapper for the `aggregate` command that is tailored specifically for the
    `adsorption` collection.

    Args:
        adsorbate           [optional] A string of the adsorbate that you want
                            to get calculations for. If you pass nothing, then
                            we get all documents regardless of adsorbate.
        extra_projections   A dictionary with key/value pairings that
                            correspond to a new projection you want to fetch
                            and its location in the Mongo docs, respectively.
                            Refer to `gaspy.defaults.adsorption_projection` for
                            examples, or to the `$project` MongoDB operation.
        filters             A dictionary whose keys are the locations of
                            elements in the Mongo collection and whose values
                            are Mongo matching commands. For examples, look up
                            Mongo `match` commands. If this argument is `None`,
                            then it will fetch the default filters from
                            `gaspy.defaults.adsorption_filters`. If you want to
                            modify them, we suggest simply fetching that
                            object, modifying it, and then passing it here.
    Returns:
        cleaned_docs    A list of dictionaries whose key/value pairings are the
                        ones given by `gaspy.defaults.adsorption_projection`
                        and who meet the filtering criteria of
                        `gaspy.defaults.adsorption_filters`
    '''
    # Set the filtering criteria of the documents we'll be getting
    if filters is None:
        filters = defaults.adsorption_filters(adsorbate)
    if adsorbate:
        filters['adsorbate'] = adsorbate
    match = {'$match': filters}

    # Establish the information that'll be contained in the documents we'll be
    # getting. Also add anything the user asked for.
    projection = defaults.adsorption_projection()
    if extra_projections:
        for key, value in extra_projections.items():
            projection[key] = value
    project = {'$project': projection}

    # Get the documents and clean them up
    pipeline = [match, project]
    with get_mongo_collection(collection_tag='adsorption') as collection:
        print('Now pulling adsorption documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm(cursor)]
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=projection.keys())

    return cleaned_docs


def _clean_up_aggregated_docs(docs, expected_keys):
    '''
    This function takes a list of dictionaries and returns a new instance of
    the list without dictionaries that have missing keys or `None` as values.
    It assumes that dictionaries are flat, thus the `aggregated` part in the
    name.

    Arg:
        docs            A list of mongo documents, AKA a list of dicts, AKA a
                        list of JSONs
        expected_keys   The dict keys that that you expect to be in every
                        document.  If a document doesn't have the right keys or
                        has `None` for one of them, then it is deleted.
    Returns:
        clean_docs  A subset of the `docs` argument with
    '''
    # A hack to ignore the _id key, which is redundant with Mongo ID
    expected_keys = set(expected_keys)
    try:
        expected_keys.remove('_id')
    except KeyError:
        pass

    cleaned_docs = []
    for doc in docs:
        clean = True

        # Clean up documents that don't have the right keys
        if set(doc.keys()) != expected_keys:
            clean = False
        # Clean up documents that have `None` or '' as values
        for key, value in doc.items():
            if (value is None) or (value is ''):
                clean = False
            # Clean up documents that have no second-shell atoms
            if key == 'neighborcoord':
                for neighborcoord in value:  # neighborcoord looks like ['Cu:Cu-Cu-Cu-Cu', 'Cu:Cu-Cu-Cu-Cu']
                    neighbor, coord = neighborcoord.split(':')
                    if not coord:
                        clean = False
                        break
            if not clean:
                break

        if clean:
            cleaned_docs.append(doc)

    # Warn the user if we did not actually get any documents out the end.
    if not cleaned_docs:
        warnings.warn('We did not find any matching documents', RuntimeWarning)

    return cleaned_docs


def get_surface_docs(extra_projections=None, filters=None):
    '''
    A wrapper for `collection.aggregate` that is tailored specifically for the
    collection that's tagged `surface_energy`.

    Args:
        extra_projections   A dictionary with key/value pairings that
                            correspond to a new projection you want to fetch
                            and its location in the Mongo docs, respectively.
                            Refer to `gaspy.defaults.surface_projection` for
                            examples, or to the `$project` MongoDB operation.
        filters             A dictionary whose keys are the locations of
                            elements in the Mongo collection and whose values
                            are Mongo matching commands. For examples, look up
                            Mongo `match` commands. If this argument is `None`,
                            then it will fetch the default filters from
                            `gaspy.defaults.surface_filters`. If you want to
                            modify them, we suggest simply fetching that
                            object, modifying it, and then passing it here.
    Returns:
        docs    A list of dictionaries whose key/value pairings are the
                ones given by `gaspy.defaults.adsorption_projection` and who
                meet the filtering criteria of `gaspy.defaults.surface_filters`
    '''
    # Set the filtering criteria of the documents we'll be getting
    if filters is None:
        filters = defaults.surface_filters()
    match = {'$match': filters}

    # Establish the information that'll be contained in the documents we'll be getting
    # Also add anything the user asked for.
    projection = defaults.surface_projection()
    if extra_projections:
        for key, value in extra_projections.items():
            projection[key] = value
    project = {'$project': projection}

    # Get the documents and clean them up
    pipeline = [match, project]
    with get_mongo_collection(collection_tag='surface_energy') as collection:
        print('Now pulling surface documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm(cursor)]
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=projection.keys())

    # Some of our old code assumes that 'initial_configuration' is at the root
    # of each dictionary/document. But right now, it's buried in the
    # 'structure' key/value. Let's move it out and clean it up.
    for doc in docs:
        doc['initial_configuration'] = deepcopy(doc['structure']['initial_configuration'])
        del doc['structure']

    return cleaned_docs


def get_catalog_docs():
    '''
    A wrapper for `collection.aggregate` that is tailored specifically for the
    collection that's tagged `catalog`.

    Returns:
        docs    A list of dictionaries whose key/value pairings are the ones
                given by `gaspy.defaults.catalog_projection`
    '''
    # Reorganize the documents to the way we want
    projection = defaults.catalog_projection()
    project = {'$project': projection}

    # Pull and clean the documents
    pipeline = [project]
    docs = _pull_catalog_from_mongo(pipeline)
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=projection.keys())

    return cleaned_docs


def _pull_catalog_from_mongo(pipeline):
    '''
    Given a Mongo pipeline, get the catalog documents.

    Arg:
        pipeline    A list object containing the pipeline of Mongo operations
                    that you want to use during Mongo aggregation. Refer to
                    pymongo documentation on aggregation.
    Returns:
        docs    A list of dictionaries containing the catalog documents as per
                your pipeline.
    '''
    with get_mongo_collection(collection_tag='catalog_readonly') as collection:
        print('Now pulling catalog documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm(cursor)]
    return docs


def get_catalog_docs_with_predictions(latest_predictions=True):
    '''
    Nearly identical to `get_catalog_docs`, except it also pulls our surrogate
    modeling predictions for adsorption energies.

    Args:
        lastest_predictions Boolean indicating whether or not you want either
                            the latest predictions or all of them.
    Returns:
        docs    A list of dictionaries whose key/value pairings are the ones
                given by `gaspy.defaults.catalog_projection`, along with a
                'predictions' key that has the surrogate modeling predictions
                of adsorption energy.
    '''
    # Get the default catalog projection, then append the projections we
    # need to get the predictions.
    projection = defaults.catalog_projection()
    projection = _add_adsorption_energy_predictions_to_projection(projection, latest_predictions)
    projection = _add_orr_predictions_to_projection(projection, latest_predictions)

    # Get the documents
    project = {'$project': projection}
    pipeline = [project]
    docs = _pull_catalog_from_mongo(pipeline)

    # Clean the documents up
    expected_keys = set(defaults.catalog_projection())
    expected_keys.add('predictions')
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=expected_keys)

    return cleaned_docs


def _add_adsorption_energy_predictions_to_projection(projection, latest_predictions):
    '''
    This function will add particular keys to a `projection` dictionary that
    can be used in a Mongo projection to get the adsorption energy predictions
    from our catalog.

    Args:
        projection          A dictionary that you plan to pass as a projection
                            command to a pymongo collection aggregation.
        lastest_predictions Boolean indicating whether or not you want either
                            the latest predictions or all of them.
    '''
    # Figure out what type of json structure our adsorption energy predictions
    # have. We do that by looking at the structure of one random document. Note
    # that this assumes that all documents are structure identically.
    with get_mongo_collection('catalog') as collection:
        cursor = collection.aggregate([{"$sample": {"size": 1}}])
        example_doc = list(cursor)[0]
    predictions = example_doc['predictions']['adsorption_energy']
    adsorbates = set(predictions.keys())
    models = set(model for adsorbate in adsorbates for model in predictions[adsorbate])

    # Make a projection query that targets predictions for each combination of
    # adsorbate and model.
    for adsorbate in adsorbates:
        for model in models:
            data_location = 'predictions.adsorption_energy.%s.%s' % (adsorbate, model)
            if latest_predictions:
                projection[data_location] = {'$arrayElemAt': ['$'+data_location, -1]}
            else:
                projection[data_location] = '$'+data_location

    return projection


def _add_orr_predictions_to_projection(projection, latest_predictions):
    '''
    This function will add particular keys to a `projection` dictionary that
    can be used in a Mongo projection to get the ORR chemistry predictions from
    our catalog.

    Args:
        projection          A dictionary that you plan to pass as a projection
                            command to a pymongo collection aggregation.
        lastest_predictions Boolean indicating whether or not you want either
                            the latest predictions or all of them.
    '''
    # Figure out what type of json structure our adsorption energy predictions
    # have. We do that by looking at the structure of one random document. Note
    # that this assumes that all documents are structure identically.
    with get_mongo_collection('catalog') as collection:
        cursor = collection.aggregate([{"$sample": {"size": 1}}])
        example_doc = list(cursor)[0]
    predictions = example_doc['predictions']['orr_onset_potential_4e']
    models = set(predictions.keys())

    # Make a projection query that targets predictions for each model.
    for model in models:
        data_location = 'predictions.orr_onset_potential_4e.%s' % model
        if latest_predictions:
            projection[data_location] = {'$arrayElemAt': ['$'+data_location, -1]}
        else:
            projection[data_location] = '$'+data_location

    return projection


def get_unsimulated_catalog_docs(adsorbate,
                                 adsorbate_rotation_list=None,
                                 vasp_settings=None):
    '''
    Gets the same documents from `get_catalog_docs`, but then filters out all
    items that also show up in `get_adsorption_docs`, i.e., gets the catalog
    items that have not yet been simulated using our default settings.

    Args:
        adsorbate               A string of the adsorbate that you want to get
                                documents for.
        adsorbate_rotation_list A list of dictionaries with the 'psi', 'theta',
                                and 'phi' keys whose values are the rotation of
                                the adsorbate for the calculations you want to
                                check for. Each dictionary you add to this list
                                will add another set of sites to the output
                                list. If you add nothing, then we'll just check
                                for the default rotation only.
        vasp_settings           [optional] An OrderedDict containing the
                                default energy cutoff, VASP pseudo-potential
                                version number (pp_version), and
                                exchange-correlational settings. This should be
                                obtained (and modified, if necessary) from
                                `gaspy.defaults.adslab_settings()['vasp']`. If
                                `None`, then pulls default settings.
    Returns:
        docs    A list of dictionaries for various projection.
    '''
    # Python doesn't like mutable default arguments
    if vasp_settings is None:
        vasp_settings = defaults.adslab_settings()['vasp']
    if adsorbate_rotation_list is None:
        adsorbate_rotation_list = [defaults.adslab_settings()['rotation']]

    docs_catalog = get_catalog_docs()
    docs_catalog_with_rotation = _duplicate_docs_per_rotations(docs_catalog, adsorbate_rotation_list)
    docs_simulated = _get_attempted_adsorption_docs(adsorbate=adsorbate,
                                                    vasp_settings=vasp_settings)

    # Hash all of the documents, which we will use to check if something
    # in the catalog has been simulated or not
    print('Hashing catalog documents...')
    catalog_dict = {}
    for doc in tqdm(docs_catalog_with_rotation):
        hash_ = _hash_doc(doc, ignore_keys=['natoms'])
        catalog_dict[hash_] = doc

    # Filter out simulated documents
    for doc in docs_simulated:
        hash_ = _hash_doc(doc, ignore_keys=['adsorbate', 'energy'])
        catalog_dict.pop(hash_, None)
    docs = list(catalog_dict.values())

    return docs


def _duplicate_docs_per_rotations(docs, adsorbate_rotation_list):
    '''
    For each set of adsorbate rotations in the `adsorbate_rotation_list`
    argument, this function will copy the `docs` argument, add the adsorbate
    rotation, and then concatenate all of the modified lists together. Note
    that this function's name calls out "catalog" because it assumes that the
    documents have a structure identical to the one returned by
    `gaspy.gasdb.get_catalog_docs`.

    Args:
        docs                        A list of dictionaries (documents)
        adsorbate_rotation_list     A list of dictionaries whose keys are
                                    'phi', 'theta', and 'psi'.
    Returns:
        docs_with_rotation  Nearly identical to the `docs` argument, except it
                            is extended n-fold for each of the n adsorbate
                            rotations in the `adsorbate_rotation_list`
                            argument.
    '''
    # If we have more than one rotation, we'll need to copy and parse the
    # documents
    if len(adsorbate_rotation_list) > 1:
        docs_with_rotation = []
        for i, adsorbate_rotation in enumerate(adsorbate_rotation_list):
            print('Making catalog copy number %i of %i...'
                  % (i+1, len(adsorbate_rotation_list)))
            docs_copy = deepcopy(docs)  # To make sure we don't modify the parent docs
            for doc in tqdm(docs_copy):
                doc['adsorbate_rotation'] = adsorbate_rotation
            docs_with_rotation += docs_copy
        return docs_with_rotation

    # If we have only one rotation, we just need to add it. No copying
    # necessary.
    else:
        for doc in docs:
            doc['adsorbate_rotation'] = adsorbate_rotation_list[0]
        return docs


def _get_attempted_adsorption_docs(adsorbate, vasp_settings=None):
    '''
    A wrapper for `collection.aggregate` that is tailored specifically for the
    collection that's tagged `adsorption`. This differs from
    `get_adsorption_docs` in two ways:  1) it does not filter out "bad
    adsorptions" and 2) it takes projections based on initial configurations,
    not final, post-relaxation cofigurations. Thus this function finds
    everything that we've attempted.

    Args:
        adsorbate       A string indicating the adsorbate that you want to find
                        the attempted calculations for.
        vasp_settings   [optional] An OrderedDict containing the default energy
                        cutoff, VASP pseudo-potential version number
                        (pp_version), and exchange-correlational settings. This
                        should be obtained (and modified, if necessary) from
                        `gaspy.defaults.adslab_settings()['vasp']`. If `None`,
                        then pulls default settings.
    Returns:
        cleaned_docs    A list of dictionaries whose key/value pairings are the
                        ones given by `gaspy.defaults.adsorption_projection`.
                        Each document represents a calculation that we have
                        tried.
    '''
    # Get only the documents that have the right calculation settings and
    # adsorbates
    if vasp_settings is None:
        vasp_settings = defaults.adslab_settings()['vasp']
    filters = {'vasp_settings.%s' % setting: value
               for setting, value in vasp_settings.items()}
    if adsorbate:
        filters['adsorbate'] = adsorbate
    match = {'$match': filters}

    # Point the fingerprint at the unrelaxed structure instead of the relaxed
    # structure
    projection = defaults.adsorption_projection()
    projection['coordination'] = '$fp_init.coordination'
    projection['neighborcoord'] = '$fp_init.neighborcoord'
    projection['nextnearestcoordination'] = '$fp_init.nextnearestcoordination'
    # Get some extra adsorbate information out of Mongo
    projection['adsorbate_rotation'] = '$adsorbate_rotation'
    projection['adsorption_site'] = '$initial_adsorption_site'
    project = {'$project': projection}

    # Get the documents and clean them up
    pipeline = [match, project]
    with get_mongo_collection(collection_tag='adsorption') as collection:
        print('Now pulling adsorption documents for sites we have attempted...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm(cursor)]
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=projection.keys())

    return cleaned_docs


def _hash_doc(doc, ignore_keys=None, _return_hash=True):
    '''
    Hash a single Mongo document (AKA dictionary). This function currently
    assumes that all keys are strings and values are hashable.

    Args:
        doc             A single-layered dictionary/json/Mongo
                        document/whatever you call it.
        ignore_keys     A sequence of strings indicating the keys that you want
                        to ignore when hashing the document.
        _return_hashe   *For unit testing only!* If `False`, returns the
                        pre-hash object
    Returns:
        hash_   A hashed version of the document
    '''
    # Python doesn't do well with mutable default arguments
    if ignore_keys is None:
        ignore_keys = []
    # Make sure we don't modify the parent document
    doc = doc.copy()

    # Remove the keys we want to ignore
    ignore_keys = deepcopy(ignore_keys)
    ignore_keys.append('mongo_id')  # Because no two things will ever share a Mongo ID
    for key in ignore_keys:
        doc.pop(key, None)

    # Serialize the document into a string, then hash it
    serialized_doc = json.dumps(doc, sort_keys=True)
    if _return_hash:
        return hash(serialized_doc)

    # For unit testing, because hashes change between instances of Python
    else:
        return serialized_doc


def get_low_coverage_docs(adsorbate, model_tag=defaults.model()):
    '''
    Each surface has many possible adsorption sites. The site with the most
    negative adsorption energy (i.e., the strongest-binding site) will tend to
    be the dominating site at low adsorbate coverages. This function will find
    and return the low-coverage binding site for each surface. The adsorption
    energies used to find these sites are taken from DFT calculations whenever
    possible; when not possible, the energies are taken from model predictions.

    If a document came from the adsorption collection, then it will inherit the
    structure it got from `get_adsorption_docs`. If a document came from the
    catalog collection, then it will inherit the structure it got from
    `get_catalog_docs`. For ease-of-use, we also copied the predicted energies
    within each catalog document into the 'energy' key of the document (just
    like the adsorption documents have) so that all documents have energies in
    one consistent location.

    Args:
        adsorbate   A string indicating the adsorbate you want to get the
                    low-coverage sites for, e.g., 'CO' or 'H'
        model_tag   A string indicating which model you want to use when using
                    non-DFT, predicted energies. Check out the
                    `predictions.adsorption_energy` key in the catalog
                    documents for valid inputs. Note that these keys are
                    created by the `GASpy_regressions` submodule.
    Returns:
        docs    A dictionary whose keys are 4-tuples of the MPID, Miller index,
                shift, and top/bottom of a surface and whose values are the
                aggregated documents we get from either `get_adsorption_docs`
                or `get_catalog_docs`
    '''
    docs_dft = get_low_coverage_dft_docs(adsorbate=adsorbate)
    docs_ml = get_low_coverage_ml_docs(adsorbate=adsorbate, model_tag=model_tag)
    docs_dft_by_surface = {get_surface_from_doc(doc): doc for doc in docs_dft}
    docs_ml_by_surface = {get_surface_from_doc(doc): doc for doc in docs_ml}

    # For each ML-predicted surface, figure out if DFT supersedes it
    docs_by_surface = deepcopy(docs_ml_by_surface)
    for surface, doc_ml in docs_by_surface.items():

        try:
            # If DFT predicts a lower energy, then DFT supersedes ML
            doc_dft = docs_dft_by_surface[surface].copy()
            if doc_dft['energy'] < doc_ml['energy']:
                docs_by_surface[surface] = doc_dft
                docs_by_surface[surface]['DFT_calculated'] = True

            # If both DFT and ML predict the same site to have the lowest
            # energy, then DFT supersedes ML.
            else:
                ml_site_hash = _hash_doc(doc_ml, ignore_keys=['adsorption_site', 'natoms', 'predictions'])
                dft_site_hash = _hash_doc(doc_dft, ignore_keys=['energy', 'adsorbate'])
                if dft_site_hash == ml_site_hash:
                    docs_by_surface[surface] = doc_dft.copy()
                    docs_by_surface[surface]['DFT_calculated'] = True
                else:
                    docs_by_surface[surface]['DFT_calculated'] = False

        # EAFP in case we don't have any DFT data for a surface.
        except KeyError:
            docs_by_surface[surface]['DFT_calculated'] = False

    # If we somehow have a DFT site that is on a surface that's not even in our
    # catalog, then just add it. This might happen because we still have data
    # from old versions of our catalog.
    surfaces_ml = set(docs_ml_by_surface.keys())
    for surface, doc_dft in docs_dft_by_surface.items():
        if surface not in surfaces_ml:
            docs_by_surface[surface] = doc_dft
            docs_by_surface[surface]['DFT_calculated'] = True

    docs = list(docs_by_surface.values())
    return docs


def get_low_coverage_dft_docs(adsorbate, filters=None):
    '''
    This function is analogous to the `get_adsorption_docs` function, except it
    only returns documents that represent the low-coverage sites for each
    surface (i.e., the sites with the lowest energy for their respective
    surface).

    Arg:
        adsorbate   A string of the adsorbate that you want to get calculations
                    for.
        adsorbate   A list of the adsorbates that you need to be present in
                    each document's corresponding atomic structure. Note that
                    if you pass a list with two adsorbates, then you will only
                    get matches for structures with *both* of those adsorbates;
                    you will *not* get structures with only one of the
                    adsorbates.
        filters     A dictionary whose keys are the locations of elements in
                    the Mongo collection and whose values are Mongo matching
                    commands. For examples, look up Mongo `match` commands. If
                    this argument is `None`, then it will fetch the default
                    filters from `gaspy.defaults.adsorption_filters`. If you
                    want to modify them, we suggest simply fetching that
                    object, modifying it, and then passing it here.
    Returns:
        docs    A list of aggregated Mongo documents (i.e., dictionaries) from
                our `adsorption` Mongo collection that happen to have the
                lowest adsorption energy on their respective surfaces, as
                defined by their (mpid, miller, shift, top) values.
    '''
    # Set the filtering criteria of the documents we'll be getting
    if filters is None:
        filters = defaults.adsorption_filters(adsorbate)
    if adsorbate:
        filters['adsorbate'] = adsorbate
    match = {'$match': filters}

    # Get the standard document projection, then round the shift so that we can
    # group more easily. Credit to Vince Browdren on Stack Exchange
    projections = defaults.adsorption_projection()
    projections['shift'] = {'$subtract': [{'$add': ['$shift', 0.0049999999999999999]},
                                          {'$mod': [{'$add': ['$shift', 0.0049999999999999999]}, 0.01]}]}
    project = {'$project': projections}

    # Now order the documents so that the low-coverage sites come first (i.e.,
    # the one with the lowest energy)
    sort = {'$sort': {'energy': 1}}

    # Get the first document for each surface, which (after sorting) is the
    # low-coverage document
    grouping_fields = dict.fromkeys(projections.keys())
    for key in grouping_fields:
        grouping_fields[key] = {'$first': '$'+key}
    grouping_fields['_id'] = {'mpid': '$mpid',
                              'miller': '$miller',
                              'shift': '$shift',
                              'top': '$top'}
    group = {'$group': grouping_fields}

    # Pull the documents
    pipeline = [match, project, sort, group]
    with get_mongo_collection(collection_tag='adsorption') as collection:
        print('Now pulling low coverage adsorption documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm(cursor)]

    # Clean and return the documents
    for doc in docs:
        del doc['_id']
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=projections.keys())
    return cleaned_docs


def get_surface_from_doc(doc):
    '''
    Some of our functions parse by "surface", which we identify by mpid, Miller
    index, shift, and whether it's on the top or bottom of the slab. This
    helper function parses an aggregated/projected Mongo document for you and
    gives you back a tuple that contains these surface identifiers.

    Arg:
        doc     A Mongo document (dictionary) that contains the keys 'mpid',
                'miller', 'shift', and 'top'.
    Returns:
        surface A 4-tuple whose elements are the mpid, Miller index, shift, and
                a Boolean indicating whether the surface is on the top or
                bottom of the slab. Note that the Miller indices will be
                formatted as a string, and the shift will be rounded to 2
                decimal places.
    '''
    surface = (doc['mpid'], str(doc['miller']), round_(doc['shift'], 2), doc['top'])
    return surface


def round_(n, decimals=0):
    '''
    Python can't round for jack. We use someone else's home-brew to do
    rounding. Credit goes to David Amos
    (<https://realpython.com/python-rounding/#rounding-half-up>).
    '''
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


def get_low_coverage_ml_docs(adsorbate, model_tag=defaults.model()):
    '''
    This function is analogous to the `get_catalog_docs` function, except
    it only returns documents that represent the low-coverage sites for
    each surface (i.e., the sites with the lowest energy for their respective surface).

    Arg:
        adsorbate   A list of the adsorbates that you need to be present in each
                    document's corresponding atomic structure. Note that if you
                    pass a list with two adsorbates, then you will only get
                    matches for structures with *both* of those adsorbates; you
                    will *not* get structures with only one of the adsorbates.
        model_tag   A string indicating which model you want to use to predict
                    the adsorption energy.
    Returns:
        docs    A list of aggregated Mongo documents (i.e., dictionaries) from
                our `catalog` Mongo collection that happen to have the lowest
                ML-predicted adsorption energy on their respective surfaces, as
                defined by their (mpid, miller, shift, top) values.
    '''
    # Get the standard document projection, then round the shift so that we can
    # group more easily. Credit to Vince Browdren on Stack Exchange
    projections = defaults.catalog_projection()
    projections['shift'] = {'$subtract': [{'$add': ['$shift', 0.0049999999999999999]},
                                          {'$mod': [{'$add': ['$shift', 0.0049999999999999999]}, 0.01]}]}

    # Add the predictions
    data_location = 'predictions.adsorption_energy.%s.%s' % (adsorbate, model_tag)
    projections['energy'] = {'$arrayElemAt': [{'$arrayElemAt': ['$'+data_location, -1]}, 1]}
    project = {'$project': projections}

    # Now order the documents so that the low-coverage sites come first (i.e.,
    # the one with the lowest energy)
    sort = {'$sort': {'energy': 1}}

    # Get the first document for each surface, which (after sorting) is the
    # low-coverage document
    grouping_fields = dict.fromkeys(projections.keys())
    for key in grouping_fields:
        grouping_fields[key] = {'$first': '$'+key}
    grouping_fields['_id'] = {'mpid': '$mpid',
                              'miller': '$miller',
                              'shift': '$shift',
                              'top': '$top'}
    group = {'$group': grouping_fields}

    # Get the documents
    pipeline = [project, sort, group]
    with get_mongo_collection(collection_tag='catalog') as collection:
        print('Now pulling low coverage catalog documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm(cursor)]

    # Clean the documents up
    for doc in docs:
        del doc['_id']
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=projections.keys())
    return cleaned_docs


def purge_adslabs(fwids):
    '''
    This function will "purge" adsorption calculations from our database by
    removing it from our Mongo collections and defusing them within FireWorks.

    Arg:
        fwids   The FireWorks IDs of the calculations in question
    '''
    lpad = get_launchpad()

    print('Defusing FWs...')
    for fwid in tqdm(fwids):
        lpad.defuse_fw(fwid)

    print('Removing FWs from atoms collection...')
    with get_mongo_collection('atoms') as collection:
        collection.delete_many({'fwid': {'$in': fwids}})

    print('Removing FWs from adsorption collection...')
    with get_mongo_collection('adsorption') as collection:
        collection.delete_many({'fwids.slab+adsorbate': {'$in': fwids}})
