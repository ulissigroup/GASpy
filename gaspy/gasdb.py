''' These tools form gaspy's API to its databases '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import warnings
import copy
import json
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
                            `gaspy.defaults.adsorption_fingerprints` for examples,
                            or to the '$project' MongoDB aggregation command.
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
    # Set the filtering criteria of the documents we'll be getting
    if not filters:
        filters = defaults.adsorption_filters(adsorbates)
    if adsorbates:
        filters['processed_data.calculation_info.adsorbate_names'] = adsorbates
    match = {'$match': filters}

    # Establish the information that'll be contained in the documents we'll be getting.
    # Also add anything the user asked for.
    fingerprints = defaults.adsorption_fingerprints()
    if extra_fingerprints:
        for key, value in extra_fingerprints.items():
            fingerprints[key] = value
    project = {'$project': fingerprints}

    # Get the documents and clean them up
    pipeline = [match, project]
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
    # A hack to ignore the _id key, which is redundant with Mongo ID
    expected_keys = set(expected_keys)
    try:
        expected_keys.remove('_id')
    except KeyError:
        pass

    cleaned_docs = []
    for doc in docs:
        is_clean = True

        # Clean up documents that don't have the right keys
        if set(doc.keys()) != expected_keys:
            is_clean = False
        # Clean up documents that have `None` or '' as values
        for key, value in doc.items():
            if (value is None) or (value is ''):
                is_clean = False
            # Clean up documents that have no second-shell atoms
            if key == 'neighborcoord':
                for neighborcoord in value:  # neighborcoord looks like ['Cu:Cu-Cu-Cu-Cu', 'Cu:Cu-Cu-Cu-Cu']
                    neighbor, coord = neighborcoord.split(':')
                    if not coord:
                        is_clean = False
                        break
            if not is_clean:
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

    Args:

        lastest_predictions Boolean indicating whether or not you want either
                            the latest predictions or all of them.
    Returns:
        docs    A list of dictionaries whose key/value pairings are the
                ones given by `gaspy.defaults.catalog_fingerprints`
    '''
    # Reorganize the documents to the way we want
    fingerprints = defaults.catalog_fingerprints()
    project = {'$project': fingerprints}

    # Pull and clean the documents
    pipeline = [project]
    with get_mongo_collection(collection_tag='relaxed_bulk_catalog_readonly') as collection:
        print('Now pulling catalog documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm.tqdm(cursor)]
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=fingerprints.keys())

    return cleaned_docs


def get_catalog_docs_with_predictions(adsorbates=None, chemistries=None, models=['model0'], latest_predictions=True):
    '''
    Nearly identical to `get_catalog_docs`, except it also pulls our surrogate
    modeling predictions for adsorption energy.

    Args:
        adsorbates          A list of strings indicating which sets of adsorbates
                            you want to get adsorption energy predictions for,
                            e.g., ['CO', 'H'] or ['O', 'OH', 'OOH'].
        chemistries         A list of strings for top-level predictions to also pull
                            e.g. ['orr_onset_potential_4e']
        models              A list of strings indicating which models whose
                            predictions you want to get.
        lastest_predictions Boolean indicating whether or not you want either
                            the latest predictions or all of them.
    Returns:
        docs    A list of dictionaries whose key/value pairings are the
                ones given by `gaspy.defaults.catalog_fingerprints`, along
                with a 'predictions' key that has the surrogate modeling
                predictions of adsorption energy.
    '''
    if isinstance(models, str):
        raise SyntaxError('The models argument must be a sequence of strings where each '
                          'element is a model name. Do not pass a single string.')

    # Reorganize the documents to the way we want
    fingerprints = defaults.catalog_fingerprints()

    # Get the prediction data
    for adsorbate in adsorbates:
        for model in models:
            data_location = 'predictions.adsorption_energy.%s.%s' % (adsorbate, model)
            if latest_predictions:
                fingerprints[data_location] = {'$arrayElemAt': ['$'+data_location, -1]}
            else:
                fingerprints[data_location] = '$'+data_location

    for chemistry in chemistries:
        for model in models:
            data_location = 'predictions.%s.%s' % (chemistry, model)
            if latest_predictions:
                fingerprints[data_location] = {'$arrayElemAt': ['$'+data_location, -1]}
            else:
                fingerprints[data_location] = '$'+data_location

    # Get the documents
    project = {'$project': fingerprints}

    print(project)
    pipeline = [project]
    with get_mongo_collection(collection_tag='relaxed_bulk_catalog_readonly') as collection:
        print('Now pulling catalog documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm.tqdm(cursor)]

    # Clean the documents up
    expected_keys = set(fingerprints.keys())
    for adsorbate in adsorbates:
        for model in models:
            expected_keys.remove('predictions.adsorption_energy.%s.%s' % (adsorbate, model))

    for chemistry in chemistries:
        for model in models:
            expected_keys.remove('predictions.%s.%s' % (chemistry, model))

    expected_keys.add('predictions')
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=expected_keys)

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
                `gaspy.defaults.surface_filters`
    '''
    # Set the filtering criteria of the documents we'll be getting
    if not filters:
        filters = defaults.surface_filters()
    match = {'$match': filters}

    # Establish the information that'll be contained in the documents we'll be getting
    # Also add anything the user asked for.
    fingerprints = defaults.surface_fingerprints()
    if extra_fingerprints:
        for key, value in extra_fingerprints.items():
            fingerprints[key] = value
    project = {'$project': fingerprints}

    # Get the documents and clean them up
    pipeline = [match, project]
    with get_mongo_collection(collection_tag='surface_energy') as collection:
        print('Now pulling surface documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm.tqdm(cursor)]
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=fingerprints.keys())

    return cleaned_docs


def get_unsimulated_catalog_docs(adsorbates, adsorbate_rotation_list=None):
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
    if adsorbate_rotation_list is None:
        adsorbate_rotation_list = [copy.deepcopy(defaults.ROTATION)]

    docs_catalog = get_catalog_docs()
    docs_catalog_with_rotation = _duplicate_docs_per_rotations(docs_catalog, adsorbate_rotation_list)
    docs_simulated = _get_attempted_adsorption_docs(adsorbates=adsorbates)

    # Round all of the shift values so that they match more easily
    for doc in docs_catalog_with_rotation + docs_simulated:
        doc['shift'] = round(doc['shift'], 2)

    # Hash all of the documents, which we will use to check if something
    # in the catalog has been simulated or not
    print('Hashing catalog documents...')
    catalog_dict = {}
    for doc in tqdm.tqdm(docs_catalog_with_rotation):
        hash_ = _hash_doc(doc, ignore_keys=['mongo_id', 'formula'])
        catalog_dict[hash_] = doc

    # Filter out simulated documents
    for doc in docs_simulated:
        hash_ = _hash_doc(doc, ignore_keys=['mongo_id',
                                            'formula',
                                            'energy',
                                            'adsorbates',
                                            'adslab_calculation_date'])
        try:
            del catalog_dict[hash_]
        except KeyError:
            pass
    docs = list(catalog_dict.values())
    return docs


def _get_attempted_adsorption_docs(adsorbates=None, calc_settings=None):
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
        calc_settings   [optional] An OrderedDict containing the default energy
                        cutoff, VASP pseudo-potential version number (pp_version),
                        and exchange-correlational settings. This should be obtained
                        (and modified, if necessary) from the
                        `gaspy.default.calc_settings` function. If `None`, then
                        pulls default settings.
    Returns:
        cleaned_docs    A list of dictionaries whose key/value pairings are the
                        ones given by `gaspy.defaults.adsorption_fingerprints`
                        and who meet the filtering criteria of
                        `gaspy.defaults.adsorption_filters`
    '''
    # Get only the documents that have the right calculation settings and adsorbates
    filters = {}
    if not calc_settings:
        calc_settings = defaults.calc_settings(defaults.ADSLAB_ENCUT)
        filters['processed_data.vasp_settings.gga'] = calc_settings['gga']
    if adsorbates:
        filters['processed_data.calculation_info.adsorbate_names'] = adsorbates
    match = {'$match': filters}

    # Establish the information that'll be contained in the documents we'll be getting
    fingerprints = defaults.adsorption_fingerprints()
    fingerprints['coordination'] = '$processed_data.fp_init.coordination'
    fingerprints['neighborcoord'] = '$processed_data.fp_init.neighborcoord'
    fingerprints['nextnearestcoordination'] = '$processed_data.fp_init.nextnearestcoordination'
    fingerprints['adsorbate_data'] = {'$arrayElemAt': ['$processed_data.calculation_info.adsorbates', 0]}
    project = {'$project': fingerprints}

    # Do a second projection to get the rotation and site information out, but keep everything else
    fingerprints_with_rotation = dict.fromkeys(fingerprints)
    del fingerprints_with_rotation['adsorbate_data']
    for key in fingerprints_with_rotation:
        fingerprints_with_rotation[key] = '$' + key
    fingerprints_with_rotation['adsorbate_rotation'] = '$adsorbate_data.adsorbate_rotation'
    fingerprints_with_rotation['adsorption_site'] = '$adsorbate_data.adsorption_site'
    project_rotation = {'$project': fingerprints_with_rotation}

    # Get the documents and clean them up
    pipeline = [match, project, project_rotation]
    with get_mongo_collection(collection_tag='adsorption') as collection:
        print('Now pulling adsorption documents for sites we have attempted...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm.tqdm(cursor)]
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=fingerprints_with_rotation.keys())

    return cleaned_docs


def _duplicate_docs_per_rotations(docs, adsorbate_rotation_list):
    '''
    For each set of adsorbate rotations in the `adsorbate_rotation_list`
    argument, this function will copy the `docs` argument, add the adsorbate
    rotation, and then concatenate all of the modified lists together.
    Note that this function's name calls out "catalog" because it assumes
    that the documentns have a structure identical to the one returned
    by `gaspy.gasdb.get_catalog_docs`.

    Args:
        docs                        A list of dictionaries (documents)
        adsorbate_rotation_list     A list of dictionaries whose keys are
                                    'phi', 'theta', and 'psi'.
    Returns:
        docs_with_rotation  Nearly identical to the `docs` argument,
                            except it is extended n-fold for each
                            of the n adsorbate rotations in the
                            `adsorbate_rotation_list` argument.
    '''
    docs_with_rotation = []
    for i, adsorbate_rotation in enumerate(adsorbate_rotation_list):
        print('Making catalog copy number %i of %i...' % (i+1, len(adsorbate_rotation_list)))
        docs_copy = copy.deepcopy(docs)     # To make sure we don't modify the parent docs
        for doc in docs_copy:
            doc['adsorbate_rotation'] = adsorbate_rotation
        docs_with_rotation += docs_copy
    return docs_with_rotation


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
              for doc in tqdm.tqdm(docs)]
    return hashes


def _hash_doc(doc, ignore_keys=None, _return_hash=True):
    '''
    Hash a single Mongo document (AKA dictionary). This function currently assumes
    that all keys are strings and values are hashable.

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

    # Remove the keys we want to ignore
    doc = doc.copy()    # To make sure we don't modify the parent document
    ignore_keys.append('mongo_id')  # Because no two things will ever share a Mongo ID
    ignore_keys.append('adslab_calculation_date')   # Can't hash datetime objects
    for key in ignore_keys:
        try:
            del doc[key]
        except KeyError:
            pass

    # Serialize the document into a string, then hash it
    serialized_doc = json.dumps(doc, sort_keys=True)
    if _return_hash:
        return hash(serialized_doc)

    # For unit testing, because hashes change between instances of Python
    else:
        return serialized_doc


def get_low_coverage_docs(adsorbates, model_tag=defaults.MODEL):
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
        docs    A dictionary whose keys are 4-tuples of the MPID, Miller index, shift,
                and top/bottom of a surface and whose values are the aggregated
                documents we get from either `get_adsorption_docs` or `get_catalog_docs`
    '''
    docs_dft = get_low_coverage_dft_docs(adsorbates=adsorbates)
    docs_ml = get_low_coverage_ml_docs(adsorbates=adsorbates, model_tag=model_tag)

    # For each ML-predicted surface, figure out if DFT supersedes it
    docs = copy.deepcopy(docs_ml)
    for surface, doc_ml in docs.items():

        try:
            # If DFT predicts a lower energy, then DFT supersedes ML
            doc_dft = docs_dft[surface].copy()
            if doc_dft['energy'] < doc_ml['energy']:
                docs[surface] = doc_dft
                docs[surface]['DFT_calculated'] = True

            # If both DFT and ML predict the same site to have the lowest energy,
            # then DFT supersedes ML.
            else:
                ml_site_hash = _hash_doc(doc_ml, ignore_keys=['mongo_id',
                                                              'formula',
                                                              'adsorption_site',
                                                              'predictions'])
                dft_site_hash = _hash_doc(doc_dft, ignore_keys=['mongo_id',
                                                                'formula',
                                                                'energy',
                                                                'adsorbates',
                                                                'adslab_calculation_date'])
                if dft_site_hash == ml_site_hash:
                    docs[surface] = doc_dft.copy()
                    docs[surface]['DFT_calculated'] = True
                else:
                    docs[surface]['DFT_calculated'] = False

        # EAFP in case we don't have any DFT data for a surface.
        except KeyError:
            docs[surface]['DFT_calculated'] = False

    # If we somehow have a DFT site that is on a surface that's not
    # even in our catalog, then just add it
    surfaces_ml = set(docs_ml.keys())
    for surface, doc_dft in docs_dft.items():
        if surface not in surfaces_ml:
            docs[surface] = doc_dft
            docs[surface]['DFT_calculated'] = True

    return docs


def get_low_coverage_dft_docs(adsorbates, filters=None):
    '''
    This function is analogous to the `get_adsorption_docs` function, except
    it only returns documents that represent the low-coverage sites for
    each surface (i.e., the sites with the lowest energy for their respective surface).

    Arg:
        adsorbates  A list of the adsorbates that you need to be present in each
                    document's corresponding atomic structure. Note that if you
                    pass a list with two adsorbates, then you will only get
                    matches for structures with *both* of those adsorbates; you
                    will *not* get structures with only one of the adsorbates.
        filters     A dictionary whose keys are the locations of elements
                    in the Mongo collection and whose values are Mongo
                    matching commands. For examples, look up Mongo `match`
                    commands. If this argument is `None`, then it will
                    fetch the default filters from
                    `gaspy.defaults.adsorption_filters`. If you want to modify
                    them, we suggest simply fetching that object, modifying it,
                    and then passing it here.
    Returns:
        docs_by_surface A dictionary whose keys are a 4-tuple indicating what a surface
                        is (mpid, miller, shift, top) and whose values are the projected
                        Mongo document for the low coverage site on that surface.
    '''
    # Set the filtering criteria of the documents we'll be getting
    if not filters:
        filters = defaults.adsorption_filters(adsorbates)
    if adsorbates:
        filters['processed_data.calculation_info.adsorbate_names'] = adsorbates
    match = {'$match': filters}

    # Project the catalog items onto what a "adsorption document" would normally look like.
    fingerprints = defaults.adsorption_fingerprints()
    # Round the shift so that we can group more easily. Credit to Vince Browdren on Stack Exchange
    fingerprints['shift'] = {'$subtract': [{'$add': ['$processed_data.calculation_info.shift',
                                                     0.0049999999999999999]},
                                           {'$mod': [{'$add': ['$processed_data.calculation_info.shift',
                                                               0.0049999999999999999]},
                                                     0.01]}]}
    project = {'$project': fingerprints}

    # Now order the documents so that the low-coverage sites come first
    # (i.e., the one with the lowest energy)
    sort = {'$sort': {'energy': 1}}

    # Get the first document for each surface, which (after sorting) is the low-coverage document
    grouping_fields = dict.fromkeys(fingerprints.keys())
    for key in grouping_fields:
        grouping_fields[key] = {'$first': '$'+key}
    grouping_fields['_id'] = {'mpid': '$mpid',
                              'miller': '$miller',
                              'shift': '$shift',
                              'top': '$top'}
    group = {'$group': grouping_fields}

    # Get the documents
    pipeline = [match, project, sort, group]
    with get_mongo_collection(collection_tag='adsorption') as collection:
        print('Now pulling low coverage adsorption documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm.tqdm(cursor)]

    # Clean the documents up
    for doc in docs:
        del doc['_id']
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=fingerprints.keys())

    # Structure the output into a surface:document dictionary
    docs_by_surface = {}
    for doc in cleaned_docs:
        surface = _get_surface_from_doc(doc)
        docs_by_surface[surface] = doc
    return docs_by_surface


def _get_surface_from_doc(doc):
    '''
    Some of our functions parse by "surface", which we identify by mpid, Miller index,
    shift, and whether it's on the top or bottom of the slab. This helper function
    parses an aggregated/projected Mongo document for you and gives you back a tuple
    that contains these surface identifiers.

    Arg:
        doc     A Mongo document (dictionary) that contains the keys 'mpid', 'miller',
                'shift', and 'top'.
    Returns:
        surface A 4-tuple whose elements are the mpid, Miller index, shift, and a
                boolean indicating whether the surface is on the top or bottom of
                the slab. Note that the Miller indices will be formatted as a string,
                and the shift will be rounded to 2 decimal places.
    '''
    surface = (doc['mpid'], str(doc['miller']), round(doc['shift'], 2), doc['top'])
    return surface


def get_low_coverage_ml_docs(adsorbates, model_tag=defaults.MODEL):
    '''
    This function is analogous to the `get_catalog_docs` function, except
    it only returns documents that represent the low-coverage sites for
    each surface (i.e., the sites with the lowest energy for their respective surface).

    Arg:
        adsorbates  A list of the adsorbates that you need to be present in each
                    document's corresponding atomic structure. Note that if you
                    pass a list with two adsorbates, then you will only get
                    matches for structures with *both* of those adsorbates; you
                    will *not* get structures with only one of the adsorbates.
        model_tag   A string indicating which model you want to use to predict
                    the adsorption energy.
    Returns:
        docs_by_surface A dictionary whose keys are a 4-tuple indicating what a surface
                        is (mpid, miller, shift, top) and whose values are the projected
                        Mongo document for the low coverage site on that surface.
    '''
    # Project the catalog items onto what a "catalog document" would normally look like.
    fingerprints = defaults.catalog_fingerprints()
    # Round the shift so that we can group more easily. Credit to Vince Browdren on Stack Exchange
    fingerprints['shift'] = {'$subtract': [{'$add': ['$processed_data.calculation_info.shift',
                                                     0.0049999999999999999]},
                                           {'$mod': [{'$add': ['$processed_data.calculation_info.shift',
                                                               0.0049999999999999999]},
                                                     0.01]}]}
    # Add the predictions
    data_location = 'predictions.adsorption_energy.%s.%s' % (adsorbates[0], model_tag)
    fingerprints['energy'] = {'$arrayElemAt': [{'$arrayElemAt': ['$'+data_location, -1]}, 1]}
    project = {'$project': fingerprints}

    # Now order the documents so that the low-coverage sites come first
    # (i.e., the one with the lowest energy)
    sort = {'$sort': {'energy': 1}}

    # Get the first document for each surface, which (after sorting) is the low-coverage document
    grouping_fields = dict.fromkeys(fingerprints.keys())
    for key in grouping_fields:
        grouping_fields[key] = {'$first': '$'+key}
    grouping_fields['_id'] = {'mpid': '$mpid',
                              'miller': '$miller',
                              'shift': '$shift',
                              'top': '$top'}
    group = {'$group': grouping_fields}

    # Get the documents
    pipeline = [project, sort, group]
    with get_mongo_collection(collection_tag='relaxed_bulk_catalog') as collection:
        print('Now pulling low coverage catalog documents...')
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True)
        docs = [doc for doc in tqdm.tqdm(cursor)]

    # Clean the documents up
    for doc in docs:
        del doc['_id']
    docs = _clean_up_aggregated_docs(docs, expected_keys=fingerprints.keys())

    # Structure the output into a surface:document dictionary
    docs_by_surface = {}
    for doc in docs:
        surface = _get_surface_from_doc(doc)
        docs_by_surface[surface] = doc
    return docs_by_surface


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


def purge_adslab(fwid):
    '''
    This function will "purge" an adsorption calculation from our database
    by removing it from our Mongo collections and defusing it within FireWorks.

    Arg:
        fwid    The FireWorks ID of the calculation in question
    '''
    lpad = utils.get_lpad()
    lpad.defuse_fw(fwid)

    with get_mongo_collection('atoms') as collection:
        collection.delete_one({'fwid': fwid})

    with get_mongo_collection('adsorption') as collection:
        collection.delete_one({'processed_data.FW_info.slab+adsorbate': fwid})
