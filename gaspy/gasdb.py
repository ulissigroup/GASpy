''' These tools form gaspy's API to its databases '''

import warnings
import json
import pickle
import glob
from itertools import islice
from bson.objectid import ObjectId
from multiprocessing import Pool
import numpy as np
import tqdm
from pymongo import MongoClient
from pymongo.collection import Collection
from ase.io.png import write_png
from . import defaults, utils, vasp_functions
from .mongo import make_atoms_from_doc


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
    # Add the enter and exit methods to pymongo.collection
    mongo_info = utils.read_rc()['mongo_info'][collection_tag]
    host = mongo_info['host']
    port = int(mongo_info['port'])
    database_name = mongo_info['database']
    user = mongo_info['user']
    password = mongo_info['password']
    collection_name = mongo_info['collection_name']

    # Access the client and authenticate
    client = MongoClient(host=host, port=port)
    database = getattr(client, database_name)
    database.authenticate(user, password)
    collection = ConnectableCollection(database=database, name=collection_name)

    return collection


# Make an extendeded version of the pymongo.collection.Collection class that can be open and closed
class ConnectableCollection(Collection):
    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, exception_traceback):   # noqa: E301
        self.database.client.close()


def get_adsorption_docs(adsorbates=None, _collection_tag='adsorption'):
    '''
    A wrapper for `aggregate_docs` that is tailored specifically for the
    collection that's tagged `adsorption`. If you don't like the way that
    this function gets documents, you can do it yourself by using
    `find_docs` or `aggregate_docs` and any of the fingerprints/filters
    in the `gaspy.defaults` submodule.

    Args:
        adsorbates      [optional] A list of the adsorbates that you need to
                        be present in each document's corresponding atomic
                        structure. Note that if you pass a list with two adsorbates,
                        then you will only get matches for structures with *both*
                        of those adsorbates; you will *not* get structures
                        with only one of the adsorbates. If you pass nothing, then we
                        get all documents regardless of adsorbates.
        _collection_tag *For unit testing only.* Do not change this.
    Returns:
        docs    A list of dictionaries whose key/value pairings are the
                ones given by `gaspy.defaults.adsorption_fingerprints`
                and who meet the filtering criteria of
                `gaspy.defaults.adsorption_filters`
    '''
    # Establish the information that'll be contained in the documents we'll be getting
    fingerprints = defaults.adsorption_fingerprints()
    group = {'$group': {'_id': fingerprints}}

    # Set the filtering criteria of the documents we'll be getting
    filters = defaults.adsorption_filters()
    if adsorbates:
        filters['processed_data.calculation_info.adsorbate_names'] = adsorbates
    match = {'$match': filters}

    # Get the documents and clean them up
    pipeline = [match, group]
    with get_mongo_collection(collection_tag=_collection_tag) as collection:
        cursor = collection.aggregate(pipeline=pipeline, allowDiskUse=True, useCursor=True)
        print('Now pulling adsorption documents...')
        docs = [doc for doc in tqdm.tqdm(cursor)]
    cleaned_docs = _clean_up_aggregated_docs(docs, expected_keys=fingerprints.keys())

    return cleaned_docs


def _clean_up_aggregated_docs(docs, expected_keys):
    '''
    Some of the Mongo documents we get are just plain dirty and end up mucking up
    downstream pipelines. This function cleans them up so that they're less
    likely to break something.

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
            break
        # Clean up documents that have `None` as values
        for key, value in doc.items():
            if value is None:
                is_clean = False
                break

        if is_clean:
            cleaned_docs.append(doc)

    # Warn the user if we did not actually get any documents out the end.
    if not cleaned_docs:
        warnings.warn('We did not find any matching documents', RuntimeWarning)

    return cleaned_docs


def hash_docs(docs, ignore_keys=None, _return_hashes=True):
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

    # Add the mongo ID to the list of ignored keys, because that'll always yield a different
    # hash. Then turn it into a set to speed up searching.
    ignore_keys.append('mongo_id')

    # Hash with a progress bar
    print('Now hashing documents...')
    hashes = [hash_doc(doc=doc, ignore_keys=ignore_keys, _return_hash=_return_hashes)
              for doc in tqdm.tqdm(docs)]
    return hashes


def hash_doc(doc, ignore_keys=None, _return_hash=True):
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


def split_catalog(ads_docs, cat_docs):
    '''
    The same as `get_docs`, but with already-simulated entries filtered out.
    This is best done right after you pull the docs; don't modify them too much
    unless you know what you're doing.

    Inputs:
        ads_docs    A list of docs coming from get_docs that correspond to the adsorption database
        cat_docs    A list of docs coming from get_docs that correspond to the catalog database
    Output:
        sim_inds    A list of indices in cat_docs that have been simulation (match sites
                    in ads_docs)
        unsim_inds  A list of indices in cat_docs that have not been simulated (do not
                    match sites in ads_docs)
    '''
    # Hash the docs so that we can filter out any items in the catalog that we have already relaxed.
    # Note that we ignore any energy values in the adsorbate collection, because there are no
    # energy values in the catalog.
    ads_hashes = hash_docs(ads_docs, ignore_keys=['energy', 'formula', 'shift', 'top'])
    cat_hashes = hash_docs(cat_docs, ignore_keys=['formula', 'shift', 'top'])
    unsim_hashes = set(cat_hashes)-set(ads_hashes)

    # Perform the filtering while simultaneously populating the `docs` output.
    unsim_inds = [ind for ind, cat_hash in tqdm.tqdm(enumerate(cat_hashes))
                  if cat_hash in unsim_hashes]
    sim_inds = [ind for ind, cat_hash in tqdm.tqdm(enumerate(cat_hashes))
                if cat_hash not in unsim_hashes]
    return set(sim_inds), set(unsim_inds)


def unsimulated_catalog(adsorbates, calc_settings=None, vasp_settings=None,
                        fingerprints=None, max_atoms=None):
    '''
    The same as `get_docs`, but with already-simulated entries filtered out

    Inputs:
        adsorbates      A list of strings indicating the adsorbates that you want to make a
                        prediction for.
        calc_settings   The calculation settings that we want to filter by. If we are using
                        something other than beef-vdw or rpbe, then we need to do some
                        more hard-coding here so that we know what in the catalog
                        can work as a flag for this new calculation method.
        vasp_settings   The vasp settings that we want to filter by.
        fingerprints    A dictionary of fingerprints and their locations in our
                        mongo documents. This is how we can pull out more (or less)
                        information from our database.
        max_atoms       The maximum number of atoms in the system that you want to pull
    Output:
        docs    A list of dictionaries for various fingerprints. Useful for
                creating lists of GASpy `parameters` dictionaries.
    '''
    # Default value for fingerprints. Since it's a mutable dictionary, we define it
    # down here instead of in the __init__ line.
    if not fingerprints:
        fingerprints = defaults.fingerprints()

    # Fetch mongo docs for our results and catalog databases so that we can
    # start filtering out cataloged sites that we've already simulated.
    with get_mongo_collection('adsorption') as ads_client:
        ads_docs = get_docs(ads_client, 'adsorption',
                            calc_settings=calc_settings,
                            vasp_settings=vasp_settings,
                            fingerprints=fingerprints,
                            adsorbates=adsorbates)
    with get_mongo_collection('catalog_readonly') as cat_client:
        cat_docs = get_docs(cat_client, 'catalog', fingerprints=fingerprints, max_atoms=max_atoms)

    # Use the `split_catalog` function to find the indices in `cat_docs` that correspond
    # with item that we have not yet simulated. Then use that list to build the documents.
    _, unsim_inds = split_catalog(ads_docs, cat_docs)
    docs = [cat_docs[i] for i in unsim_inds]
    return docs


def remove_duplicates():
    '''
    This function will find duplicate entries in the adsorption and atoms collections
    and delete them.
    '''
    # Get the FW info for everything in adsorption DB
    ads_client = get_mongo_collection('adsorption')
    ads_docs = list(ads_client.db.adsorption.find({}, {'processed_data.FW_info.slab+adsorbate': 1, '_id': 1}))

    # Find all of the unique slab+adsorbate FW ID's
    uniques, inverse, counts = np.unique([doc['processed_data']['FW_info']['slab+adsorbate']
                                          for doc in ads_docs],
                                         return_counts=True, return_inverse=True)

    # For each unique FW ID, see if there is more than one entry.
    # If so, remove all but the first instance
    for ind, count in enumerate(counts):
        if count > 1:
            matching = np.where(inverse == ind)[0]
            for match in matching[1:]:
                mongo_id = ads_docs[match]['_id']
                fwid = ads_docs[match]['processed_data']['FW_info']['slab+adsorbate']
                ads_client.db.adsorption.remove({'_id': mongo_id})
                print('Just removed Mongo item %s (duplicate for FWID %s) from adsorption collection' % (mongo_id, fwid))

    # Do it all again, but for the AuxDB (AKA the "atoms" collection)
    atoms_client = get_mongo_collection('atoms')
    atoms_docs = list(atoms_client.db.atoms.find({}, {'fwid': 1, '_id': 1}))

    # Find all of the unique slab+adsorbate FW ID's
    uniques, inverse, counts = np.unique([doc['fwid']
                                          for doc in atoms_docs],
                                         return_counts=True, return_inverse=True)

    # For each unique FW ID, see if there is more than one entry.
    # If so, remove all but the first instance
    for ind, count in enumerate(counts):
        if count > 1:
            matching = np.where(inverse == ind)[0]
            print(matching)
            for match in matching[1:]:
                mongo_id = atoms_docs[match]['_id']
                fwid = atoms_docs[match]['fwid']
                atoms_client.db.atoms.remove({'_id': atoms_docs[match]['_id']})
                print('Just removed Mongo item %s (duplicate for FWID %s) from atoms collection' % (mongo_id, fwid))


def dump_adsorption_to_json(fname):
    '''
    Dump the adsorption collection to a json file

    Input:
        fname   A string indicating the file name you want to dump to
    '''
    # Define the data that we want to pull out of Mongo.
    # The defaults gives us miscellaneous useful information.
    # The 'results' and 'atoms' are necessary to turn the doc into atoms.
    fingerprints = defaults.fingerprints(simulated=True)
    fingerprints['results'] = '$results'
    fingerprints['atoms'] = '$atoms'
    # Pull out only documents that had "good" relaxations
    doc_filters = defaults.filters_for_adsorption_docs()
    docs = get_docs(fingerprints=fingerprints, **doc_filters)

    # Preprocess the docs before dumping
    for doc in docs:
        # Make the documents json serializable
        del doc['mongo_id']
        time_object = doc['adslab_calculation_date']
        time = time_object.isoformat()
        doc['adslab_calculation_date'] = time

        # Put the atoms hex in for others to be able to decode it
        atoms = make_atoms_from_doc(doc)
        _hex = vasp_functions.atoms_to_hex(atoms)
        doc['atoms_hex'] = _hex

    # Save
    with open(fname, 'w') as file_handle:
        json.dump(docs, file_handle)


# TODO:  Comment and clean up everything below here
ads_dict = defaults.adsorbates_dict()
del ads_dict['']
del ads_dict['U']
ads_to_run = ads_dict.keys()
ads_to_run = ['CO', 'H']
dump_dir = '/global/cscratch1/sd/zulissi/GASpy_DB/images/'
databall_template = {'CO': '/project/projectdirs/m2755/GASpy/GASpy_regressions/cache/predictions/CO2RR_predictions_TPOT_FEATURES_coordatoms_chemfp0_neighbors_chemfp0_RESPONSES_energy_BLOCKS_adsorbate.pkl',
                     'H': '/project/projectdirs/m2755/GASpy/GASpy_regressions/cache/predictions/HER_predictions_TPOT_FEATURES_coordatoms_chemfp0_neighbors_chemfp0_RESPONSES_energy_BLOCKS_adsorbate.pkl'}


def writeImages(input):
    doc, adsorbate = input
    atoms = make_atoms_from_doc(doc)
    slab = atoms.copy()
    ads_pos = slab[0].position
    del slab[0]
    ads = ads_dict[adsorbate].copy()
    ads.set_constraint()
    ads.translate(ads_pos)
    adslab = ads + slab
    adslab.cell = slab.cell
    adslab.pbc = [True, True, True]
    adslab.set_constraint()
    adslab = utils.constrain_slab(adslab)
    size = adslab.positions.ptp(0)
    i = size.argmin()
    # rotation = ['-90y', '90x', ''][i]
    # rotation = ''
    size[i] = 0.0
    scale = min(25, 100 / max(1, size.max()))
    write_png(dump_dir + 'catalog/'+str(doc['_id']) + '-' + adsorbate + '.png',
              adslab, show_unit_cell=1, scale=scale)
    write_png(dump_dir + 'catalog/' + str(doc['_id']) + '-' + adsorbate + '-side.png',
              adslab, show_unit_cell=1, rotation='90y, 90z', scale=scale)


def writeAdsorptionImages(doc):
    atoms = make_atoms_from_doc(doc)
    adslab = atoms.copy()
    size = adslab.positions.ptp(0)
    i = size.argmin()
    # rotation = ['-90y', '90x', ''][i]
    # rotation = ''
    size[i] = 0.0
    scale = min(25, 100 / max(1, size.max()))
    write_png(dump_dir + 'adsorption/'+str(doc['_id']) + '.png', adslab, show_unit_cell=1, scale=scale)
    write_png(dump_dir + 'adsorption/'+str(doc['_id']) + '-side.png', adslab, show_unit_cell=1,
              rotation='90y, 90z', scale=scale)


def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:    # stops when iterator is depleted
        def chunk():          # construct generator for next chunk
            yield first       # yield element from for loop
            for more in islice(iterator, size-1):
                yield more    # yield more elements from the iterator
        yield chunk()         # in outer generator, yield next chunkdef chunks(iterable, size=10):
    iterator = iter(iterable)


def MakeImages(todo, collection, completed_images):
    pool = Pool(32)
    k = 0
    for chunk in chunks(todo, 1000):
        chunk = list(chunk)
        ids, adsorbates = zip(*chunk)
        uniques, inverse = np.unique(ids, return_inverse=True)
        docs = np.array([collection.find_one({"_id": ObjectId(id)}) for id in uniques])
        to_run = zip(docs[inverse], adsorbates)
        pool.map(writeImages, to_run)
        k += 1
        print('%d/%d' % (k*len(to_run), len(todo)))
        completed_images += chunk
        pickle.dump(completed_images, open(dump_dir+'completed_images.pkl', 'w'))
    pool.close()


def MakeImagesAdsorption(todo, collection, completed_images):
    pool = Pool(32)
    k = 0
    for chunk in chunks(todo, 1000):
        ids = list(chunk)
        uniques, inverse = np.unique(ids, return_inverse=True)
        docs = np.array([collection.find_one({"_id": ObjectId(id)}) for id in uniques])
        to_run = docs[inverse]
        pool.map(writeAdsorptionImages, to_run)
        k += 1
        print('%d/%d' % (k*len(to_run), len(todo)))
        completed_images += ids
        pickle.dump(completed_images, open(dump_dir+'completed_images.pkl', 'w'))
    pool.close()


def dump_images():
    if len(glob.glob(dump_dir+'completed_images.pkl')) > 0:
        completed_images = pickle.load(open(dump_dir+'completed_images.pkl'))
    else:
        completed_images = []

    for adsorbate in ['CO', 'H']:
        results = pickle.load(open(databall_template[adsorbate]))
        dft_ids = [a[0]['mongo_id'] for a in results[0]]
        todo = list(set(dft_ids) - set(completed_images))
        MakeImagesAdsorption(todo, get_mongo_collection('adsorption').db.adsorption, completed_images)

        dft_ids = [(a[0]['mongo_id'], adsorbate) for a in results[1]]
        todo = list(set(dft_ids) - set(completed_images))
        MakeImages(todo, get_mongo_collection('catalog_readonly').db.catalog, completed_images)
