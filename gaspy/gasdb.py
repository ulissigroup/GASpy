''' Various functions that may be used across GASpy and its submodules '''

import warnings
from itertools import islice
import json
import pickle
from multiprocessing import Pool
import glob
import numpy as np
import tqdm
from pymongo import MongoClient
from ase.io.png import write_png
from . import defaults, utils, vasp_functions
from .mongo import make_atoms_from_doc
from bson.objectid import ObjectId



class mongo_collection(object):
    def __init__(self,collection_name = 'adsorption'):
        mongo_info = utils.read_rc()['mongo_info'][collection_name]
        host = mongo_info['host']
        port = int(mongo_info['port'])
        database_name = mongo_info['database']
        user = mongo_info['user']
        password = mongo_info['password']

        # Access the client and authenticate
        self.client = MongoClient(host=host, port=port)
        database = getattr(self.client, database_name)
        database.authenticate(user, password)
        self.collection = getattr(database, collection_name)
        #return collection

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.client.close()

    def find(self, *args, **kwargs):
        return self.collection.find(*args, **kwargs)

    def count(self, *args, **kwargs):
        return self.collection.count(*args, **kwargs)

    def insert(self, *args, **kwargs):
        return self.collection.insert(*args, **kwargs)



def get_mongo_collection(collection_name='adsorption'):
    '''
    Get a mongo client for a certain collection. This function accesses the client
    using information in the `.gaspyrc.json` file.
    Arg:
        collection_name A string indicating the collection you want to access. Currently works
                        for values such as 'adsorption', 'catalog', 'atoms', 'catalog_readonly',
                        and 'surface_energy'.
    Returns:
        collection  An instance of `pymongo.MongoClient` that is connected and authenticated
    '''
    # Fetch the information we need to access the client.
    mongo_info = utils.read_rc()['mongo_info'][collection_name]
    host = mongo_info['host']
    port = int(mongo_info['port'])
    database_name = mongo_info['database']
    user = mongo_info['user']
    password = mongo_info['password']

    # Access the client and authenticate
    client = MongoClient(host=host, port=port)
    database = getattr(client, database_name)
    database.authenticate(user, password)
    collection = getattr(database, collection_name)
    return collection



def get_docs(collection_name='adsorption', fingerprints=None,
             adsorbates=None, calc_settings=None, vasp_settings=None,
             energy_min=None, energy_max=None, f_max=None, max_atoms=None,
             ads_move_max=None, bare_slab_move_max=None, slab_move_max=None):
    '''
    This function uses a mongo aggregator to find unique mongo docs and then returns them
    in two different forms:  a "raw" form (list of dicts) and a "parsed" form (dict of lists).
    Note that since we use a mongo aggregator, this function will return only unique mongo
    docs (as per the fingerprints supplied by the user); do not expect a mongo doc per
    matching database entry.

    Inputs:
        client              String indicating which collection in the database you want to pull from.
                            A good try would be 'adsorption', 'catalog', 'catalog_readonly',
                            'atoms', or 'surface_energy'.
        fingerprints        A dictionary of fingerprints and their locations in our
                            mongo documents. For example:
                                fingerprints = {'mpid': '$processed_data.calculation_info.mpid',
                                                'coordination': '$processed_data.fp_init.coordination'}
                            If `None`, then pull the default set of fingerprints.
                            If 'simulated', then pull the default set of fingerprints for
                            simulated documents.
        adsorbates          A list of adsorbates that you want to find matches for
        calc_settings       An optional argument that will only pull out data with these
                            calc settings (e.g., 'beef-vdw' or 'rpbe').
        vasp_settings       An optional argument that will only pull out data with these
                            vasp settings. Any assignments to `gga` here will overwrite
                            `calc_settings`.
        energy_min          The minimum adsorption energy to pull from the Local DB (eV)
        energy_max          The maximum adsorption energy to pull from the Local DB (eV)
        f_max               The upper limit on the maximum force on an atom in the system
        max_atoms           The maximum number of atoms in the system that you want to pull
        ads_move_max        The maximum distance that an adsorbate atom may move (angstrom)
        bare_slab_move_max  The maxmimum distance that a slab atom may move when it is relaxed
                            without an adsorbate (angstrom)
        slab_move_max       The maximum distance that a slab atom may move (angstrom)
    Output:
        docs    Mongo docs; a list of dictionaries for each database entry
    '''
    if not fingerprints:
        fingerprints = defaults.fingerprints()
    if fingerprints == 'simulated':
        fingerprints = defaults.fingerprints(simulated=True)

    # Put the "fingerprinting" into a `group` dictionary, which we will
    # use to pull out data from the mongo database. Also, initialize
    # a `match` dictionary, which we will use to filter results.
    group = {'$group': {'_id': fingerprints}}
    match = {'$match': {}}

    # Create `match` filters to search by. Use if/then statements to create the filters
    # only if the user specifies them.
    if calc_settings:
        xc_settings = defaults.exchange_correlational_settings()
        gga_method = xc_settings[calc_settings]['gga']
        match['$match']['processed_data.vasp_settings.gga'] = gga_method
    if vasp_settings:
        for key, value in vasp_settings.items():
            match['$match']['processed_data.vasp_settings.%s' % key] = value
        # Alert the user that they tried to specify the gga twice.
        if ('gga' in vasp_settings and calc_settings):
            warnings.warn('User specified both calc_settings and vasp_settings.gga. GASpy will default to the given vasp_settings.gga', SyntaxWarning)
    if adsorbates:
        match['$match']['processed_data.calculation_info.adsorbate_names'] = adsorbates
    # Multi-conditional for the energy for the different ways a user can define
    # energy constraints
    if (energy_max and energy_min):
        match['$match']['results.energy'] = {'$gt': energy_min, '$lt': energy_max}
    elif (energy_max and not energy_min):
        match['$match']['results.energy'] = {'$lt': energy_max}
    elif (not energy_max and energy_min):
        match['$match']['results.energy'] = {'$gt': energy_min}
    # We do a doubly-nested element match because `results.forces` is a doubly-nested
    # list of forces (1st layer is atoms, 2nd layer is cartesian directions, final
    # layer is the forces on that atom in that direction).
    if f_max:
        match['$match']['results.forces'] = {'$not': {'$elemMatch': {'$elemMatch': {'$gt': f_max}}}}
    if max_atoms:
        match['$match']['atoms.natoms'] = {'$lt': max_atoms}
    if ads_move_max:
        match['$match']['processed_data.movement_data.max_adsorbate_movement'] = \
            {'$lt': ads_move_max}
    if bare_slab_move_max:
        match['$match']['processed_data.movement_data.max_bare_slab_movement'] = \
            {'$lt': bare_slab_move_max}
    if slab_move_max:
        match['$match']['processed_data.movement_data.max_surface_movement'] = \
            {'$lt': slab_move_max}

    # Compile the pipeline; add matches only if any matches are specified
    if match['$match']:
        pipeline = [match, group]
    else:
        pipeline = [group]
    # Get the particular collection from the mongo client's database.
    # We we're pulling the catalog, then get the read only version of the database
    # so that we pull it even faster.
    if collection_name == 'catalog':
        collection = get_mongo_collection('catalog_readonly')
    else:
        collection = get_mongo_collection(collection_name)

    # Create the cursor. We set allowDiskUse=True to allow mongo to write to
    # temporary files, which it needs to do for large databases. We also
    # set useCursor=True so that `aggregate` returns a cursor object
    # (otherwise we run into memory issues).
    print('Starting to pull documents...')
    cursor = collection.aggregate(pipeline, allowDiskUse=True, useCursor=True)
    # Use the cursor to pull all of the information we want out of the database.
    docs = [doc['_id'] for doc in tqdm.tqdm(cursor)]
    if not docs:
        warnings.warn('We did not find any matching documents', RuntimeWarning)

    # If any document is missing fingerprints or has any empty keys, then delete it.
    for doc in docs:
        if set(fingerprints.keys()).issubset(doc.keys()) and all(doc.values()):
            pass
        else:
            del doc

    return docs


def hash_docs(docs, ignore_keys=None):
    '''
    This function helps convert the important characteristics of our systems into hashes
    so that we may sort through them more quickly. This is important to do when trying to
    compare entries in our two databases; it helps speed things up.

    Input:
        docs            Mongo docs (list of dictionaries) that have been created using the
                        gaspy.gasdb.get_docs function. Note that this is the unparsed version
                        of mongo documents.
        ignore_keys     A list of strings indicating the keys that you want to ignore
                        when hashing.
    Output:
        systems     An ordered dictionary whose keys are hashes of the each doc in
                    `docs` and whose values are empty. This dictionary is intended
                    to be parsed alongside another `docs` object, which is why
                    it's ordered.
    '''
    # Python doesn't do well with mutable default arguments. We define defaults like this
    # to address that issue.
    if not ignore_keys:
        ignore_keys = []
    # Add the mongo ID to the list of ignored keys, because that'll always yield a different
    # hash. Then turn it into a set to speed up searching.
    ignore_keys.append('mongo_id')
    ignore_keys = set(ignore_keys)

    def hash_doc(doc):
        ''' Make a function that hashes one document so that we can monitor progress '''
        # `system` will be one long string of the fingerprints
        system = ''
        for key in sorted(doc.keys()):
            # Skip this key if we want to ignore it
            if key not in ignore_keys:
                # Round floats to increase chances of matching
                value = doc[key]
                if isinstance(value, float):
                    value = round(value, 2)

                # Note that we turn the values into strings explicitly, because some
                # fingerprint features may not be strings (e.g., list of miller indices).
                system += str(key + '=' + str(value) + '; ')
        return hash(system)

    # Hash with a progress bar
    systems = [hash_doc(doc) for doc in tqdm.tqdm(docs)]
    return systems


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
    print('Hashing adsorbates...')
    ads_hashes = hash_docs(ads_docs, ignore_keys=['energy', 'formula', 'shift', 'top'])
    print('Hashing catalog...')
    cat_hashes = hash_docs(cat_docs, ignore_keys=['formula', 'shift', 'top'])
    unsim_hashes = set(cat_hashes)-set(ads_hashes)

    # Perform the filtering while simultaneously populating the `docs` output.
    print('Filtering by the hashes...')
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
