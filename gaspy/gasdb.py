''' Various functions that may be used across GASpy and its submodules '''

import pdb  # noqa: F401
import warnings
from itertools import islice
import cPickle as pickle
from multiprocessing import Pool
import glob
import numpy as np
import tqdm
from ase.io.png import write_png
from vasp.mongo import MongoDatabase, mongo_doc_atoms
from . import defaults, utils


def get_catalog_client():
    ''' This is the information for the `catalog` collection in our vasp.mongo database '''
    # Open the appropriate information from the RC file
    kwargs = utils.read_rc()['catalog_client']
    # Turn the port number into an integer
    for key, value in kwargs.iteritems():
        if key == 'port':
            kwargs[key] = int(value)
    return MongoDatabase(**kwargs)


def get_catalog_client_readonly():
    '''
    This is the information for a read-only version of our `catalog` collection
    in our vasp.mongo database. This is useful for pulling information more
    quickly than normal while acknowledging that we will not be changing it.
    '''
    # Open the appropriate information from the RC file
    kwargs = utils.read_rc()['catalog_client_readonly']
    # Turn the port number into an integer
    for key, value in kwargs.iteritems():
        if key == 'port':
            kwargs[key] = int(value)
    return MongoDatabase(**kwargs)


def get_atoms_client():
    ''' This is the information for the `atoms` collection in our vasp.mongo database '''
    # Open the appropriate information from the RC file
    kwargs = utils.read_rc()['atoms_client']
    # Turn the port number into an integer
    for key, value in kwargs.iteritems():
        if key == 'port':
            kwargs[key] = int(value)
    return MongoDatabase(**kwargs)


def get_adsorption_client():
    ''' This is the information for the `adsorption` collection in our vasp.mongo database '''
    # Open the appropriate information from the RC file
    kwargs = utils.read_rc()['adsorption_client']
    # Turn the port number into an integer
    for key, value in kwargs.iteritems():
        if key == 'port':
            kwargs[key] = int(value)
    return MongoDatabase(**kwargs)


def get_docs(client=get_adsorption_client(), collection_name='adsorption', fingerprints=None,
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
        client              Mongo client object
        collection_name     The collection name within the client that you want to look at
        fingerprints        A dictionary of fingerprints and their locations in our
                            mongo documents. For example:
                                fingerprints = {'mpid': '$processed_data.calculation_info.mpid',
                                                'coordination': '$processed_data.fp_init.coordination'}
                            If `None`, then pull the default set of fingerprints.
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

    # Put the "fingerprinting" into a `group` dictionary, which we will
    # use to pull out data from the mongo database. Also, initialize
    # a `match` dictionary, which we will use to filter results.
    group = {'$group': {'_id': fingerprints}}
    match = {'$match': {}}

    # Create `match` filters to search by. Use if/then statements to create the filters
    # only if the user specifies them.
    if not calc_settings:
        pass
    elif calc_settings == 'rpbe':
        match['$match']['processed_data.vasp_settings.gga'] = 'RP'
    elif calc_settings == 'beef-vdw':
        match['$match']['processed_data.vasp_settings.gga'] = 'BF'
    else:
        raise Exception('Unknown calc_settings')
    if vasp_settings:
        for key, value in vasp_settings.iteritems():
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
        collection = getattr(get_catalog_client_readonly().db, collection_name)
    else:
        collection = getattr(client.db, collection_name)

    # Create the cursor. We set allowDiskUse=True to allow mongo to write to
    # temporary files, which it needs to do for large databases. We also
    # set useCursor=True so that `aggregate` returns a cursor object
    # (otherwise we run into memory issues).
    print('Starting to pull documents...')
    cursor = collection.aggregate(pipeline, allowDiskUse=True, useCursor=True)
    # Use the cursor to pull all of the information we want out of the database, and
    # then parse it. Note that we forgo parsing if we did not find any documents.
    docs = [doc['_id'] for doc in tqdm.tqdm(cursor)]
    if not docs:
        warnings.warn('We did not find any matching documents', RuntimeWarning)

    return docs


def hash_docs(docs, ignore_ads=False, ignore_energy=True):
    '''
    This function helps convert the important characteristics of our systems into hashes
    so that we may sort through them more quickly. This is important to do when trying to
    compare entries in our two databases; it helps speed things up.

    Input:
        docs            Mongo docs (list of dictionaries) that have been created using the
                        gaspy.gasdb.get_docs function. Note that this is the unparsed version
                        of mongo documents.
        ignore_ads      A boolean that decides whether or not we hash the adsorbate.
                        This is useful mainly for the "matching_ads" function.
        ignore_energy   A boolean that decides whether or not we hash the energy.
    Output:
        systems     An ordered dictionary whose keys are hashes of the each doc in
                    `docs` and whose values are empty. This dictionary is intended
                    to be parsed alongside another `docs` object, which is why
                    it's ordered.
    '''
    def hash_doc(doc):
        ''' Make a function that hashes one document so that we can monitor progress '''
        # `system` will be one long string of the fingerprints
        system = ''
        for key in sorted(doc.keys()):
            # Ignore mongo ID, because that'll always cause things to hash differently
            if key != 'mongo_id':
                # Ignore adsorbates if the user wants to, as per the argument.
                # Do the same for energy.
                if not (ignore_ads and key == 'adsorbate_names'):
                    if not (ignore_energy and key == 'energy'):
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
    The same as `get_docs`, but with already-simulated entries filtered out

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
    print('Hashing adsorbates...')
    ads_hashes = hash_docs(ads_docs)
    print('Hashing catalog...')
    cat_hashes = hash_docs(cat_docs)
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
    with get_adsorption_client() as ads_client:
        ads_docs = get_docs(ads_client, 'adsorption',
                            calc_settings=calc_settings,
                            vasp_settings=vasp_settings,
                            fingerprints=fingerprints,
                            adsorbates=adsorbates)
    with get_catalog_client() as cat_client:
        cat_docs = get_docs(cat_client, 'catalog', fingerprints=fingerprints, max_atoms=max_atoms)

    # Use the `split_catalog` function to find the indices in `cat_docs` that correspond
    # with item that we have not yet simulated. Then use that list to build the documents.
    _, unsim_inds = split_catalog(ads_docs, cat_docs)
    docs = [cat_docs[i] for i in unsim_inds]
    return docs


# TODO:  Commend and clean up everything below here
ads_dict = defaults.adsorbates_dict()
del ads_dict['']
del ads_dict['U']
ads_to_run = ads_dict.keys()
ads_to_run = ['CO', 'H']
dump_dir = '/global/cscratch1/sd/zulissi/GASpy_DB/images/'
databall_template = {'CO': '/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/CO2RR_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl',
                     'H': '/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/HER_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl'}


def writeImages(input):
    doc, adsorbate = input
    atoms = mongo_doc_atoms(doc)
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
    atoms = mongo_doc_atoms(doc)
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
        docs = np.array([collection.find_one({"_id": id}) for id in uniques])
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
        docs = np.array([collection.find_one({"_id": id}) for id in uniques])
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
        MakeImagesAdsorption(todo, get_adsorption_client().db.adsorption, completed_images)

        dft_ids = [(a[0]['mongo_id'], adsorbate) for a in results[1]]
        todo = list(set(dft_ids) - set(completed_images))
        MakeImages(todo, get_catalog_client().db.catalog, completed_images)
