'''
This module houses the function and helper tasks to enumerate and populate our
`catalog` Mongo collection of adsorption sites.

WARNING:  You may want to run these bash commands prior to running this
`udpate_catalog_colection`:
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export OMP_NUM_THREADS=1
They will stop numpy/scipy from trying to parallelize over all the cores, which
will slow us down since we are already parallelizing via multiprocess.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

from datetime import datetime
import luigi
import multiprocess
from pymatgen.ext.matproj import MPRester
from ..core import (schedule_tasks,
                    get_task_output,
                    save_task_output,
                    make_task_output_object)
from ..atoms_generators import GenerateAllSitesFromBulk
from ..calculation_finders import FindBulk
from ... import defaults
from ...utils import read_rc, unfreeze_dict
from ...mongo import make_atoms_from_doc
from ...gasdb import get_mongo_collection
from ...atoms_operators import fingerprint_adslab

DFT_CALCULATOR = defaults.DFT_CALCULATOR
BULK_SETTINGS = defaults.bulk_settings()
SLAB_SETTINGS = defaults.slab_settings()
ADSLAB_SETTINGS = defaults.adslab_settings()


def update_catalog_collection(elements, max_miller,
                              bulk_dft_settings=BULK_SETTINGS[DFT_CALCULATOR],
                              n_processes=1, mp_query=None):
    '''
    This function will add enumerate and add adsorption sites to our `catalog`
    Mongo collection.

    Args:
        elements            A list of strings indicating the elements you are
                            looking for, e.g., ['Cu', 'Al']
        max_miller          An integer indicating the maximum Miller index to
                            be enumerated
        bulk_dft_settings   A dictionary containing the DFT settings you want to
                            use for the bulk calculations. Defaults to using
                            `gaspy.defaults.bulk_dft_settings()['vasp']`.
        n_processes         An integer indicating how many threads you want to
                            use when running the tasks. If you do not expect
                            many updates, stick to the default of 1, or go up
                            to 4. If you are re-creating your collection from
                            scratch, you may want to want to increase this
                            argument as high as you can.
        mp_query            We get our bulks from The Materials Project. This
                            dictionary argument is used as a Mongo query to The
                            Materials Project Database. If `None`, then it will
                            automatically filter out bulks whose energies above the
                            hull are greater than 0.1 eV and whose formation energy
                            per atom are above 0 eV.
    '''
    # Python doesn't like mutable arguments
    if mp_query is None:
        mp_query = {'e_above_hull': {'$lt': 0.1},
                    'formation_energy_per_atom': {'$lte': 0.}}

    # Figure out the MPIDs we need to enumerate
    get_mpid_task = _GetMpids(elements=elements, mp_query=mp_query)
    schedule_tasks([get_mpid_task])
    mpids = get_task_output(get_mpid_task)

    # For each MPID, enumerate all the sites and then add them to our `catalog`
    # Mongo collection. Do this in parallel because it can be.
    if n_processes > 1:
        with multiprocess.Pool(n_processes) as pool:
            list(pool.imap(func=lambda mpid: __run_insert_to_catalog_task(mpid, max_miller),
                           iterable=mpids, chunksize=20))
    else:
        for mpid in mpids:
            __run_insert_to_catalog_task(mpid, max_miller, bulk_dft_settings)


class _GetMpids(luigi.Task):
    '''
    This task will get all the Materials Project ID numbers of all bulk
    materials that contain elements from a specified set of elements.
    We also make sure that we get only materials whose formation energy
    per atom is >= 0. eV, and whose energy above the hull is less than
    0.1 eV.

    Args:
        elements    A list of strings indicating the elements you are
                    looking for, e.g., ['Cu', 'Al']
    Return:
        mpids   A set of strings indicating the MPIDs that we found
                that meet the criteria of the arguments,
                e.g., `{'mp-2'}`
    '''
    elements = luigi.ListParameter()
    mp_query = luigi.DictParameter()

    def run(self):
        '''
        Query the Materials Project database
        '''
        # Define the elements that we don't want to see
        all_elements = set(['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au',
                            'B', 'Ba', 'Be', 'Bh', 'Bi', 'Bk', 'Br', 'C', 'Ca',
                            'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Cn', 'Co', 'Cr',
                            'Cs', 'Cu', 'Db', 'Ds', 'Dy', 'Er', 'Es', 'Eu',
                            'F', 'Fe', 'Fl', 'Fm', 'Fr', 'Ga', 'Gd', 'Ge', 'H',
                            'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K',
                            'Kr', 'La', 'Li', 'Lr', 'Lu', 'Lv', 'Mc', 'Md',
                            'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na', 'Nb', 'Nd',
                            'Ne', 'Nh', 'Ni', 'No', 'Np', 'O', 'Og', 'Os', 'P',
                            'Pa', 'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu',
                            'Ra', 'Rb', 'Re', 'Rf', 'Rg', 'Rh', 'Rn', 'Ru',
                            'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn',
                            'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl',
                            'Tm', 'Ts', 'U', 'V', 'W', 'Xe', 'Y', 'Yb', 'Zn',
                            'Zr'])
        elements_allowed = set(self.elements)
        elements_restricted = all_elements - elements_allowed

        # Instantiate the Mongo query to the Materials Project database, and
        # then attach defaults
        query = {'elements': {'$nin': list(elements_restricted)}}
        for key, value in unfreeze_dict(self.mp_query).items():
            query[key] = value

        # Ask Materials Project for any matches
        with MPRester(read_rc('matproj_api_key')) as rester:
            results = rester.query(query, ['task_id'])

        # Save
        mpids = {result['task_id'] for result in results}
        save_task_output(self, mpids)

    def output(self):
        return make_task_output_object(self)


def __run_insert_to_catalog_task(mpid, max_miller, bulk_dft_settings):
    '''
    Very light wrapper to instantiate a `_InsertSitesToCatalog` task and then
    run it manually.

    Args:
        mpid                A string indicating the Materials Project ID of the
                            bulk you want to enumerate sites from
        max_miller          An integer indicating the maximum Miller index to
                            be enumerated
    '''
    task = _InsertSitesToCatalog(mpid=mpid, max_miller=max_miller,
                                 bulk_dft_settings=bulk_dft_settings)

    # We need bulk calculations to enumerate our catalog. If these calculations
    # aren't done, then we won't find the Luigi task pickles. If this happens,
    # then we should just move on to the next thing.
    try:
        schedule_tasks([task], local_scheduler=True)
    except RuntimeError:
        pass


class _InsertSitesToCatalog(luigi.Task):
    '''
    This task will enumerate a set of adsorption sites, and then it will find
    these sites in the `catalog` Mongo collection. If any site is not in the
    catalog, then this task will insert it to the catalog.

    Luigi normally likes unit tasks (i.e., a task that adds only one site to
    the catalog, instead of a bunch), but this would end up spawing millions
    of tasks, which Luigi is not fast at handling. So we instead make this
    task, so Luigi only has to handle O(X0,000) tasks.

    Args:
        mpid                    A string indicating the Materials Project ID of
                                the bulk you want to enumerate sites from
        max_miller              An integer indicating the maximum Miller index
                                to be enumerated
        min_xy                  A float indicating the minimum width (in both
                                the x and y directions) of the slab (Angstroms)
                                before we enumerate adsorption sites on it.
        slab_generator_settings We use pymatgen's `SlabGenerator` class to
                                enumerate surfaces. You can feed the arguments
                                for that class here as a dictionary.
        get_slab_settings       We use the `get_slabs` method of pymatgen's
                                `SlabGenerator` class. You can feed the
                                arguments for the `get_slabs` method here
                                as a dictionary.
        bulk_dft_settings      A dictionary containing the VASP settings of
                                the relaxed bulk to enumerate slabs from
    Returns:
        docs    A list of all of the Mongo documents (i.e., dictionaries)
                from the `catalog` collection that match the arguments you
                fed to this task + the documents that we just added
    '''
    mpid = luigi.Parameter()
    max_miller = luigi.IntParameter()
    min_xy = luigi.FloatParameter(ADSLAB_SETTINGS['min_xy'])
    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
    bulk_dft_settings = luigi.DictParameter(BULK_SETTINGS[DFT_CALCULATOR])

    def requires(self):
        bulk_finder = FindBulk(mpid=self.mpid,
                               dft_settings=self.bulk_dft_settings)
        site_gen = GenerateAllSitesFromBulk(mpid=self.mpid,
                                            max_miller=self.max_miller,
                                            min_xy=self.min_xy,
                                            slab_generator_settings=self.slab_generator_settings,
                                            get_slab_settings=self.get_slab_settings,
                                            bulk_dft_settings=self.bulk_dft_settings)
        return {'bulk_finder': bulk_finder,
                'site_generator': site_gen}

    def run(self, _testing=False):
        '''
        Don't use the `_testing` argument unless you're unit testing
        '''
        reqs = self.requires()
        bulk_finder = reqs['bulk_finder']
        site_gen = reqs['site_generator']

        # Calculate the k-points, if needed
        bulk_dft_settings = unfreeze_dict(self.bulk_dft_settings)
        if bulk_dft_settings['kpts'] == 'bulk':
            bulk_dft_settings['kpts'] = bulk_finder.calculate_bulk_k_points()

        # Grab all of the sites we want in the catalog
        site_docs = get_task_output(site_gen)

        # Try to find each adsorption site in our catalog
        with get_mongo_collection('catalog') as collection:
            incumbent_docs = []
            inserted_docs = []
            for site_doc in site_docs:
                query = {'mpid': self.mpid,
                         'miller': site_doc['miller'],
                         'min_xy': self.min_xy,
                         'slab_generator_settings': unfreeze_dict(self.slab_generator_settings),
                         'get_slab_settings': unfreeze_dict(self.get_slab_settings),
                         'bulk_dft_settings': bulk_dft_settings,
                         'shift': {'$gt': site_doc['shift'] - 0.01,
                                   '$lt': site_doc['shift'] + 0.01},
                         'top': site_doc['top'],
                         'slab_repeat': site_doc['slab_repeat'],
                         'adsorption_site.0': {'$gt': site_doc['adsorption_site'][0] - 0.01,
                                               '$lt': site_doc['adsorption_site'][0] + 0.01},
                         'adsorption_site.1': {'$gt': site_doc['adsorption_site'][1] - 0.01,
                                               '$lt': site_doc['adsorption_site'][1] + 0.01},
                         'adsorption_site.2': {'$gt': site_doc['adsorption_site'][2] - 0.01,
                                               '$lt': site_doc['adsorption_site'][2] + 0.01}}
                docs_in_catalog = list(collection.find(query))

                # If a site is in the catalog, then we don't need to add it
                if len(docs_in_catalog) >= 1:
                    incumbent_docs.append(docs_in_catalog[0])

                # If a site is not in the catalog, then create the document
                elif len(docs_in_catalog) == 0:
                    doc = site_doc.copy()
                    doc['mpid'] = self.mpid
                    doc['min_xy'] = self.min_xy
                    doc['slab_generator_settings'] = unfreeze_dict(self.slab_generator_settings)
                    doc['get_slab_settings'] = unfreeze_dict(self.get_slab_settings)
                    doc['bulk_dft_settings'] = bulk_dft_settings
                    doc['adsorption_site'] = tuple(doc['adsorption_site'])
                    doc['fwids'] = site_doc['fwids']
                    # Add fingerprint information to the document
                    atoms = make_atoms_from_doc(doc)
                    fingerprint = fingerprint_adslab(atoms)
                    for key, value in fingerprint.items():
                        doc[key] = value

                    # It's faster to write in bulk instead of one-at-a-time, so
                    # save the document to one list that we'll write to
                    inserted_docs.append(doc)

            # Add the documents to the catalog
            if not _testing and len(inserted_docs) > 0:
                collection.insert_many(inserted_docs)
                print('[%s] Created %i new entries in the catalog collection'
                      % (datetime.now(), len(inserted_docs)))
        save_task_output(self, incumbent_docs + inserted_docs)

    def output(self):
        return make_task_output_object(self)
