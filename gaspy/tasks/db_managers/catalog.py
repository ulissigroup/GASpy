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

import pickle
import luigi
import multiprocess
from pymatgen.ext.matproj import MPRester
from ..core import (run_tasks,
                    get_task_output,
                    save_task_output,
                    make_task_output_object,
                    evaluate_luigi_task)
from ..atoms_generators import GenerateAllSitesFromBulk
from ... import defaults
from ...utils import read_rc, unfreeze_dict
from ...mongo import make_atoms_from_doc
from ...gasdb import get_mongo_collection
from ...atoms_operators import fingerprint_adslab

BULK_SETTINGS = defaults.bulk_settings()
SLAB_SETTINGS = defaults.slab_settings()
ADSLAB_SETTINGS = defaults.adslab_settings()


def update_catalog_collection(elements, max_miller, n_processes=1):
    '''
    This function will add enumerate and add adsorption sites to our `catalog`
    Mongo collection.

    Args:
        elements            A list of strings indicating the elements you
                            are looking for, e.g., ['Cu', 'Al']
        max_miller          An integer indicating the maximum Miller index
                            to be enumerated
        n_processes         An integer indicating how many threads you want to
                            use when running the tasks. If you do not expect
                            many updates, stick to the default of 1, or go up
                            to 4. If you are re-creating your collection from
                            scratch, you may want to want to increase this
                            argument as high as you can.
    '''
    # Figure out the MPIDs we need to enumerate
    get_mpid_task = _GetMpids(elements=elements)
    run_tasks([get_mpid_task])
    mpids = get_task_output(get_mpid_task)

    # For each MPID, enumerate all the sites and then add them to our `catalog`
    # Mongo collection. Do this in parallel because it can be.
    if n_processes > 1:
        with multiprocess.Pool(n_processes) as pool:
            list(pool.imap(func=lambda mpid: __run_insert_to_catalog_task(mpid, max_miller),
                           iterable=mpids, chunksize=100))
    else:
        for mpid in mpids:
            __run_insert_to_catalog_task(mpid, max_miller)


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

        # Ask Materials Project for any matches
        with MPRester(read_rc('matproj_api_key')) as rester:
            results = rester.query({'elements': {'$nin': list(elements_restricted)},
                                    'e_above_hull': {'$lt': 0.1},
                                    'formation_energy_per_atom': {'$lte': 0.}},
                                   ['task_id'])

        # Save
        mpids = set(result['task_id'] for result in results)
        save_task_output(self, mpids)

    def output(self):
        return make_task_output_object(self)


def __run_insert_to_catalog_task(mpid, max_miller):
    '''
    Very light wrapper to instantiate a `_InsertSitesToCatalog` task and then
    run it manually.

    Args:
        mpid                A string indicating the Materials Project ID of the
                            bulk you want to enumerate sites from
        max_miller          An integer indicating the maximum Miller index to
                            be enumerated
    '''
    task = _InsertSitesToCatalog(mpid, max_miller)
    try:
        evaluate_luigi_task(task)

    # We need bulk calculations to enumerate our catalog. If these calculations
    # aren't done, then we won't find the Luigi task pickles. If this happens,
    # then we should just move on to the next thing.
    except FileNotFoundError:
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
        bulk_vasp_settings      A dictionary containing the VASP settings of
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
    bulk_vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def requires(self):
        return GenerateAllSitesFromBulk(mpid=self.mpid,
                                        max_miller=self.max_miller,
                                        min_xy=self.min_xy,
                                        slab_generator_settings=self.slab_generator_settings,
                                        get_slab_settings=self.get_slab_settings,
                                        bulk_vasp_settings=self.bulk_vasp_settings)

    def run(self, _testing=False):
        '''
        Don't use the `_testing` argument unless you're unit testing
        '''
        with open(self.input().path, 'rb') as file_handle:
            site_docs = pickle.load(file_handle)
        with get_mongo_collection('catalog') as collection:

            # Try to find each adsorption site in our catalog
            incumbent_docs = []
            inserted_docs = []
            for site_doc in site_docs:
                query = {'mpid': self.mpid,
                         'miller': site_doc['miller'],
                         'min_xy': self.min_xy,
                         'slab_generator_settings': unfreeze_dict(self.slab_generator_settings),
                         'get_slab_settings': unfreeze_dict(self.get_slab_settings),
                         'bulk_vasp_settings': unfreeze_dict(self.bulk_vasp_settings),
                         'shift': site_doc['shift'],
                         'top': site_doc['top'],
                         'slab_repeat': site_doc['slab_repeat'],
                         'adsorption_site': tuple(site_doc['adsorption_site'])}
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
                    doc['bulk_vasp_settings'] = unfreeze_dict(self.bulk_vasp_settings)
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
                print('Just created %i new entries in the cataog collection'
                      % len(inserted_docs))
        save_task_output(self, incumbent_docs + inserted_docs)

    def output(self):
        return make_task_output_object(self)
