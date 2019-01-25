'''
This module houses various tasks to manage our Mongo collections/databases.

'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import pickle
import datetime
import uuid
import subprocess
import multiprocess
from tqdm import tqdm
import luigi
import ase
import ase.io
from ase.calculators.vasp import Vasp2
from pymatgen.ext.matproj import MPRester
from .core import (run_tasks,
                   get_task_output,
                   save_task_output,
                   make_task_output_object)
from .atoms_generators import GenerateAllSitesFromBulk
from .. import defaults
from ..utils import read_rc, unfreeze_dict
from ..mongo import make_atoms_from_doc, make_doc_from_atoms
from ..gasdb import get_mongo_collection
from ..fireworks_helper_scripts import get_launchpad, get_atoms_from_fw
from ..atoms_operators import fingerprint_adslab

BULK_SETTINGS = defaults.bulk_settings()
SLAB_SETTINGS = defaults.slab_settings()
ADSLAB_SETTINGS = defaults.adslab_settings()


def update_atoms_collection(n_processes=1, progress_bar=False):
    '''
    This function will dump all of the completed FireWorks into our `atoms` Mongo
    collection. It will not dump anything that is already there.

    You may notice some methods with the term "patched" in them. If you are
    using GASpy and not in the Ulissigroup, then you can probably skip all the
    patching by not calling the `__patch_old_document` method at all.

    Args:
        n_processes     An integer indicating how many threads you want to use
                        when converting FireWorks into Mongo documents. If you
                        do not expect many updates, stick to the default of 1.
                        If you are re-creating your collection from a full
                        FireWorks database, you may want to increase this
                        argument.
        progress_bar    A Booliean indicating whether or not you want to show
                        a progress bar when converting FireWorks into Mongo
                        documents. Useful when re-populating an empty
                        `atoms` collection with a big FireWorks database,
                        but not useful for small, periodic updates.
    '''
    fwids_missing = _find_fwids_missing_from_atoms_collection()

    # Multithread in case we have a lot of things to update
    if n_processes > 1:
        with multiprocess.Pool(n_processes) as pool:
            doc_generator = pool.imap(func=_make_doc_from_fwid,
                                      iterable=fwids_missing,
                                      chunksize=100)
            # Show a progress bar only if you asked for it
            if progress_bar is True:
                docs = list(tqdm(doc_generator, total=len(fwids_missing)))
            else:
                docs = list(doc_generator)

    # Keep it simple if we're not multithreading
    else:
        if progress_bar is True:
            doc_generator = (_make_doc_from_fwid(fwid) for fwid in fwids_missing)
            docs = list(tqdm(doc_generator, total=len(fwids_missing)))
        else:
            docs = [_make_doc_from_fwid(fwid) for fwid in fwids_missing]

    with get_mongo_collection('atoms') as collection:
        collection.insert_many(docs)


def _find_fwids_missing_from_atoms_collection():
    '''
    This method will get the FireWork IDs that are marked as 'COMPLETED' in
    our LaunchPad, but have not yet been added to our `atoms` Mongo
    collection.

    Returns:
        fwids_missing   A set of integers containing the FireWork IDs of
                        the calculations we have not yet added to our
                        `atoms` Mongo collection.
    '''
    with get_mongo_collection('atoms') as collection:
        docs = list(collection.find({}, {'fwid': 'fwid', '_id': 0}))
    fwids_in_atoms = set(doc['fwid'] for doc in docs)

    lpad = get_launchpad()
    fwids_completed = set(lpad.get_fw_ids({'state': 'COMPLETED'}))

    fwids_missing = fwids_completed - fwids_in_atoms
    return fwids_missing


def _make_doc_from_fwid(fwid):
    '''
    For each fireworks object, turn the results into a mongo doc so that we
    can dump the mongo doc into the Aux DB.

    Args:
        fwid    An integer indicating the FWID you want to turn into a
                document
    Returns:
        doc     A dictionary that contains various information about
                a calculation. Intended to be inserted into Mongo.
    '''
    # Get the `ase.Atoms` objects of the initial and final images
    lpad = get_launchpad()
    fw = lpad.get_fw_by_id(fwid)
    starting_atoms = get_atoms_from_fw(fw, index=0)
    atoms = get_atoms_from_fw(fw, index=-1)

    # Turn the atoms objects into a document and then add additional
    # information
    doc = make_doc_from_atoms(atoms)
    doc['initial_configuration'] = make_doc_from_atoms(starting_atoms)
    doc['fwname'] = fw.name
    doc['fwid'] = fwid
    doc['directory'] = fw.launches[-1].launch_dir

    # Fix some of our old FireWorks
    doc = __patch_old_document(doc, atoms, fw)
    return doc


def __patch_old_document(doc, atoms, fw):
    '''
    We've tried, tested, and failed a lot of times when making FireWorks.
    This method will parse some of our old iterations of FireWorks to meet
    the appropriate standards.

    Arg:
        doc     Dictionary that you'll be adding as a Mongo document to the
                `atoms` collection.
        atoms   Instance of the final `ase.Atoms` image of the Firework you
                are patching. This should NOT be the initial image.
        fw      Instance of the `fireworks.Firework` that you want to patch
    Return:
        patched_doc     Patched version of the `doc` argument.
    '''
    patched_doc = doc.copy()

    # Fix the energies in the atoms object
    patched_atoms = __patch_atoms_from_old_vasp(atoms, fw)
    for key, value in make_doc_from_atoms(patched_atoms).items():
        patched_doc[key] = value

    # Guess some VASP setttings we never recorded
    patched_doc['fwname']['vasp_settings'] = __get_patched_vasp_settings(fw)

    # Some of our old FireWorks had string-formatted Miller indices. Let's
    # fix those.
    if 'miller' in fw.name:
        patched_doc['fwname']['miller'] = __get_patched_miller(fw.name['miller'])

    return patched_doc


def __patch_atoms_from_old_vasp(atoms, fw):
    '''
    The VASP calculator, when used with ASE optimization, was incorrectly
    recording the internal forces in atoms objects with the stored forces
    including constraints. If such incompatible constraints exist and the
    calculations occured before the switch to the Vasp2 calculator, we
    should get the correct (VASP) forces from a backup of the directory
    which includes the INCAR, ase-sort.dat, etc files

    Args:
        atoms   Instance of an `ase.Atoms` object
        fw      Instance of an `fireworks.Firework` object created by the
                `gaspy.fireworks_helper_scripts.make_firework` function
    Returns:
        patched_atoms    Foo
    '''
    # If the FireWork is old and has constraints more complicated than
    # 'FixAtoms', then replace the atoms with another copy that has the
    # correct VASP forces
    if any(constraint.todict()['name'] != 'FixAtoms' for constraint in atoms.constraints):
        if fw.created_on < datetime(2018, 12, 1):
            launch_id = fw.launches[-1].launch_id
            patched_atoms = __get_final_atoms_object_with_vasp_forces(launch_id)
            return patched_atoms

    # If the FireWork is new or simple, then no patching is necessary
    return atoms


def __get_final_atoms_object_with_vasp_forces(launch_id):
    '''
    This function will return an ase.Atoms object from a particular
    FireWorks launch ID. It will also make sure that the ase.Atoms object
    will have VASP-calculated forces attached to it.

    Arg:
        launch_id   An integer representing the FireWorks launch ID of the
                    atoms object you want to get
    Returns:
        atoms   ase.Atoms object with the VASP-calculated forces
    '''
    # We will be opening a temporary directory where we will unzip the
    # FireWorks launch directory
    fw_launch_file = (read_rc('fireworks_info.backup_directory') +
                      '/%d.tar.gz' % launch_id)
    temp_loc = __dump_file_to_tmp(fw_launch_file)

    # Load the atoms object and then load the correct (DFT) forces from the
    # OUTCAR/etc info
    try:
        atoms = ase.io.read('%s/slab_relaxed.traj' % temp_loc)
        vasp2 = Vasp2(atoms, restart=True, directory=temp_loc)
        vasp2.read_results()

    # Clean up behind us
    finally:
        subprocess.call('rm -r %s' % temp_loc, shell=True)

    return atoms


def __dump_file_to_tmp(file_name):
    '''
    Take the contents of a directory and then dump it into a temporary
    directory while simultaneously unzipping it. This makes reading from
    FireWorks directories faster.

    Arg:
        file_name   String indicating what directory you want to dump
    Returns:
        temp_loc    A string indicating where we just dumped the directory
    '''
    # Make the temporary directory
    temp_loc = '/tmp/%s/' % uuid.uuid4()
    subprocess.call('mkdir %s' % temp_loc, shell=True)

    # Move to the temporary folder and unzip everything
    subprocess.call('tar -C %s -xf %s' % (temp_loc, file_name), shell=True)
    subprocess.call('gunzip -q %s/* > /dev/null' % temp_loc, shell=True)

    return temp_loc


def __get_patched_vasp_settings(fw):
    '''
    Gets the VASP settings we used for a FireWork. Some of them are old and
    have missing information, so this function will also fill in some
    blanks.  If you're using GASpy outside of the Ulissi group, then you
    should probably delete all of the lines past
    `vasp_settings=fw.name['vasp_settings']`.

    Arg:
        fw  Instance of a `fireworks.core.firework.Firework` class that
            should probably get obtained from our Launchpad
    Returts:
        vasp_settings   A dictionary containing the VASP settings we used
                        to perform the relaxation for this FireWork.
    '''
    vasp_settings = fw.name['vasp_settings']

    # Guess the pseudotential version if it's not present
    if 'pp_version' not in vasp_settings:
        if 'arjuna' in fw.launches[-1].fworker.name:
            vasp_settings['pp_version'] = '5.4'
        else:
            vasp_settings['pp_version'] = '5.3.5'
        vasp_settings['pp_guessed'] = True

    # ...I think this is to patch some of our old, incorrectly tagged
    # calculations? I'm not sure. I think that it just assumes that
    # untagged calculations are RPBE.
    if 'gga' not in vasp_settings:
        settings = defaults.xc_settings(xc='rpbe')
        for key in settings:
            vasp_settings[key] = settings[key]

    return vasp_settings


def __get_patched_miller(miller):
    '''
    Some of our old Fireworks have Miller indices stored as strings instead
    of arrays of integers. We fix that here.

    Arg:
        miller          Either a string or a sequence indicating the Miller
                        indices
    Returns:
        patched_miller  The Miller indices formatted as a tuple of integers
    '''
    if isinstance(miller, str):     # Only patch it if the format is incorrect
        patched_miller = tuple([int(index) for index in miller.lstrip('(').rstrip(')').split(',')])
        return patched_miller
    else:
        return miller


def update_catalog_collection(elements, max_miller, workers=1, local_scheduler=True):
    '''
    This function will add enumerate and add adsorption sites to our `catalog`
    Mongo collection.

    Args:
        elements            A list of strings indicating the elements you
                            are looking for, e.g., ['Cu', 'Al']
        max_miller          An integer indicating the maximum Miller index
                            to be enumerated
        workers             An integer indicating how many processes/workers
                            you want executing the tasks and prerequisite
                            tasks.
        local_scheduler     A Boolean indicating whether or not you want to
                            use the local scheduler. Normally we want to
                            use the Luigi daemon (i.e., not a local
                            scheduler), but this function will spawn a
                            heck ton of tasks that might bog down the
                            daemon. And we use atomic writes for our Luigi
                            pickles, so it's less of a risk... and we
                            don't update the catalog often... so it's ok
                            to use a local scheduler here.
    '''
    # Figure out the MPIDs we need to enumerate
    get_mpid_task = _GetMpids(elements=elements)
    run_tasks([get_mpid_task])
    mpids = get_task_output(get_mpid_task)

    # For each MPID, enumerate and add all the sites and then add them to our
    # `catalog` Mongo collection
    insertion_tasks = []
    for mpid in mpids:
        task = _InsertSitesToCatalog(mpid=mpid, max_miller=max_miller)
        insertion_tasks.append(task)
    run_tasks(insertion_tasks, workers=workers, local_scheduler=local_scheduler)


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
                    # Add fingerprint information to the document
                    atoms = make_atoms_from_doc(doc)
                    fingerprint = fingerprint_adslab(atoms)
                    for key, value in fingerprint.items():
                        doc[key] = value

                    # It's faster to write in bulk instead of one-at-a-time, so
                    # save the document to one list that we'll write to
                    inserted_docs.append(doc)
            if not _testing:
                collection.insert_many(inserted_docs)
        save_task_output(self, incumbent_docs + inserted_docs)

    def output(self):
        return make_task_output_object(self)


#class UpdateAllDB(luigi.WrapperTask):
#    '''
#    First, dump from the Primary database to the Auxiliary database.
#    Then, dump from the Auxiliary database to the Local adsorption energy database.
#    Finally, re-request the adsorption energies to re-initialize relaxations & FW submissions.
#    '''
#    # max_processes is the maximum number of calculation sets to Dump If it's set to zero,
#    # then there is no limit. This is used to limit the scope of a DB update for
#    # debugging purposes.
#    max_processes = luigi.IntParameter(0)
#
#    def requires(self):
#        '''
#        Luigi automatically runs the `requires` method whenever we tell it to execute a
#        class. Since we are not truly setting up a dependency (i.e., setting up `requires`,
#        `run`, and `output` methods), we put all of the "action" into the `requires`
#        method.
#        '''
#        # Dump from the Primary DB to the Aux DB
#        DumpToAuxDB().run()
#
#        # Get every doc in the Aux database
#        with gasdb.get_mongo_collection(collection_tag='atoms') as collection:
#            ads_docs = list(collection.find({'type': 'slab+adsorbate'}))
#            surface_energy_docs = list(collection.find({'type': 'slab_surface_energy'}))
#
#        # Get all of the current fwids numbers in the adsorption collection.
#        # Turn the list into a dictionary so that we can parse through it faster.
#        with gasdb.get_mongo_collection('adsorption') as collection:
#            fwids = [doc['processed_data']['FW_info']['slab+adsorbate'] for doc in collection.find()]
#        fwids = dict.fromkeys(fwids)
#
#        with gasdb.get_mongo_collection('surface_energy') as collection:
#            surface_fwids = [doc['processed_data']['FW_info'].values() for doc in collection.find()]
#        surface_fwids = dict.fromkeys([item for sublist in surface_fwids for item in sublist])
#
#        # We are also going to save the doc info for each submitted calc so that we can purge long-standing problems
#        self.surface_energy_docs = []
#        self.ads_docs = []
#
#        # For each adsorbate/configuration and surface energy calc, make a task to write the results to the output
#        # database. We also start a counter, `i`, for how many tasks we've processed.
#        i = 0
#        for doc in surface_energy_docs:
#            # Only make the task if the fireworks task is not already in the database
#            if doc['fwid'] not in surface_fwids:
#                # Pull information from the Aux DB
#                mpid = doc['fwname']['mpid']
#                miller = doc['fwname']['miller']
#                shift = doc['fwname']['shift']
#                keys = ['gga', 'encut', 'zab_vdw', 'lbeefens', 'luse_vdw', 'pp', 'pp_version']
#                settings = OrderedDict()
#                for key in keys:
#                    if key in doc['fwname']['vasp_settings']:
#                        settings[key] = doc['fwname']['vasp_settings'][key]
#
#                # Set the bulk settings correctly, which will differ from the adslab settings
#                # only in the encut
#                settings_bulk = copy.deepcopy(settings)
#                settings_bulk['encut'] = defaults.BULK_ENCUT
#
#                # Create the nested dictionary of information that we will store in the Aux DB
#                parameters = {'bulk': defaults.bulk_parameters(mpid, settings=settings_bulk),
#                              'slab': defaults.slab_parameters(miller=miller,
#                                                               shift=shift,
#                                                               top=True,
#                                                               settings=settings)}
#
#                # default to three points needed for the linear interpolation fit
#                parameters['slab']['slab_surface_energy_num_layers'] = 3
#
#                i += 1
#                if i >= self.max_processes and self.max_processes > 0:
#                    print('Reached the maximum number of processes, %s' % self.max_processes)
#                    break
#
#                #Save the doc for introspection later
#                self.surface_energy_docs.append(doc)
#
#                yield DumpToSurfaceEnergyDB(parameters)
#
#        for doc in ads_docs:
#            # Only make the task if 1) the fireworks task is not already in the database, and
#            # 2) there is an adsorbate
#            if (doc['fwid'] not in fwids and doc['fwname']['adsorbate'] != ''):
#                # Pull information from the Aux DB
#                mpid = doc['fwname']['mpid']
#                miller = doc['fwname']['miller']
#                adsorption_site = doc['fwname']['adsorption_site']
#                if 'adsorbate_rotation' in doc['fwname']:
#                    adsorbate_rotation = doc['fwname']['adsorbate_rotation']
#                else:
#                    adsorbate_rotation = copy.deepcopy(defaults.ROTATION)
#                adsorbate = doc['fwname']['adsorbate']
#                top = doc['fwname']['top']
#                num_slab_atoms = doc['fwname']['num_slab_atoms']
#                slabrepeat = doc['fwname']['slabrepeat']
#                shift = doc['fwname']['shift']
#                keys = ['gga', 'encut', 'zab_vdw', 'lbeefens', 'luse_vdw', 'pp', 'pp_version']
#                settings = OrderedDict()
#                for key in keys:
#                    if key in doc['fwname']['vasp_settings']:
#                        settings[key] = doc['fwname']['vasp_settings'][key]
#
#                #Set the bulk settings correctly, which will differ from the adslab settings
#                # only in the encut
#                settings_bulk = copy.deepcopy(settings)
#                settings_bulk['encut'] = defaults.BULK_ENCUT
#
#                # Create the nested dictionary of information that we will store in the Aux DB
#                parameters = {'bulk': defaults.bulk_parameters(mpid, settings=settings_bulk),
#                              'gas': defaults.gas_parameters(gasname='CO', settings=settings),
#                              'slab': defaults.slab_parameters(miller=miller,
#                                                               shift=shift,
#                                                               top=top,
#                                                               settings=settings),
#                              'adsorption': defaults.adsorption_parameters(adsorbate=adsorbate,
#                                                                           num_slab_atoms=num_slab_atoms,
#                                                                           slabrepeat=slabrepeat,
#                                                                           adsorption_site=adsorption_site,
#                                                                           adsorbate_rotation=adsorbate_rotation,
#                                                                           settings=settings)}
#
#                # If we have duplicates, the FWID might trigger a DumpToAdsorptionDB
#                # even if there is basically an identical calculation in the database
#                DTADB = DumpToAdsorptionDB(parameters)
#                if not(DTADB.complete()):
#                    # If we've hit the maxmum number of processes, flag and stop
#                    i += 1
#                    if i >= self.max_processes and self.max_processes > 0:
#                        print('Reached the maximum number of processes, %s' % self.max_processes)
#                        break
#
#                    #save the doc for introspection later
#                    self.ads_docs.append(doc)
#
#                    yield DumpToAdsorptionDB(parameters)


#class DumpToAdsorptionDB(luigi.Task):
#    ''' This class dumps the adsorption energies from our pickles to our tertiary databases '''
#    parameters = luigi.DictParameter()
#
#    def requires(self):
#        '''
#        We want the lowest energy structure (with adsorption energy), the fingerprinted structure,
#        and the bulk structure
#        '''
#        return [CalculateEnergy(self.parameters),
#                FingerprintRelaxedAdslab(self.parameters),
#                SubmitToFW(calctype='bulk',
#                           parameters={'bulk': self.parameters['bulk']})]
#
#    def run(self):
#        # Load the structure
#        best_sys_pkl = pickle.load(open(self.input()[0].fn, 'rb'))
#        # Extract the atoms object
#        best_sys = best_sys_pkl['atoms']
#        # Get the lowest energy bulk structure
#        bulk = pickle.load(open(self.input()[2].fn, 'rb'))
#        bulkmin = np.argmin([x['results']['energy'] for x in bulk])
#        # Load the fingerprints of the initial and final state
#        fingerprints = pickle.load(open(self.input()[1].fn, 'rb'))
#        fp_final = fingerprints[0]
#        fp_init = fingerprints[1]
#
#        # Create and use tools to calculate the angle between the bond length of the diatomic
#        # adsorbate and the z-direction of the bulk. We are not currently calculating triatomics
#        # or larger.
#        def unit_vector(vector):
#            ''' Returns the unit vector of the vector.  '''
#            return vector / np.linalg.norm(vector)
#
#        def angle_between(v1, v2):
#            ''' Returns the angle in radians between vectors 'v1' and 'v2'::  '''
#            v1_u = unit_vector(v1)
#            v2_u = unit_vector(v2)
#            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
#        if self.parameters['adsorption']['adsorbates'][0]['name'] in ['CO', 'OH']:
#            angle = angle_between(best_sys[-1].position-best_sys[-2].position, best_sys.cell[2])
#            if self.parameters['slab']['top'] is False:
#                angle = np.abs(angle - math.pi)
#        else:
#            angle = 0.
#        angle = angle/2./np.pi*360
#
#        '''
#        Calculate the maximum movement of surface atoms during the relaxation. then we do it again,
#        but for adsorbate atoms.
#        '''
#        # First, calculate the number of adsorbate atoms
#        num_adsorbate_atoms = len(utils.decode_hex_to_atoms(self.parameters['adsorption']['adsorbates'][0]['atoms']))
#
#        # An earlier version of GASpy added the adsorbate to the slab instead of the slab to the
#        # adsorbate. Thus, the indexing for the slabs change. Here, we deal with that.
#        lpad = fwhs.get_launchpad()
#        fw = lpad.get_fw_by_id(best_sys_pkl['slab+ads']['fwid'])
#        # *_start and *_end are the list indices to use when trying to pull out the * from
#        # the adslab atoms.
#        if fw.created_on < datetime(2017, 7, 20):
#            slab_start = None
#            slab_end = -num_adsorbate_atoms
#            ads_start = -num_adsorbate_atoms
#            ads_end = None
#        else:
#            slab_start = num_adsorbate_atoms
#            slab_end = None
#            ads_start = None
#            ads_end = num_adsorbate_atoms
#
#        # Get just the adslab's slab atoms in their initial and final state
#        slab_initial = make_atoms_from_doc(best_sys_pkl['slab+ads']['initial_configuration'])[slab_start:slab_end]
#        slab_final = best_sys[slab_start:slab_end]
#        max_surface_movement = utils.find_max_movement(slab_initial, slab_final)
#        # Repeat the procedure, but for adsorbates
#        adsorbate_initial = make_atoms_from_doc(best_sys_pkl['slab+ads']['initial_configuration'])[ads_start:ads_end]
#        adsorbate_final = best_sys[ads_start:ads_end]
#        max_adsorbate_movement = utils.find_max_movement(adsorbate_initial, adsorbate_final)
#        # Repeat the procedure, but for the relaxed bare slab
#        bare_slab_initial = make_atoms_from_doc(best_sys_pkl['slab']['initial_configuration'])
#        bare_slab_final = make_atoms_from_doc(best_sys_pkl['slab'])
#        max_bare_slab_movement = utils.find_max_movement(bare_slab_initial, bare_slab_final)
#
#
#        # Make a dictionary of tags to add to the database
#        processed_data = {'fp_final': fp_final,
#                          'fp_init': fp_init,
#                          'vasp_settings': self.parameters['adsorption']['vasp_settings'],
#                          'calculation_info': {'type': 'slab+adsorbate',
#                                               'formula': best_sys.get_chemical_formula('hill'),
#                                               'mpid': self.parameters['bulk']['mpid'],
#                                               'miller': self.parameters['slab']['miller'],
#                                               'num_slab_atoms': self.parameters['adsorption']['num_slab_atoms'],
#                                               'top': self.parameters['slab']['top'],
#                                               'slabrepeat': self.parameters['adsorption']['slabrepeat'],
#                                               'relaxed': True,
#                                               'adsorbates': self.parameters['adsorption']['adsorbates'],
#                                               'adsorbate_names': [str(x['name']) for x in self.parameters['adsorption']['adsorbates']],
#                                               'shift': best_sys_pkl['slab+ads']['fwname']['shift']},
#                          'FW_info': {'slab+adsorbate': best_sys_pkl['slab+ads']['fwid'],
#                                      'slab': best_sys_pkl['slab']['fwid'],
#                                      'bulk': bulk[bulkmin]['fwid'],
#                                      'adslab_calculation_date': fw.created_on},
#                          'movement_data': {'max_surface_movement': max_surface_movement,
#                                            'max_adsorbate_movement': max_adsorbate_movement,
#                                            'max_bare_slab_movement': max_bare_slab_movement}}
#        best_sys_pkl_slab_ads = make_doc_from_atoms(best_sys_pkl['atoms'])
#        best_sys_pkl_slab_ads['initial_configuration'] = best_sys_pkl['slab+ads']['initial_configuration']
#        best_sys_pkl_slab_ads['processed_data'] = processed_data
#        # Write the entry into the database
#
#        with gasdb.get_mongo_collection('adsorption') as collection:
#            collection.insert_one(best_sys_pkl_slab_ads)
#
#        # Write a blank token file to indicate this was done so that the entry is not written again
#        with self.output().temporary_path() as self.temp_output_path:
#            with open(self.temp_output_path, 'w') as fhandle:
#                fhandle.write(' ')
#
#    def output(self):
#        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))


#class DumpToSurfaceEnergyDB(luigi.Task):
#    ''' This class dumps the surface energies from our pickles to our tertiary databases '''
#    parameters = luigi.DictParameter()
#
#    def requires(self):
#        '''
#        We want the slabs at various thicknessed from CalculateSlabSurfaceEnergy and
#        the relevant bulk relaxation
#        '''
#        return [CalculateSlabSurfaceEnergy(self.parameters),
#                SubmitToFW(calctype='bulk',
#                           parameters={'bulk': self.parameters['bulk']})]
#
#    def run(self):
#
#        # Load the structures
#        surface_energy_pkl = pickle.load(open(self.input()[0].fn, 'rb'))
#
#        # Extract the atoms object
#        surface_energy_atoms = surface_energy_pkl[0]['atoms']
#
#        # Get the lowest energy bulk structure
#        bulk = pickle.load(open(self.input()[1].fn, 'rb'))
#        bulkmin = np.argmin([x['results']['energy'] for x in bulk])
#
#        # Calculate the movement for each relaxed slab
#        max_surface_movement = [utils.find_max_movement(doc['atoms'], make_atoms_from_doc(doc['initial_configuration']))
#                                for doc in surface_energy_pkl]
#
#        # Make a dictionary of tags to add to the database
#        processed_data = {'vasp_settings': self.parameters['slab']['vasp_settings'],
#                          'calculation_info': {'type': 'slab_surface_energy',
#                                               'formula': surface_energy_atoms.get_chemical_formula('hill'),
#                                               'mpid': self.parameters['bulk']['mpid'],
#                                               'miller': self.parameters['slab']['miller'],
#                                               'num_slab_atoms': len(surface_energy_atoms),
#                                               'relaxed': True,
#                                               'shift': surface_energy_pkl[0]['fwname']['shift']},
#                          'FW_info': surface_energy_pkl[0]['processed_data']['FW_info'],
#                          'surface_energy_info': surface_energy_pkl[0]['processed_data']['surface_energy_info'],
#                          'movement_data': {'max_surface_movement': max_surface_movement}}
#        processed_data['FW_info']['bulk'] = bulk[bulkmin]['fwid']
#        surface_energy_pkl_slab = make_doc_from_atoms(surface_energy_atoms)
#        surface_energy_pkl_slab['initial_configuration'] = surface_energy_pkl[0]['initial_configuration']
#        surface_energy_pkl_slab['processed_data'] = processed_data
#
#        # Write the entry into the database
#        with gasdb.get_mongo_collection('surface_energy') as collection:
#            collection.insert_one(surface_energy_pkl_slab)
#
#        # Write a blank token file to indicate this was done so that the entry is not written again
#        with self.output().temporary_path() as self.temp_output_path:
#            with open(self.temp_output_path, 'w') as fhandle:
#                fhandle.write(' ')
#
#    def output(self):
#        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))
