'''
This module houses various tasks to manage our databases.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import copy
import math
from datetime import datetime
from collections import OrderedDict
import pickle
import numpy as np
import luigi
import tqdm
import multiprocess as mp
from .fireworks_submitters import SubmitToFW
from .metadata_calculators import (FingerprintRelaxedAdslab,
                                   FingerprintUnrelaxedAdslabs,
                                   CalculateEnergy,
                                   CalculateSlabSurfaceEnergy)
from ..mongo import make_doc_from_atoms, make_atoms_from_doc
from .. import defaults, utils, gasdb
from .. import fireworks_helper_scripts as fwhs


GASDB_PATH = utils.read_rc('gasdb_path')


class UpdateAllDB(luigi.WrapperTask):
    '''
    First, dump from the Primary database to the Auxiliary database.
    Then, dump from the Auxiliary database to the Local adsorption energy database.
    Finally, re-request the adsorption energies to re-initialize relaxations & FW submissions.
    '''
    # max_processes is the maximum number of calculation sets to Dump If it's set to zero,
    # then there is no limit. This is used to limit the scope of a DB update for
    # debugging purposes.
    max_processes = luigi.IntParameter(0)

    def requires(self):
        '''
        Luigi automatically runs the `requires` method whenever we tell it to execute a
        class. Since we are not truly setting up a dependency (i.e., setting up `requires`,
        `run`, and `output` methods), we put all of the "action" into the `requires`
        method.
        '''
        # Dump from the Primary DB to the Aux DB
        DumpToAuxDB().run()

        # Get every doc in the Aux database
        with gasdb.get_mongo_collection(collection_tag='atoms') as collection:
            ads_docs = list(collection.find({'type': 'slab+adsorbate'}))
            surface_energy_docs = list(collection.find({'type': 'slab_surface_energy'}))

        # Get all of the current fwids numbers in the adsorption collection.
        # Turn the list into a dictionary so that we can parse through it faster.
        with gasdb.get_mongo_collection('adsorption') as collection:
            fwids = [doc['processed_data']['FW_info']['slab+adsorbate'] for doc in collection.find()]
        fwids = dict.fromkeys(fwids)

        with gasdb.get_mongo_collection('surface_energy') as collection:
            surface_fwids = [doc['processed_data']['FW_info'].values() for doc in collection.find()]
        surface_fwids = dict.fromkeys([item for sublist in surface_fwids for item in sublist])

        # We are also going to save the doc info for each submitted calc so that we can purge long-standing problems
        self.surface_energy_docs = []
        self.ads_docs = []

        # For each adsorbate/configuration and surface energy calc, make a task to write the results to the output
        # database. We also start a counter, `i`, for how many tasks we've processed.
        i = 0
        for doc in surface_energy_docs:
            # Only make the task if the fireworks task is not already in the database
            if doc['fwid'] not in surface_fwids:
                # Pull information from the Aux DB
                mpid = doc['fwname']['mpid']
                miller = doc['fwname']['miller']
                shift = doc['fwname']['shift']
                keys = ['gga', 'encut', 'zab_vdw', 'lbeefens', 'luse_vdw', 'pp', 'pp_version']
                settings = OrderedDict()
                for key in keys:
                    if key in doc['fwname']['vasp_settings']:
                        settings[key] = doc['fwname']['vasp_settings'][key]

                # Set the bulk settings correctly, which will differ from the adslab settings
                # only in the encut
                settings_bulk = copy.deepcopy(settings)
                settings_bulk['encut'] = defaults.BULK_ENCUT

                # Create the nested dictionary of information that we will store in the Aux DB
                parameters = {'bulk': defaults.bulk_parameters(mpid, settings=settings_bulk),
                              'slab': defaults.slab_parameters(miller=miller,
                                                               shift=shift,
                                                               top=True,
                                                               settings=settings)}

                # default to three points needed for the linear interpolation fit
                parameters['slab']['slab_surface_energy_num_layers'] = 3

                i += 1
                if i >= self.max_processes and self.max_processes > 0:
                    print('Reached the maximum number of processes, %s' % self.max_processes)
                    break

                #Save the doc for introspection later
                self.surface_energy_docs.append(doc)

                yield DumpToSurfaceEnergyDB(parameters)

        for doc in ads_docs:
            # Only make the task if 1) the fireworks task is not already in the database, and
            # 2) there is an adsorbate
            if (doc['fwid'] not in fwids and doc['fwname']['adsorbate'] != ''):
                # Pull information from the Aux DB
                mpid = doc['fwname']['mpid']
                miller = doc['fwname']['miller']
                adsorption_site = doc['fwname']['adsorption_site']
                if 'adsorbate_rotation' in doc['fwname']:
                    adsorbate_rotation = doc['fwname']['adsorbate_rotation']
                else:
                    adsorbate_rotation = copy.deepcopy(defaults.ROTATION)
                adsorbate = doc['fwname']['adsorbate']
                top = doc['fwname']['top']
                num_slab_atoms = doc['fwname']['num_slab_atoms']
                slabrepeat = doc['fwname']['slabrepeat']
                shift = doc['fwname']['shift']
                keys = ['gga', 'encut', 'zab_vdw', 'lbeefens', 'luse_vdw', 'pp', 'pp_version']
                settings = OrderedDict()
                for key in keys:
                    if key in doc['fwname']['vasp_settings']:
                        settings[key] = doc['fwname']['vasp_settings'][key]

                #Set the bulk settings correctly, which will differ from the adslab settings
                # only in the encut
                settings_bulk = copy.deepcopy(settings)
                settings_bulk['encut'] = defaults.BULK_ENCUT

                # Create the nested dictionary of information that we will store in the Aux DB
                parameters = {'bulk': defaults.bulk_parameters(mpid, settings=settings_bulk),
                              'gas': defaults.gas_parameters(gasname='CO', settings=settings),
                              'slab': defaults.slab_parameters(miller=miller,
                                                               shift=shift,
                                                               top=top,
                                                               settings=settings),
                              'adsorption': defaults.adsorption_parameters(adsorbate=adsorbate,
                                                                           num_slab_atoms=num_slab_atoms,
                                                                           slabrepeat=slabrepeat,
                                                                           adsorption_site=adsorption_site,
                                                                           adsorbate_rotation=adsorbate_rotation,
                                                                           settings=settings)}

                # If we have duplicates, the FWID might trigger a DumpToAdsorptionDB
                # even if there is basically an identical calculation in the database
                DTADB = DumpToAdsorptionDB(parameters)
                if not(DTADB.complete()):
                    # If we've hit the maxmum number of processes, flag and stop
                    i += 1
                    if i >= self.max_processes and self.max_processes > 0:
                        print('Reached the maximum number of processes, %s' % self.max_processes)
                        break

                    #save the doc for introspection later
                    self.ads_docs.append(doc)

                    yield DumpToAdsorptionDB(parameters)


class UpdateEnumerations(luigi.Task):
    '''
    This class re-requests the enumeration of adsorption sites to re-initialize our various
    generating functions. It then dumps any completed site enumerations into our Local DB
    for adsorption sites.
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        ''' Get the generated adsorbate configurations '''
        if 'unrelaxed' in self.parameters and self.parameters['unrelaxed'] == 'relaxed_bulk':
            return [FingerprintUnrelaxedAdslabs(self.parameters),
                    SubmitToFW(calctype='bulk',
                               parameters={'bulk': self.parameters['bulk']})]
        else:
            return [FingerprintUnrelaxedAdslabs(self.parameters)]

    def run(self):
        # Find the unique configurations based on the fingerprint of each site
        configs = pickle.load(open(self.input()[0].fn, 'rb'))
        unq_configs, unq_inds = np.unique([str([x['shift'],
                                                x['fp']['coordination'],
                                                x['fp']['neighborcoord']]) for x in
                                           configs],
                                          return_index=True)

        if 'unrelaxed' in self.parameters and self.parameters['unrelaxed'] == 'relaxed_bulk':
            bulk = pickle.load(open(self.input()[1].fn, 'rb'))
            bulkmin = np.argmin([x['results']['energy'] for x in bulk])
            FW_info = bulk[bulkmin]['fwid']
            vasp_settings = bulk[bulkmin]['fwname']['vasp_settings']
        else:
            FW_info = ''
            vasp_settings = ''

        # For each configuration, write a doc to the database
        doclist = []
        for i in unq_inds:
            config = configs[i]
            atoms = config['atoms']
            slabadsdoc = make_doc_from_atoms(config['atoms'])
            processed_data = {'fp_init': config['fp'],
                              'calculation_info': {'type': 'slab+adsorbate',
                                                   'formula': atoms.get_chemical_formula('hill'),
                                                   'mpid': self.parameters['bulk']['mpid'],
                                                   'miller': config['miller'],
                                                   'num_slab_atoms': len(atoms)-len(config['adsorbate']),
                                                   'top': config['top'],
                                                   'slabrepeat': config['slabrepeat'],
                                                   'relaxed': False,
                                                   'adsorbate': config['adsorbate'],
                                                   'shift': config['shift'],
                                                   'adsorption_site': config['adsorption_site']},
                               'vasp_settings': vasp_settings,
                               'FW_info': {'bulk': FW_info}}
            slabadsdoc['processed_data'] = processed_data
            slabadsdoc['predictions'] = {'adsorption_energy': {}}
            doclist.append(slabadsdoc)

        if ('unrelaxed' in self.parameters) and self.parameters['unrelaxed'] == 'relaxed_bulk':
            with gasdb.get_mongo_collection('relaxed_bulk_catalog') as collection:
                collection.insert_many(doclist)
        else:
            with gasdb.get_mongo_collection('catalog') as collection:
                collection.insert_many(doclist)

        # Write a token file to indicate this task has been completed and added to the DB
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(' ')

    def output(self):
        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))


class DumpToAuxDB(luigi.Task):
    '''
    This class will load the results for the relaxations from the Primary FireWorks
    database into the Auxiliary vasp.mongo database.
    '''
    num_procs = luigi.IntParameter(4)

    def run(self):
        lpad = fwhs.get_launchpad()

        # Get all of the FW numbers that have been loaded into the atoms collection already.
        # We turn the list into a dictionary so that we can parse through it more quickly.
        with gasdb.get_mongo_collection('atoms') as collection:
            atoms_fws = [a['fwid'] for a in collection.find({'fwid': {'$exists': True}})]
        atoms_fws = dict.fromkeys(atoms_fws)

        # Get all of the completed fireworks from the Primary DB
        fws_cmpltd = lpad.get_fw_ids({'state': 'COMPLETED',
                                      'name.calculation_type': 'unit cell optimization'}) + \
            lpad.get_fw_ids({'state': 'COMPLETED',
                             'name.calculation_type': 'gas phase optimization'}) + \
            lpad.get_fw_ids({'state': 'COMPLETED',
                             'name.calculation_type': 'slab optimization',
                             'name.shift': {'$exists': True}}) + \
            lpad.get_fw_ids({'state': 'COMPLETED',
                             'name.calculation_type': 'slab+adsorbate optimization',
                             'name.shift': {'$exists': True}}) + \
            lpad.get_fw_ids({'state': 'COMPLETED',
                             'name.calculation_type': 'slab_surface_energy optimization',
                             'name.shift': {'$exists': True}})

        # For each fireworks object, turn the results into a mongo doc so that we can
        # dump the mongo doc into the Aux DB.
        def process_fwid(fwid):
            if fwid not in atoms_fws:
                # Get the information from the class we just pulled from the launchpad.
                # Move on if we fail to get the info.
                fw = lpad.get_fw_by_id(fwid)
                try:
                    atoms, starting_atoms, trajectory, vasp_settings = fwhs.get_firework_info(fw)
                except RuntimeError:
                    return

                # In an older version of GASpy, we did not use tags to identify
                # whether an atom was part of the slab or an adsorbate. Here, we
                # add the tags back in.
                if (fw.created_on < datetime(2017, 7, 20) and
                        fw.name['calculation_type'] == 'slab+adsorbate optimization'):
                    # In this old version, the adsorbates were added onto the slab.
                    # Thus, the slab atoms came before the adsorbate atoms in
                    # the indexing. We use this information to create the tags list.
                    n_ads_atoms = len(fw.name['adsorbate'])
                    n_slab_atoms = len(atoms) - n_ads_atoms
                    tags = [0]*n_slab_atoms
                    tags.extend([1]*n_ads_atoms)
                    # Now set the tags for the atoms
                    atoms.set_tags(tags)
                    starting_atoms.set_tags(tags)

                # The VASP calculator, when used with ASE optimization, was
                # incorrectly recording the internal forces in atoms objects
                # with the stored forces including constraints. If such
                # incompatible constraints exist and the calculations occured
                # before the switch to the Vasp2 calculator, we should get the
                # correct (VASP) forces from a backup of the directory which
                # includes the INCAR, ase-sort.dat, etc files
                allowable_constraints = ['FixAtoms']
                constraint_not_allowable = [constraint.todict()['name'] not in allowable_constraints
                                            for constraint in atoms.constraints]
                vasp_incompatible_constraints = np.any(constraint_not_allowable)
                if (fw.created_on < datetime(2018, 12, 1) and vasp_incompatible_constraints):
                    atoms = utils.get_final_atoms_object_with_vasp_forces(fw.launches[-1].launch_id)

                # Initialize the mongo document, doc, and the populate it with the fw info
                doc = make_doc_from_atoms(atoms)
                doc['initial_configuration'] = make_doc_from_atoms(starting_atoms)
                doc['fwname'] = fw.name
                doc['fwid'] = fwid
                doc['directory'] = fw.launches[-1].launch_dir
                if fw.name['calculation_type'] == 'unit cell optimization':
                    doc['type'] = 'bulk'
                elif fw.name['calculation_type'] == 'gas phase optimization':
                    doc['type'] = 'gas'
                elif fw.name['calculation_type'] == 'slab optimization':
                    doc['type'] = 'slab'
                elif fw.name['calculation_type'] == 'slab_surface_energy optimization':
                    doc['type'] = 'slab_surface_energy'
                elif fw.name['calculation_type'] == 'slab+adsorbate optimization':
                    doc['type'] = 'slab+adsorbate'

                # Convert the miller indices from strings to integers
                if 'miller' in fw.name:
                    if isinstance(fw.name['miller'], str):
                        doc['fwname']['miller'] = eval(doc['fwname']['miller'])

                return doc

        with mp.Pool(self.num_procs) as pool:
            fwids_to_process = [fwid for fwid in fws_cmpltd if fwid not in atoms_fws]
            docs = list(tqdm.tqdm(pool.imap(process_fwid, fwids_to_process, chunksize=100), total=len(fwids_to_process)))

        docs_not_none = [doc for doc in docs if doc is not None]
        if len(docs_not_none) > 0:
            with gasdb.get_mongo_collection('atoms') as collection:
                collection.insert_many(docs_not_none)

    def output(self):
        return luigi.LocalTarget(GASDB_PATH+'/DumpToAuxDB.token')


class DumpToAdsorptionDB(luigi.Task):
    ''' This class dumps the adsorption energies from our pickles to our tertiary databases '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        We want the lowest energy structure (with adsorption energy), the fingerprinted structure,
        and the bulk structure
        '''
        return [CalculateEnergy(self.parameters),
                FingerprintRelaxedAdslab(self.parameters),
                SubmitToFW(calctype='bulk',
                           parameters={'bulk': self.parameters['bulk']})]

    def run(self):
        # Load the structure
        best_sys_pkl = pickle.load(open(self.input()[0].fn, 'rb'))
        # Extract the atoms object
        best_sys = best_sys_pkl['atoms']
        # Get the lowest energy bulk structure
        bulk = pickle.load(open(self.input()[2].fn, 'rb'))
        bulkmin = np.argmin([x['results']['energy'] for x in bulk])
        # Load the fingerprints of the initial and final state
        fingerprints = pickle.load(open(self.input()[1].fn, 'rb'))
        fp_final = fingerprints[0]
        fp_init = fingerprints[1]

        # Create and use tools to calculate the angle between the bond length of the diatomic
        # adsorbate and the z-direction of the bulk. We are not currently calculating triatomics
        # or larger.
        def unit_vector(vector):
            ''' Returns the unit vector of the vector.  '''
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            ''' Returns the angle in radians between vectors 'v1' and 'v2'::  '''
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if self.parameters['adsorption']['adsorbates'][0]['name'] in ['CO', 'OH']:
            angle = angle_between(best_sys[-1].position-best_sys[-2].position, best_sys.cell[2])
            if self.parameters['slab']['top'] is False:
                angle = np.abs(angle - math.pi)
        else:
            angle = 0.
        angle = angle/2./np.pi*360

        '''
        Calculate the maximum movement of surface atoms during the relaxation. then we do it again,
        but for adsorbate atoms.
        '''
        # First, calculate the number of adsorbate atoms
        num_adsorbate_atoms = len(utils.decode_hex_to_atoms(self.parameters['adsorption']['adsorbates'][0]['atoms']))

        # An earlier version of GASpy added the adsorbate to the slab instead of the slab to the
        # adsorbate. Thus, the indexing for the slabs change. Here, we deal with that.
        lpad = fwhs.get_launchpad()
        fw = lpad.get_fw_by_id(best_sys_pkl['slab+ads']['fwid'])
        # *_start and *_end are the list indices to use when trying to pull out the * from
        # the adslab atoms.
        if fw.created_on < datetime(2017, 7, 20):
            slab_start = None
            slab_end = -num_adsorbate_atoms
            ads_start = -num_adsorbate_atoms
            ads_end = None
        else:
            slab_start = num_adsorbate_atoms
            slab_end = None
            ads_start = None
            ads_end = num_adsorbate_atoms

        # Get just the adslab's slab atoms in their initial and final state
        slab_initial = make_atoms_from_doc(best_sys_pkl['slab+ads']['initial_configuration'])[slab_start:slab_end]
        slab_final = best_sys[slab_start:slab_end]
        max_surface_movement = utils.find_max_movement(slab_initial, slab_final)
        # Repeat the procedure, but for adsorbates
        adsorbate_initial = make_atoms_from_doc(best_sys_pkl['slab+ads']['initial_configuration'])[ads_start:ads_end]
        adsorbate_final = best_sys[ads_start:ads_end]
        max_adsorbate_movement = utils.find_max_movement(adsorbate_initial, adsorbate_final)
        # Repeat the procedure, but for the relaxed bare slab
        bare_slab_initial = make_atoms_from_doc(best_sys_pkl['slab']['initial_configuration'])
        bare_slab_final = make_atoms_from_doc(best_sys_pkl['slab'])
        max_bare_slab_movement = utils.find_max_movement(bare_slab_initial, bare_slab_final)


        # Make a dictionary of tags to add to the database
        processed_data = {'fp_final': fp_final,
                          'fp_init': fp_init,
                          'vasp_settings': self.parameters['adsorption']['vasp_settings'],
                          'calculation_info': {'type': 'slab+adsorbate',
                                               'formula': best_sys.get_chemical_formula('hill'),
                                               'mpid': self.parameters['bulk']['mpid'],
                                               'miller': self.parameters['slab']['miller'],
                                               'num_slab_atoms': self.parameters['adsorption']['num_slab_atoms'],
                                               'top': self.parameters['slab']['top'],
                                               'slabrepeat': self.parameters['adsorption']['slabrepeat'],
                                               'relaxed': True,
                                               'adsorbates': self.parameters['adsorption']['adsorbates'],
                                               'adsorbate_names': [str(x['name']) for x in self.parameters['adsorption']['adsorbates']],
                                               'shift': best_sys_pkl['slab+ads']['fwname']['shift']},
                          'FW_info': {'slab+adsorbate': best_sys_pkl['slab+ads']['fwid'],
                                      'slab': best_sys_pkl['slab']['fwid'],
                                      'bulk': bulk[bulkmin]['fwid'],
                                      'adslab_calculation_date': fw.created_on},
                          'movement_data': {'max_surface_movement': max_surface_movement,
                                            'max_adsorbate_movement': max_adsorbate_movement,
                                            'max_bare_slab_movement': max_bare_slab_movement}}
        best_sys_pkl_slab_ads = make_doc_from_atoms(best_sys_pkl['atoms'])
        best_sys_pkl_slab_ads['initial_configuration'] = best_sys_pkl['slab+ads']['initial_configuration']
        best_sys_pkl_slab_ads['processed_data'] = processed_data
        # Write the entry into the database

        with gasdb.get_mongo_collection('adsorption') as collection:
            collection.insert_one(best_sys_pkl_slab_ads)

        # Write a blank token file to indicate this was done so that the entry is not written again
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(' ')

    def output(self):
        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))


class DumpToSurfaceEnergyDB(luigi.Task):
    ''' This class dumps the surface energies from our pickles to our tertiary databases '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        We want the slabs at various thicknessed from CalculateSlabSurfaceEnergy and
        the relevant bulk relaxation
        '''
        return [CalculateSlabSurfaceEnergy(self.parameters),
                SubmitToFW(calctype='bulk',
                           parameters={'bulk': self.parameters['bulk']})]

    def run(self):

        # Load the structures
        surface_energy_pkl = pickle.load(open(self.input()[0].fn, 'rb'))

        # Extract the atoms object
        surface_energy_atoms = surface_energy_pkl[0]['atoms']

        # Get the lowest energy bulk structure
        bulk = pickle.load(open(self.input()[1].fn, 'rb'))
        bulkmin = np.argmin([x['results']['energy'] for x in bulk])

        # Calculate the movement for each relaxed slab
        max_surface_movement = [utils.find_max_movement(doc['atoms'], make_atoms_from_doc(doc['initial_configuration']))
                                for doc in surface_energy_pkl]

        # Make a dictionary of tags to add to the database
        processed_data = {'vasp_settings': self.parameters['slab']['vasp_settings'],
                          'calculation_info': {'type': 'slab_surface_energy',
                                               'formula': surface_energy_atoms.get_chemical_formula('hill'),
                                               'mpid': self.parameters['bulk']['mpid'],
                                               'miller': self.parameters['slab']['miller'],
                                               'num_slab_atoms': len(surface_energy_atoms),
                                               'relaxed': True,
                                               'shift': surface_energy_pkl[0]['fwname']['shift']},
                          'FW_info': surface_energy_pkl[0]['processed_data']['FW_info'],
                          'surface_energy_info': surface_energy_pkl[0]['processed_data']['surface_energy_info'],
                          'movement_data': {'max_surface_movement': max_surface_movement}}
        processed_data['FW_info']['bulk'] = bulk[bulkmin]['fwid']
        surface_energy_pkl_slab = make_doc_from_atoms(surface_energy_atoms)
        surface_energy_pkl_slab['initial_configuration'] = surface_energy_pkl[0]['initial_configuration']
        surface_energy_pkl_slab['processed_data'] = processed_data

        # Write the entry into the database
        with gasdb.get_mongo_collection('surface_energy') as collection:
            collection.insert_one(surface_energy_pkl_slab)

        # Write a blank token file to indicate this was done so that the entry is not written again
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(' ')

    def output(self):
        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))
